# Copyright (c)  2024  Xiaomi Corporation (sherpa-onnx, Apache-2.0)
# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
#
# Offline speaker diarization of sherpa-onnx v1.13.2 (sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h,
# fast-clustering.cc, offline-speaker-diarization-result.cc, speaker-embedding-extractor-general-impl.h), Apache-2.0,
# reimplemented in PyTorch: the pyannote segmentation-3.0 and 3D-Speaker ERes2Net models run natively under MMGP,
# the Kaldi filterbank features of kaldi-native-fbank (Apache-2.0) are computed on the GPU. Same windows, powerset
# decoding, per-chunk embeddings, complete-linkage clustering and segment post-processing as sherpa-onnx.
"""Who speaks when: [(start_s, end_s, speaker)] for 16 kHz mono audio."""
import contextlib
import threading

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mmgp import offload, runtime
from shared.utils import files_locator as fl
from shared.utils import offload_registry
from shared.utils.media_control import inference_checkpoint
from .assets import EMBEDDING_FILE, SEGMENTATION_FILE

SAMPLE_RATE = 16000
WINDOW_SIZE, WINDOW_SHIFT = 160000, 16000  # sherpa-onnx: 10 s windows every tenth of a window
PYANNET_SHIFT, PYANNET_FIELD = 270, 991  # receptive field of one segmentation frame, in samples
CHUNK_BATCH = 32  # segmentation windows per forward: 20 MB of audio
_OWNER = "speaker_separation"
_lock = threading.RLock()
# Powerset classes of segmentation-3.0 -> local speakers: none, 0, 1, 2, (0, 1), (0, 2), (1, 2)
_POWERSET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int32)


def _load():
    from accelerate import init_empty_weights
    from .eres2net import ERes2Net
    from .pyannet import PyanNet
    if not torch.cuda.is_available():
        raise RuntimeError("Speaker Separation Requires a CUDA GPU.")
    tqdm.write("Loading - Speaker Separation Models")
    with init_empty_weights(include_buffers=True):
        segmentation, embedding = PyanNet(), ERes2Net()
    for model, filename in ((segmentation, SEGMENTATION_FILE), (embedding, EMBEDDING_FILE)):
        offload.load_model_data(model, fl.locate_file(filename), writable_tensors=False, default_dtype=None)
        model._convertWeightsFloatTo = None
        model.eval().requires_grad_(False)
    modules = {"speaker_segmentation": segmentation, "speaker_embedding": embedding}
    previous_device = torch.get_default_device()
    try:
        manager = offload.profile(modules, 0, budgets={name: 0 for name in modules}, quantizeTransformer=False)
    finally:
        torch.set_default_device(None if previous_device.type == "cpu" else previous_device)
    offload_registry.register_offloadobj(_OWNER, manager, manager.release)
    return segmentation, embedding, manager


def _release(manager):
    offload_registry.unregister_offloadobj(_OWNER, manager)
    manager.release()


@contextlib.contextmanager
def _full_fp32():
    """FP32 convolutions, LSTM and matmuls without TF32 (other models enable it globally): TF32 flips frame decisions of
    the segmentation model as BF16 does, and the clustering then changes whole speaker turns."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def _segment(model, audio):
    """Powerset labels per window: list of (frames, 3) 0/1 arrays, sherpa's windows (the last one zero-padded)."""
    n, window = audio.numel(), WINDOW_SIZE
    if n <= window:
        chunks = [F.pad(audio, (0, window - n))[None]]
    else:
        num_chunks = (n - window) // WINDOW_SHIFT + 1
        chunks = list(audio[:window + (num_chunks - 1) * WINDOW_SHIFT].unfold(0, window, WINDOW_SHIFT).split(CHUNK_BATCH))
        if (n - window) % WINDOW_SHIFT:
            chunks.append(F.pad(audio[num_chunks * WINDOW_SHIFT:], (0, window - (n - num_chunks * WINDOW_SHIFT)))[None])
    dtype = model.classifier.weight.dtype
    labels = []
    for batch in chunks:
        classes = model(batch[:, None].to(dtype=dtype, memory_format=torch.contiguous_format)).argmax(-1).cpu().numpy()
        labels.extend(_POWERSET[classes])
    return labels


def _povey_window(device):
    return (0.5 - 0.5 * torch.cos(2 * torch.pi * torch.arange(400, device=device, dtype=torch.float64) / 399)).pow(0.85).float()


def _mel_banks(device):
    # Kaldi MelBanks: 80 triangular bins in the 1127 ln(1 + f / 700) mel scale, 20 Hz - 7600 Hz, 512-point FFT
    mel = lambda f: 1127.0 * np.log1p(np.asarray(f, dtype=np.float32) / np.float32(700.0))
    low, high = mel(20.0), mel(8000.0 - 400.0)
    delta = (high - low) / np.float32(81)
    freqs = mel(np.float32(31.25) * np.arange(256, dtype=np.float32))
    left = low + np.arange(80, dtype=np.float32)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    banks = np.where((freqs > left) & (freqs < right), np.where(freqs <= center, (freqs - left) / (center - left), (right - freqs) / (right - center)), 0.0)
    return torch.from_numpy(banks.astype(np.float32)).to(device)


def _fbank(wave, window, banks):
    """Kaldi fbank of kaldi-native-fbank / sherpa-onnx: 25 ms frames every 10 ms, edges reflected (snip_edges false),
    DC removed, pre-emphasis 0.97, Povey window, power spectrum, 80 log mel energies."""
    n = wave.numel()
    frames = (n + 80) // 160
    index = torch.arange(frames, device=wave.device)[:, None] * 160 - 120 + torch.arange(400, device=wave.device)
    index = torch.where(index < 0, -index - 1, index)
    index = torch.where(index >= n, 2 * n - 1 - index, index)
    x = wave[index]
    x = x - x.mean(dim=1, keepdim=True)
    x = torch.cat((x[:, :1] * (1 - 0.97), x[:, 1:] - 0.97 * x[:, :-1]), dim=1) * window
    spectrum = torch.fft.rfft(F.pad(x, (0, 112)))
    power = spectrum.real.square() + spectrum.imag.square()
    return torch.log((power[:, :256] @ banks.T).clamp_min(torch.finfo(torch.float32).eps))


def _sample_indexes(labels):
    """(chunk, local speaker) pairs active in at least 10 frames without overlap, with their sample ranges."""
    pairs, ranges = [], []
    for chunk_index, label in enumerate(labels):
        label = np.where((label.sum(axis=1) < 2)[:, None], label, 0)
        num_frames = label.shape[0]
        offset = chunk_index * WINDOW_SHIFT
        to_sample = lambda frame: int(np.float32(frame) / np.float32(num_frames) * np.float32(WINDOW_SIZE) + np.float32(offset))
        for speaker in range(label.shape[1]):
            active = label[:, speaker]
            if active.sum() < 10:
                continue
            edges = np.diff(np.concatenate(([0], active, [0])))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            ends = np.where(ends == num_frames, num_frames - 1, ends)
            pairs.append((chunk_index, speaker))
            ranges.append([(to_sample(s), to_sample(e)) for s, e in zip(starts, ends)])
    return pairs, ranges


def _embeddings(model, audio, ranges, checkpoint):
    window, banks = _povey_window(audio.device), _mel_banks(audio.device)
    dtype = model.seg_1.weight.dtype
    rows, valid = [], []
    for index, segments in enumerate(ranges):
        checkpoint()
        wave = torch.cat([audio[start:min(end, audio.numel())] for start, end in segments if min(end, audio.numel()) > start])
        features = _fbank(wave, window, banks)
        embedding = model((features - features.mean(dim=0, keepdim=True))[None].to(dtype)).float()[0]
        if not torch.isnan(embedding).any():
            rows.append(embedding)
            valid.append(index)
    return (torch.stack(rows).cpu().numpy() if rows else np.zeros((0, 512), np.float32)), valid


def _cluster(embeddings, num_clusters, threshold):
    """sherpa-onnx FastClustering: complete linkage on cosine distances, cut into num_clusters (or at threshold)."""
    count = embeddings.shape[0]
    if count <= 1:
        return np.zeros(count, dtype=np.int64)
    from scipy.cluster.hierarchy import cut_tree, fcluster, linkage
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    rows, cols = np.triu_indices(count, k=1)
    distances = np.maximum(1.0 - np.einsum("ij,ij->i", unit[rows].astype(np.float64), unit[cols].astype(np.float64)), 0.0)
    tree = linkage(distances, method="complete")
    labels = cut_tree(tree, n_clusters=min(num_clusters, count))[:, 0] if num_clusters > 0 else fcluster(tree, threshold, criterion="distance") - 1
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def _overlap_add(labels, columns):
    num_frames = (WINDOW_SIZE + (len(labels) - 1) * WINDOW_SHIFT) // PYANNET_SHIFT + 1
    total = np.zeros((num_frames, columns), dtype=np.int64)
    weight = np.zeros(num_frames, dtype=np.float32)
    for index, label in enumerate(labels):
        start = int(float(np.float32(index) * np.float32(WINDOW_SHIFT) / np.float32(PYANNET_SHIFT)) + 0.5)
        total[start:start + label.shape[0]] += label
        weight[start:start + label.shape[0]] += 1
    return total, weight



def _segments(final_labels, min_duration_on, min_duration_off):
    scale, offset = np.float32(PYANNET_SHIFT / SAMPLE_RATE), np.float32(0.5 * PYANNET_FIELD / SAMPLE_RATE)
    result = []
    num_frames = final_labels.shape[0]
    for speaker in range(final_labels.shape[1]):
        active = final_labels[:, speaker] > 0
        edges = np.diff(np.concatenate(([False], active, [False])).astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        ends = np.where(ends == num_frames, num_frames - 1, ends)
        segments = [[float(np.float32(s) * scale + offset), float(np.float32(e) * scale + offset)] for s, e in zip(starts, ends)]
        merged = []
        for segment in segments:  # gaps no longer than min_duration_off are closed
            if merged and merged[-1][1] < segment[0] and merged[-1][1] + min_duration_off >= segment[0]:
                merged[-1][1] = segment[1]
            else:
                merged.append(segment)
        result.extend((start, end, speaker) for start, end in merged if end - start > min_duration_on)
    return sorted(result, key=lambda item: item[0])


@torch.inference_mode()
def diarize(audio, num_clusters=2, threshold=0.5, min_duration_on=0.3, min_duration_off=0.5, checkpoint=inference_checkpoint):
    """audio: 16 kHz mono float32 array. Returns [(start_s, end_s, speaker_index)] sorted by start. checkpoint raises on abort."""
    if audio.size == 0:
        return []
    with _lock:
        segmentation, embedding, manager = _load()
        try:
            with runtime.execution_scope(checkpoint), _full_fp32():
                device = runtime.execution_device(segmentation)
                samples = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(device)
                labels = _segment(segmentation, samples)
                checkpoint()
                if len(labels) == 1:
                    return _segments(labels[0], min_duration_on, min_duration_off)
                counts, weight = _overlap_add(labels, 3)
                speakers_per_frame = (counts.sum(axis=1).astype(np.float32) / (weight + np.float32(1e-12)) + np.float32(0.5)).astype(np.int32)
                if speakers_per_frame.max() == 0:
                    return []
                pairs, ranges = _sample_indexes(labels)
                vectors, valid = _embeddings(embedding, samples, ranges, checkpoint)
                del samples
        finally:
            _release(manager)
    pairs = [pairs[i] for i in valid]
    clusters = _cluster(vectors, num_clusters, threshold)
    if clusters.size == 0:
        return []
    to_cluster = dict(zip(pairs, clusters))
    columns = int(clusters.max()) + 1
    relabeled = []
    for chunk_index, label in enumerate(labels):
        new_label = np.zeros((label.shape[0], columns), dtype=np.int32)
        for speaker in range(label.shape[1]):
            if (chunk_index, speaker) in to_cluster:
                new_label[:, to_cluster[chunk_index, speaker]] |= label[:, speaker]
        relabeled.append(new_label)
    counts, _ = _overlap_add(relabeled, columns)
    if (audio.size - WINDOW_SIZE) % WINDOW_SHIFT:
        counts = counts[:min(audio.size // PYANNET_SHIFT, counts.shape[0] - 1) + 1]
    final_labels = np.zeros_like(counts, dtype=np.int32)
    for frame in range(counts.shape[0]):
        k = speakers_per_frame[frame]
        if k:
            final_labels[frame, np.argsort(-counts[frame], kind="stable")[:k]] = 1
    return _segments(final_labels, min_duration_on, min_duration_off)
