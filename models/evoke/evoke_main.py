# Adapted from Alaya-EVOKE (https://github.com/AlayaLab/Evoke, evoke/pipelines/pipeline_evoke.py), Copyright 2025 The Helios Team and The
# HuggingFace Team, Copyright 2026 The Evoke Team, Apache License 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""EVOKE-Turbo world generation: from a start image, 1.5 s chunks one after the other along a camera path. Each chunk: the memory view
(the stored frames rendered into the chunk's poses; for the first chunk of a new world, the start image's own depth) is VAE-encoded and
lightly noised, its covered tokens join the history (the start image latent, the last 19 generated latents in three tiers), three pyramid
stages denoise the chunk, the persistent causal decoder turns its 9 latents into 36 frames, and 12 of them are written to the memory.

A world is returned with its .wgpworld file (world_file.py) and stays open as a session; given a world file, generation continues it,
or serves its interactive exploration (shared.utils.world_exploration) one move at a time until the viewer leaves."""
import io
import itertools
import queue
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import torch
from accelerate import init_empty_weights
from mmgp import offload, runtime
from PIL import Image

from models.wan.configs.shared_config import wan_shared_cfg
from models.wan.modules.t5 import T5EncoderModel
from models.wan.modules.vae import WanVAE, count_conv3d
from shared.utils import files_locator as fl
from shared.utils.phase_progress import check_abort, generation_progress, set_phase_status
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.world_exploration import EXPLORE_PROMPT, exploration_for
from shared.utils.world_instructions import split_instructions
from . import sampler, world_file
from .camera import CHUNK_FRAMES, CHUNK_LATENTS, FPS, TARGET_FRAMES, Camera, intrinsics, plan, source_keys
from .exploration import DEFAULT_MEMORY_SECONDS, LATENTS_PER_SECOND
from .transformer import HISTORY_SIZES, LATENT_CHANNELS, EvokeTransformer, token_plan
from .vigeo import DepthStream, ViGeo
from .world_bank import WorldBank, single_image_view, visibility_to_latent

VIGEO_FILE = "evoke/vigeo1_1.safetensors"
VAE_FILE = "Wan2.1_VAE.safetensors"
SPATIAL_RATIO, TEMPORAL_RATIO = 8, 4
WARP_SIGMA_MAX = 0.135  # noise level of the covered memory view latents, drawn per latent frame in [0, WARP_SIGMA_MAX]
MOVE_SECONDS = 1.5  # a move: one chunk unless longer
EVENT_SECONDS = 3.0  # least length of a move with an event: one chunk leaves it too little time to happen


def chunk_count(seconds, event=False):
    """Chunks (1.5 s) of a move of `seconds`, EVENT_SECONDS at least with an event."""
    return max(1, round(max(seconds, EVENT_SECONDS if event else 0.0) / (CHUNK_FRAMES / FPS)))


@dataclass
class WorldSession:
    """An open world: its manifest, the generated latents (CPU), the conditioning latents of its start image, the pose of every frame, the
    camera after its last frame, the memory, the depth stream and the decoder's causal cache."""
    manifest: dict
    image_latent: torch.Tensor
    fake_latent: torch.Tensor
    latents: torch.Tensor
    poses: torch.Tensor
    camera: Camera
    bank: WorldBank
    stream: DepthStream
    decoder_cache: list
    prompts: list = field(default_factory=list)
    moves: int = 0
    last_frame: torch.Tensor | None = None
    pending: tuple | None = None  # (decoded frames, poses, first frame id) of the last chunk, written to the memory after its clip
    stats: dict = field(default_factory=dict)

    @property
    def chunks(self):
        return self.latents.shape[2] // CHUNK_LATENTS

    def history(self):
        """(long, mid, 1x) tiers: the last 19 latents of [19 zeros, the start image's fake latent, the generated latents]."""
        full = torch.cat([torch.zeros_like(self.fake_latent).expand(-1, -1, sum(HISTORY_SIZES), -1, -1), self.fake_latent, self.latents], dim=2)
        return full[:, :, -sum(HISTORY_SIZES):].split(list(HISTORY_SIZES), dim=2)


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, text_encoder_filename, save_quantized=False, VAE_dtype=torch.float16):
        self.device = torch.device("cuda")
        self.model_type = model_type
        model_filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        with init_empty_weights():
            transformer = EvokeTransformer()
        transformer.lock_layers_dtypes()
        offload.load_model_data(transformer, model_filename, writable_tensors=False, default_dtype=torch.bfloat16)  # the FP32 modules are locked
        self.transformer = transformer.eval().requires_grad_(False)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, model_filename, torch.bfloat16, None)
        with init_empty_weights():
            vigeo = ViGeo()
        offload.load_model_data(vigeo, fl.locate_file(VIGEO_FILE), writable_tensors=False, default_dtype=None)
        for module in vigeo.modules():  # FP16 matrices, FP32 norms / scales / tokens / head: kept as stored
            parameter = next(module.parameters(recurse=False), None)
            if parameter is not None:
                module._lock_dtype = parameter.dtype
        self.vigeo = vigeo.eval().requires_grad_(False)
        self.vae = WanVAE(vae_pth=fl.locate_file(VAE_FILE), dtype=VAE_dtype, device="cpu")
        self.vae.device = self.device
        tokenizer_path = fl.locate_folder("umt5-xxl")
        self.text_encoder = T5EncoderModel(text_len=wan_shared_cfg.text_len, dtype=wan_shared_cfg.t5_dtype, device=torch.device("cpu"),
                                           checkpoint_path=text_encoder_filename, tokenizer_path=tokenizer_path)
        self.text_cache = TextEncoderCache()
        self.session = None
        self._interrupt = False

    # ---- conditioning

    def _encode_text(self, prompts):
        contexts = self.text_encoder(prompts, self.device)
        return [torch.cat([context, context.new_zeros(wan_shared_cfg.text_len - context.shape[0], context.shape[1])]) for context in contexts]

    def _context(self, prompt):
        """The UMT5 states of a prompt (1, 512, 4096), zero-padded, cached on the CPU."""
        return self.text_cache.encode(self._encode_text, [prompt], device=self.device)[0].unsqueeze(0)

    def _vae_encode(self, video, generator):
        """Normalized latents (1, 16, T, h, w) of video (1, 3, F, H, W) in [-1, 1]: a posterior sample (mean + std * noise, as the source's
        latent_dist.sample)."""
        model = self.vae.model
        runtime.ensure_loaded(model)  # the encoder is driven here, not through the hooked encode()
        model.clear_cache()
        frames = video.shape[2]
        parts = []
        for index in range(1 + (frames - 1) // 4):
            model._enc_conv_idx = [0]
            chunk = video[:, :, :1] if index == 0 else video[:, :, 1 + 4 * (index - 1):1 + 4 * index]
            parts.append(model.encoder(chunk.to(self.vae.dtype), feat_cache=model._enc_feat_map, feat_idx=model._enc_conv_idx))
        model.clear_cache()
        mu, log_var = model.conv1(torch.cat(parts, 2)).chunk(2, dim=1)
        del parts
        latents = mu.float() + torch.exp(0.5 * log_var.float().clamp(-30.0, 20.0)) * torch.randn(mu.shape, generator=generator, device=mu.device)
        mean, scale = (value.to(latents.device).float().view(1, -1, 1, 1, 1) for value in self.vae.scale)
        return (latents - mean) * scale

    def _start_latents(self, image, generator):
        """(image latent, fake latent) of the start image (3, H, W) in [-1, 1]: the image as one frame, and the last latent of the image
        repeated over 33 frames (the source's history placeholder)."""
        image = image.to(self.device).unsqueeze(0).unsqueeze(2)
        return self._vae_encode(image, generator), self._vae_encode(image.expand(-1, -1, TARGET_FRAMES, -1, -1), generator)[:, :, -1:]

    def _decode(self, session, latents, first):
        """36 frames (3, 36, H, W) in [-1, 1] of a chunk's 9 latents, the decoder's causal cache kept across the world's chunks (the start
        image latent warms the first chunk, its frame dropped)."""
        model = self.vae.model
        runtime.ensure_loaded(model)  # the decoder is driven here (persistent cache), not through the hooked decode()
        if first:
            session.decoder_cache = [None] * count_conv3d(model.decoder)
            latents = torch.cat([session.image_latent.to(self.device), latents], dim=2)
        mean, scale = self.vae.scale
        z = latents.to(self.vae.dtype) / scale.to(self.device).view(1, -1, 1, 1, 1) + mean.to(self.device).view(1, -1, 1, 1, 1)
        x = model.conv2(z)
        frames = torch.cat([model.decoder(x[:, :, i:i + 1], feat_cache=session.decoder_cache, feat_idx=[0]) for i in range(x.shape[2])], dim=2)
        del x, z
        return (frames[:, :, 1:] if first else frames)[0].clamp_(-1, 1)

    # ---- one chunk

    def _memory_view(self, session, targets, K, image, generator, height, width):
        """(view latents (1, 16, 9, h, w), kept short-tier token indices) of a chunk: the stored frames rendered into the 33 target poses
        (the start image's own depth for a new world's first chunk), VAE-encoded, the covered latents noised by a per-frame sigma in
        [0, WARP_SIGMA_MAX] and the uncovered ones pure noise (they are dropped)."""
        if not len(session.bank) and image is not None:
            warp, visibility = single_image_view(session.stream, image.to(self.device), targets[0], targets, K, height, width)
        else:
            warp, visibility = session.bank.render(targets, K, height, width)
        latents = self._vae_encode(warp, generator)
        del warp
        mask = visibility_to_latent(visibility, CHUNK_LATENTS, latents.shape[-2:])
        sigmas = torch.rand(CHUNK_LATENTS, device=self.device, generator=generator) * WARP_SIGMA_MAX
        sigmas = mask * sigmas.view(1, 1, -1, 1, 1) + (1.0 - mask)
        latents = sigmas * torch.randn(latents.shape, generator=generator, device=self.device) + (1.0 - sigmas) * latents
        coverage = float(visibility.mean())
        # short tier tokens: the start latent, the 9 view latents (visible tokens only), the previous chunk's last latent
        tokens = (latents.shape[-2] // 2) * (latents.shape[-1] // 2)
        visible = mask[0, 0, :, ::2, ::2].reshape(-1) > 0.5
        keep = torch.cat([torch.ones(tokens, dtype=torch.bool, device=self.device), visible, torch.ones(tokens, dtype=torch.bool, device=self.device)])
        return latents, keep.nonzero().flatten(), coverage

    def _chunk(self, session, prompt, keys, generator, image=None, on_step=None):
        """Generates one chunk at the end of the world with keys held: returns its frames (3, 36, H, W) in [-1, 1]; the chunk is added to
        the session (latents, poses, camera) and written to the memory later (session.pending, _write_pending)."""
        self._write_pending(session)
        device, manifest = self.device, session.manifest
        height, width = manifest["height"], manifest["width"]
        latent_hw = (height // SPATIAL_RATIO, width // SPATIAL_RATIO)
        first = session.chunks == 0
        poses, camera = plan(session.camera, keys, first=first)
        K = intrinsics(height, width)
        start = session.chunks * CHUNK_FRAMES
        view, keep, coverage = self._memory_view(session, poses[:TARGET_FRAMES], K, image, generator, height, width)
        long, mid, previous = session.history()
        if first:
            previous = session.image_latent
        image_latent = session.image_latent.to(device)
        tiers = {"long": long.to(device), "mid": mid.to(device)}
        short_full = torch.cat([image_latent, view, previous.to(device)], dim=2)  # stage 0: with the memory view
        short_plain = torch.cat([image_latent, previous.to(device)], dim=2)  # stages 1-2
        base = 1 + sum(HISTORY_SIZES)
        indices = {"long": list(range(1, 1 + HISTORY_SIZES[0])), "mid": list(range(1 + HISTORY_SIZES[0], base - 1)),
                   "noise": list(range(base, base + CHUNK_LATENTS))}
        context = self._context(prompt)
        noise = torch.randn((1, LATENT_CHANNELS, CHUNK_LATENTS, *latent_hw), generator=generator, device=device)

        def predict(latents, stage, timestep):
            short = short_full if stage == 0 else short_plain
            short_indices = [0, *indices["noise"], base - 1] if stage == 0 else [0, base - 1]
            layout = token_plan({**indices, "short": short_indices}, short.shape[2], latent_hw, latents.shape[-2:], keep if stage == 0 else None, device)
            # int64, as the source passes it: MMGP would round a float32 timestep to BF16 at the root before the FP32 sinusoid
            return self.transformer(latents, torch.tensor(timestep, dtype=torch.int64, device=device), context, {**tiers, "short": short}, layout)

        latents = sampler.sample(predict, noise, generator, on_step)
        del short_full, short_plain, view, tiers, context
        session.latents = torch.cat([session.latents, latents.to("cpu", torch.bfloat16)], dim=2)
        frames = self._decode(session, latents, first)
        session.poses = torch.cat([session.poses, poses])
        session.camera = camera
        session.last_frame = frames[:, -1].clone()
        session.pending = (frames, poses, start)
        session.stats = {"coverage": round(coverage, 3)}
        return frames

    def _write_pending(self, session):
        """Writes the last chunk to the memory (12 of its frames through ViGeo), if not done yet."""
        if session.pending is not None:
            frames, poses, start = session.pending
            session.pending = None
            set_phase_status("Writing World Memory")
            session.bank.ingest(session.stream, frames.transpose(0, 1).to(self.device), poses, start, intrinsics(session.manifest["height"], session.manifest["width"]))

    def _generator(self, session, chunk):
        return torch.Generator(device=self.device).manual_seed(session.manifest["seed"] * 1_000_003 + chunk)

    # ---- worlds

    def _new_session(self, image, seed, height, width, description):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image_latent, fake_latent = self._start_latents(image, generator)
        manifest = {"model_type": self.model_type, "width": width, "height": height, "fps": FPS, "steps": 3, "seed": seed, "world_id": uuid.uuid4().hex,
                    "world": {"description": description, "segments": []}, "segments": [], "retention_seconds": DEFAULT_MEMORY_SECONDS}
        stream = DepthStream(self.vigeo)
        stream.reset()
        return WorldSession(manifest, image_latent.to("cpu", torch.bfloat16), fake_latent.to("cpu", torch.bfloat16),
                            torch.zeros((1, LATENT_CHANNELS, 0, height // SPATIAL_RATIO, width // SPATIAL_RATIO), dtype=torch.bfloat16),
                            torch.zeros((0, 4, 4)), Camera(), WorldBank(self.device, DEFAULT_MEMORY_SECONDS * FPS), stream, [])

    def close_session(self):
        self.session = None

    @staticmethod
    def _jpeg(frame, size=None):
        image = Image.fromarray(((frame.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy())
        if size is not None:
            image.thumbnail((size, size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85 if size else 92)
        return buffer.getvalue()

    def _world_file(self, session):
        session.manifest.update(camera=session.camera.as_dict(), depth_scale=session.stream.scale)
        return world_file.build(session.manifest, session, self._jpeg(session.last_frame, 320), self._jpeg(session.last_frame))

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, seed, frame_num, height, width, image_start=None, input_custom=None, callback=None, world_output=None, **kwargs):
        if input_custom is not None and input_prompt.strip() == EXPLORE_PROMPT:
            exploration = exploration_for(input_custom)
            return None if exploration is None else self._explore(exploration, callback)
        description, instructions = split_instructions(input_prompt)
        if input_custom is not None:
            session = self.open_world(input_custom)
            description = description or session.manifest["world"]["description"]
        else:  # a start image and a description (validate_generative_settings)
            session = self._new_session(image_start, seed, height, width, description)
        self.close_session()
        moves = [(source_keys(instruction["keys"]), instruction["event"], chunk_count(instruction["seconds"] or MOVE_SECONDS, bool(instruction["event"])), instruction["text"]) for instruction in instructions]
        if not moves:  # no instructions: the requested length with a still camera
            moves = [(set(), "", max(1, round(frame_num / CHUNK_FRAMES)), "")]
        total = sum(count for _, _, count, _ in moves)
        for prompt in dict.fromkeys(description if not event else f"{description} {event}" for _, event, _, _ in moves):
            self._context(prompt)  # every prompt encoded (and cached) before the denoising steps are announced
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=total * sampler.STAGES)
        frames_out, done, image = [], 0, image_start if input_custom is None else None
        for keys, event, count, text in moves:
            prompt = description if not event else f"{description} {event}"
            first_chunk = session.chunks
            for _ in range(count):
                set_phase_status(f"Generating Chunk {done + 1}/{total}")
                generator = self._generator(session, session.chunks)

                def on_step(stage, latents, done=done):  # the preview: each pyramid stage's clean latents, coarse to full size
                    if callback is not None:
                        callback(done * sampler.STAGES + stage, latents[0], False)

                frames_out.append(self._chunk(session, prompt, keys, generator, image, on_step).cpu())
                image, done = None, done + 1
                check_abort()
            session.manifest["segments"].append({"latent_start": first_chunk * CHUNK_LATENTS, "latent_frames": count * CHUNK_LATENTS, "video_frames": count * CHUNK_FRAMES,
                                                  "keys": "+".join(sorted(keys)), "event": event, **({"instruction": text} if text else {})})
            session.manifest["world"]["segments"].append({"visual_description_long": description if not session.manifest["world"]["segments"] else "",
                                                          "description_change": event or None, "instruction": text})
        self._write_pending(session)
        video = torch.cat(frames_out, dim=1)
        session.manifest["world_id"] = uuid.uuid4().hex
        self.session = session
        if (world_output or "world") == "video":
            return {"x": video}
        return {"x": video, "side_files": {world_file.EXTENSION: self._world_file(session)}}

    @torch.inference_mode()
    def open_world(self, path):
        """Makes the world of a .wgpworld file the open session: its latents, poses, camera and stored memory; the depth stream restarts
        with the world's depth scale, primed on its first and last stored windows, and the decoder's cache is warmed on its last latents."""
        manifest, tensors = world_file.read(path)
        session = self.session
        if session is not None and session.manifest["world_id"] == manifest["world_id"] and session.moves == 0:
            return session
        self.close_session()
        stream = DepthStream(self.vigeo)
        stream.reset()
        bank = WorldBank(self.device, manifest["retention_seconds"] * FPS)
        bank.load_state(tensors)
        session = WorldSession(manifest, tensors["image_latent"], tensors["fake_latent"], tensors["latents"], tensors["poses"], Camera(**manifest["camera"]),
                               bank, stream, [])
        set_phase_status("Preparing World")
        stream.scale = manifest["depth_scale"]
        windows = sorted({gid // CHUNK_FRAMES for gid in bank.frames})
        for window in windows[:1] + windows[-5:] if len(windows) > 6 else windows:
            ids = [gid for gid in sorted(bank.frames) if gid // CHUNK_FRAMES == window]
            stream.window(torch.stack([bank.frames[gid].rgb for gid in ids]).to(self.device).float() / 255)
            check_abort()
        stream.scale = manifest["depth_scale"]
        warm = min(session.latents.shape[2], 2 * CHUNK_LATENTS)  # the causal decoder's state from its last latents
        if warm:
            self._decode(session, session.latents[:, :, -warm:].to(self.device), True)
        session.last_frame = None
        self.session = session
        return session

    # ---- exploration

    @torch.inference_mode()
    def move(self, keys, text="", seconds=MOVE_SECONDS, on_step=None, memory=None):
        """Continues the open world by one move: whole chunks (1.5 s) with the explorer's held keys ({"keys": "w+left"}, None for none) and
        an optional text event; memory: the video the memory keeps, in latent frames (the explorer's Memory menu). Returns the frames
        (3, frames, H, W) in [-1, 1]; the last chunk's memory write waits for finish_move (or the next move)."""
        session, keys = self.session, source_keys((keys or {}).get("keys"))
        if memory:
            session.manifest["retention_seconds"] = round(memory / LATENTS_PER_SECOND)
            session.bank.retention_frames = session.manifest["retention_seconds"] * FPS
        description = session.manifest["world"]["description"]
        prompt = description if not text else f"{description} {text}"
        frames = []
        for _ in range(chunk_count(seconds, bool(text))):
            frames.append(self._chunk(session, prompt, keys, self._generator(session, session.chunks), on_step=on_step))
            check_abort()
        session.moves += 1
        return torch.cat(frames, dim=1)

    @torch.inference_mode()
    def finish_move(self):
        if self.session is not None:
            self._write_pending(self.session)

    @torch.inference_mode()
    def explore(self, world_path):
        """Serves the exploration of a world outside the queue (the Worlds gallery took the GPU from a running generation)."""
        exploration = exploration_for(world_path)
        if exploration is not None:
            self._explore(exploration, None)

    def _move_stats(self, seconds, frames, request):
        """Statistics the explorer shows (G), as Hy-Nomas gives them: the GPU and RAM, the move's timing, what was explored (the world's
        own video, then the moves) and what the memory keeps of it, in seconds, video frames and moves of the chosen length; with the
        share of the last memory view the memory covered."""
        from shared.utils.stats import live_stats
        session = self.session
        move = chunk_count(request["seconds"]) * CHUNK_FRAMES

        def amount(video_frames):
            return {"seconds": round(video_frames / FPS, 1), "frames": video_frames, "moves": round(video_frames / move)}

        world = sum(segment["video_frames"] for segment in session.manifest["segments"])
        explored, kept = session.chunks * CHUNK_FRAMES, session.manifest["retention_seconds"] * FPS
        memory = (f"Memory   the last {min(explored, kept) / FPS:.1f} s of the {explored / FPS:.1f} s explored" + (", older places forgotten" if explored > kept else "")
                  + f" · the last view was {round(100 * session.stats['coverage'])} % remembered")
        return {"move_seconds": round(seconds, 2), "video_seconds": round(frames / FPS, 2), "steps": sampler.STAGES, "moves": session.moves,
                "peak_gb": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 1), **live_stats(), "world": amount(world), "explored_moves": amount(explored - world),
                "explored": amount(explored), "memory": amount(min(explored, kept)), "memory_full": explored > kept, "room_moves": max(0, kept - explored) // move,
                "memory_line": memory}

    def _render(self, session):
        """The whole video explored so far, decoded again from the start (uint8, frames x H x W x 3)."""
        replay = WorldSession(session.manifest, session.image_latent, session.fake_latent, session.latents, session.poses, session.camera, session.bank, session.stream, [])
        parts = []
        for chunk in range(session.chunks):
            frames = self._decode(replay, session.latents[:, :, chunk * CHUNK_LATENTS:(chunk + 1) * CHUNK_LATENTS].to(self.device), chunk == 0)
            parts.append(((frames.float() + 1) * 127.5).round().clamp_(0, 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy())
            check_abort()
        return np.concatenate(parts)

    def _explore(self, exploration, callback):
        """Serves the moves of an exploration until the viewer leaves (or the task is aborted); nothing is saved."""
        try:
            set_phase_status("Preparing World")
            self.open_world(exploration.world_path)
            self._context(self.session.manifest["world"]["description"])
            exploration.publish("ready")
            while True:
                set_phase_status("Exploring World - Waiting for a Move")
                try:
                    request = exploration.next_request(0.25)
                except queue.Empty:
                    check_abort()
                    continue
                if request is None:
                    break
                if "render" in request:
                    set_phase_status("Exploring World - Rendering the Video")
                    exploration.publish("rendered", frames=self._render(self.session), fps=FPS, record=request["render"], moves=self.session.moves)
                    continue
                set_phase_status(f"Exploring World - Move {self.session.moves + 1}")
                chunks = chunk_count(request["seconds"], bool(request["text"]))
                description = self.session.manifest["world"]["description"]
                self._context(description if not request["text"] else f"{description} {request['text']}")  # encoded before the steps are announced
                if callback is not None:
                    callback(-1, None, True, override_num_inference_steps=chunks * sampler.STAGES)
                steps = itertools.count()  # no preview while exploring: nobody sees the progress panel
                on_step = None if callback is None else (lambda stage, latents: callback(next(steps), None, False))
                torch.cuda.reset_peak_memory_stats(self.device)
                started = time.perf_counter()
                video = self.move(request["keys"], request["text"], request["seconds"], on_step, request.get("memory"))
                frames = ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
                exploration.publish("clip", frames=frames, fps=FPS, stats=self._move_stats(time.perf_counter() - started, len(frames), request))
                del video
                self.finish_move()  # while the clip plays
        finally:
            exploration.publish("ended")
            self.close_session()
        return None
