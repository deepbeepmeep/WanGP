# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Select bit-exact PTQ1 decode fusions before CUDA graph capture."""
import statistics

import torch
from torch._subclasses.fake_tensor import FakeTensor
from tqdm import tqdm

_choices = {}


def select_decode(input, raw, signs, bias, rows, grouped_shape, native):
    if isinstance(input, FakeTensor) or torch.compiler.is_compiling():
        return 0
    key = (input.device, input.dtype, input.shape[-1], rows, bias is not None, tuple(grouped_shape))
    if key in _choices:
        return _choices[key]
    if torch.cuda.is_current_stream_capturing():
        # Do not freeze an untuned choice when capture is the first caller.
        return 0
    with torch.inference_mode(), torch.cuda.device(input.device):
        def reference():
            rotated = native.prism_hadamard(input, signs, False, grouped_shape)
            return native.linear(raw, "PTQ1_0", (rows, input.shape[-1]), rotated, bias, input.dtype)

        expected = reference()
        candidates = {0: reference}
        for tile in (1, 2, 4):
            def candidate(tile=tile):
                return native.prism_decode(input, raw, signs, bias, rows, grouped_shape, tile)
            # Other architectures may choose a different baseline reduction.
            # Reject a differing candidate rather than aborting generation.
            try:
                torch.testing.assert_close(candidate(), expected, rtol=0, atol=0)
            except AssertionError:
                continue
            candidates[tile] = candidate
        if len(candidates) == 1:
            _choices[key] = 0
            tqdm.write(f"[GGUF][Prism] Decode {rows}x{input.shape[-1]}: Existing Kernel (Fused Numerical Check Failed).")
            return 0
        graphs = {}
        outputs = {}
        try:
            for tile, fn in list(candidates.items()):
                for _ in range(3):
                    fn()
                graph = torch.cuda.CUDAGraph()
                graphs[tile] = graph
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    for _ in range(16):
                        outputs[tile] = fn()
                graph.replay()
                try:
                    torch.testing.assert_close(outputs[tile], expected, rtol=0, atol=0)
                except AssertionError:
                    if tile == 0:
                        raise
                    graphs.pop(tile).reset()
                    outputs.pop(tile)
                    candidates.pop(tile)
            timings = {tile: [] for tile in candidates}
            for repeat in range(5):
                rotation = repeat % len(candidates)
                for tile in list(candidates)[rotation:] + list(candidates)[:rotation]:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(8):
                        graphs[tile].replay()
                    end.record()
                    end.synchronize()
                    timings[tile].append(start.elapsed_time(end) / 128)
            medians = {tile: statistics.median(values) for tile, values in timings.items()}
            best = min(medians, key=medians.get)
            # Retain the established kernel when differences are within noise.
            choice = best if medians[best] < medians[0] * .97 else 0
        finally:
            outputs.clear()
            for captured in graphs.values():
                captured.reset()
            graphs.clear()
            captured = None
            graph = None
        _choices[key] = choice
        from mmgp import offload
        if offload.default_verboseLevel >= 2:  # the choice of each shape, only in verbose mode
            tqdm.write(f"[GGUF][Prism] Decode {rows}x{input.shape[-1]}: "
                       f"{'Fused, Row Tile ' + str(choice) if choice else 'Existing Kernel'} "
                       f"({medians[0]*1000:.2f} -> {medians[choice]*1000:.2f} us).")
        return choice
