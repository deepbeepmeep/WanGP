# Adapted from LakonLab (https://github.com/Lakonik/LakonLab), released under the Apache License, Version 2.0: pi-H3's schedule and
# its Decoding Without Decoder pixel head (lakonlab/models/architectures/dwd_head.py, sol_h3/sol_h3.py).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""pi-H3: few-step students of H3 FL2VA, and their Decoding Without Decoder (DWD) head.

A DWD model does not decode its last latent: the last step's forward also hands the video rows after the last block to a pixel head,
which predicts the frames from them. The head is an AdaLN (a weightless LayerNorm modulated from the step's time) followed by the
last 8 layers of Sol-H3-VAE's decoder, retrained: every transformer token (2 x 2 latents) is expanded into 2 x 2 Sol tokens, and
the first latent of each native 5-latent block has its own one-frame output projection."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmgp import transfer

from .components.sol_decoder import BLOCK, CHUNK_ROWS, EMIT_BYTES, EMIT_TOKEN_BYTES, MiniMaxH3SolDecoder3d, _State

PI_MAX_RAW_T = {2: 0.93, 4: 0.96}  # the first time of the trained schedules, before the flow shift
PI_FINAL_STEP_SCALE = 0.6  # the last step is this fraction of the others


def pi_h3_sigmas(steps, shift=12.0):
    """The video sigmas of a pi-H3 student trained for this number of steps: equal steps in unshifted time, a shorter last one."""
    first = PI_MAX_RAW_T[int(steps)]
    raw = torch.linspace(first, first * PI_FINAL_STEP_SCALE / (int(steps) - 1 + PI_FINAL_STEP_SCALE), int(steps), dtype=torch.float64)
    return (float(shift) * raw / (1.0 + (float(shift) - 1.0) * raw)).tolist()


class _AdaLN(nn.Module):
    def __init__(self, time_dim, hidden):
        super().__init__()
        self.linear = nn.Linear(time_dim, 2 * hidden, dtype=torch.float32)


class MiniMaxH3DWDRefiner(MiniMaxH3SolDecoder3d):
    """The last layers of Sol-H3-VAE's decoder on the transformer's features instead of latents."""

    def __init__(self, in_channels=5376, expansion=2, num_layers=8):
        super().__init__(num_layers=num_layers)
        del self.x_embedder
        self.expansion = expansion
        self.feature_expand = nn.Linear(in_channels, self.dim * expansion * expansion)
        self.first_frame_proj_out = nn.Linear(self.dim, self.out_channels * self.patch_size * self.patch_size)  # stored and run in FP32, as proj_out

    def reset_rope(self):
        """The rotary frequencies, computed (not stored): for a model built without weights."""
        self.rope = type(self.rope)(int(self.head_dim * 0.75), theta=100.0)

    def _emit(self, x, lo, hi, state):
        """As the Sol decoder's, the first latent of each block through its own projection of one frame."""
        p, H, W, HW = self.patch_size, state.H, state.W, state.HW
        band = max(1, EMIT_BYTES // (W * EMIT_TOKEN_BYTES))
        for i in range(hi - lo):
            head, frames = (self.first_frame_proj_out, 1) if (lo + i) % BLOCK == 0 else (self.proj_out, self.patch_size_t)
            for top in range(0, H, band):
                bottom = min(top + band, H)
                hidden = F.layer_norm(x[i * HW + top * W:i * HW + bottom * W].float(), (self.dim,), self.norm_out.weight.float(), self.norm_out.bias.float(), self.norm_out.eps)
                pixels = head(hidden.to(head.weight.dtype)).view(1, bottom - top, W, self.out_channels, frames, p, p)
                del hidden
                state.emit(pixels.permute(0, 3, 4, 1, 5, 2, 6).reshape(1, self.out_channels, frames, (bottom - top) * p, W * p), top * p, bottom == H)
                del pixels

    def forward(self, features_list, grid, emit):
        """features_list: [(t * h * w, in_channels) rows in (t, h, w) order], handed off. emit: as the Sol decoder's."""
        features = features_list.pop()
        (t, h, w), e = grid, self.expansion
        state = _State(self, features.new_empty(1).expand(1, 1, t, h * e, w * e), emit)
        tokens, dtype = h * w, self.feature_expand.weight.dtype
        for frame in range(t):  # a token's e x e Sol tokens, into the frame's rows of Sol tokens
            expanded = self.feature_expand(features[frame * tokens:(frame + 1) * tokens].to(dtype)).view(h, w, e, e, self.dim)
            state.x[frame * state.HW:(frame + 1) * state.HW].view(h, e, w, e, self.dim).copy_(expanded.permute(0, 2, 1, 3, 4))
            del expanded
        del features
        state.x[state.N:] = torch.cat([self.register_tokens[0].to(state.dtype), state.x.new_zeros(1, self.dim)]).repeat(state.nb, 1)
        for layer, block in enumerate(self.transformer_blocks):
            block(state, layer)


class MiniMaxH3DWDHead(nn.Module):
    def __init__(self, hidden=5376, time_dim=8, table_size=1025, eps=1e-5):
        super().__init__()
        self.register_buffer("time_table", torch.empty(table_size, time_dim, dtype=torch.float32))  # the students' own timestep curve
        self.adaln_proj = _AdaLN(time_dim, hidden)
        self.eps = eps
        self.refiner = MiniMaxH3DWDRefiner(hidden)

    def features(self, hidden, timestep, row, dtype):
        """The head's input: hidden, the video rows after the last block (borrowed), through a weightless LayerNorm modulated from the
        rows' times. timestep: the forward's distinct times; row: the video rows' index into them, or spans (start, stop, index)."""
        table = self.time_table
        position = timestep.float().flatten().clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        temb = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
        scale, shift = self.adaln_proj.linear(F.silu(temb)).chunk(2, dim=-1)
        output = torch.empty(hidden.shape, dtype=dtype, device=hidden.device)
        for first, last, index in row if isinstance(row, list) else [(0, hidden.shape[0], row)]:
            for start in range(first, last, CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, last)
                output[start:stop] = F.layer_norm(hidden[start:stop].float(), (hidden.shape[-1],), None, None, self.eps).mul_(1.0 + scale[index]).add_(shift[index])
        return output

    def decode(self, features_list, grid, convert, interrupt=False):
        """The frames of the features on their (latent frames, rows, columns) grid of tokens, in RAM as (1, channels, frames, height,
        width). convert: the decoder's pixels (ImageNet-normalized RGB) of a band of rows to their output form, on the GPU."""
        refiner = self.refiner
        refiner._interrupt = interrupt
        frames, height = refiner.output_frames(grid[0]), grid[1] * refiner.expansion * refiner.patch_size
        decoded, position = None, 0
        with transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer:
            def emit(pixels, row, last):
                nonlocal decoded, position
                part = convert(pixels)
                if decoded is None:
                    decoded = torch.empty(1, part.shape[1], frames, height, part.shape[-1], dtype=part.dtype, device="cpu")
                writer.write(decoded[:, :, position:position + part.shape[2], row:row + part.shape[-2]], part)
                if last:
                    position += part.shape[2]

            refiner(features_list, grid, emit)
        if position != frames:
            raise RuntimeError(f"MiniMax H3 DWD head decoded {position} frames, expected {frames}")
        return decoded
