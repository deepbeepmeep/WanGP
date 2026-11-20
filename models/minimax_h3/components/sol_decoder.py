# Adapted from Sol-H3-VAE (https://huggingface.co/Efficient-Large-Model/Sol-H3-VAE, code/sol_h3_vae), released under the Apache
# License, Version 2.0: the window, global-token and rotary geometry of its noncausal decoder.
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
"""Sol-H3-VAE's video decoder: H3's ViT decoder pruned to 24 layers and distilled to decode whole frames, each native 5-latent block
(17 frames) at once, instead of 256-pixel tiles of 7-latent clips.

- Windows of 16 x 16 tokens over the 5 latents of a block; odd layers shift them by 8 tokens and 2 latents, so that a window then
  covers the end of one block and the start of the next.
- Each block has 5 global tokens (H3's 4 registers and a zero token): they attend to their whole block, and every window also attends
  to the global tokens of the blocks it covers.
- Rotary ids are local to the window, so a clip of any length decodes the same way.

The residual stream holds the clip; queries, keys and values exist for two blocks at a time (the ring). A layer runs in its block's
forward (MMGP loads a streamed layer there): at the step of block b, the windows ending in block b are attended, and the blocks whose
rows are then final get their output projection and FFN. The rows of the last layer are projected to pixels as they become final, one
latent frame at a time."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.attention import indexed_window_plans, pay_attention
from shared.kernels import ops as kernels, vae_ops

from ..interrupt import GenerationInterrupted
from .video_autoencoder import _FUSED_DTYPES, MiniMaxH3VideoRotaryPosEmbed, MiniMaxH3VideoTransformerBlock

BLOCK, WINDOW, TIME_SHIFT, FRAMES_TO_TRIM = 5, 16, 2, 3  # latents per native block, window tokens, shifts of odd layers, frames the first latent of a block drops
CHUNK_ROWS = 8192  # rows of one projection or FFN call
WINDOW_BYTES = 48 << 20  # the gathered keys of one window attention call
EMIT_BYTES, EMIT_TOKEN_BYTES = 64 << 20, 40 << 10  # a band of the output head, and what a token holds there (FP32 norm, projection, pixels)


def _norm(norm, x):
    return F.rms_norm(x.float(), (x.shape[-1],), norm.weight.float(), norm.eps).to(x.dtype)


class _Parity:
    """The windows of the even (unshifted) or odd (shifted) layers over a (T, H, W) grid, by steps: step j attends the windows of the
    temporal window j, the windows of block j, or, shifted, the last two latents of block j - 1 with the first three of block j (plus a
    last step for the end of the last block when it is long enough). Their tokens are then final for this layer.

    Ring layout of q / k / v: [last two latents of the previous block | its global tokens | the current block | its global tokens].
    steps[j]: (block projected at this step or None, attended latents lo..hi, groups of equal windows as (query ring positions (n, L),
    key ring positions (n, L + global tokens), rows of the queries in the attended range (n, L))); rope(b): block b's rotary table."""

    def __init__(self, decoder, state, odd):
        T, H, W, HW, S, nb, starts = state.T, state.H, state.W, state.HW, state.S, state.nb, state.starts
        shift, time_shift = (WINDOW // 2, TIME_SHIFT) if odd else (0, 0)
        t, y, x = (axis.reshape(-1) for axis in torch.meshgrid(torch.arange(T), torch.arange(H), torch.arange(W), indexing="ij"))
        j = (t + time_shift) // BLOCK  # temporal window
        columns, rows = -(-(W + shift) // WINDOW), -(-(H + shift) // WINDOW)
        window = (j * rows + (y + shift) // WINDOW) * columns + (x + shift) // WINDOW
        current, base = j.clamp(max=nb - 1), TIME_SHIFT * HW + S  # the block projected last when window j is attended, where it starts
        ring = torch.where(t // BLOCK == current, base + (t - BLOCK * current) * HW, (t - (BLOCK * current - TIME_SHIFT)) * HW) + y * W + x
        order = torch.argsort(window, stable=True)  # tokens (ids t * HW + y * W + x) by window, in (t, y, x) order inside each
        _, lengths = torch.unique_consecutive(window[order], return_counts=True)
        firsts = torch.cumsum(lengths, 0) - lengths
        window_j, spans_two = j[order[firsts]], t[order[firsts]] // BLOCK != t[order[firsts + lengths - 1]] // BLOCK
        steps = int(j.max()) + 1
        lows = torch.full((steps,), T).scatter_reduce(0, j, t, "amin")
        highs = torch.zeros(steps, dtype=torch.long).scatter_reduce(0, j, t + 1, "amax")
        # Indices are relative to the ring and to the first output row. Interior steps with the same
        # temporal bounds can reuse the entire window plan, including its device index buffers.
        self.steps, layouts, first_steps = [], {}, []
        for step in range(steps):
            block = min(step, nb - 1)
            lo, hi = int(lows[step]), int(highs[step])
            bounds = (lo - block * BLOCK, hi - block * BLOCK, starts[block + 1] - starts[block])
            first = layouts.setdefault(bounds, step)
            first_steps.append(first)
            groups = [] if first == step else self.steps[first][3]
            self.steps.append((block if step < nb else None, lo, hi, groups))
        for step, length, two in sorted(set(zip(window_j.tolist(), lengths.tolist(), spans_two.tolist()))):
            if step != first_steps[step]:
                continue
            chosen = ((window_j == step) & (lengths == length) & (spans_two == two)).nonzero().flatten()
            tokens = order[firsts[chosen][:, None] + torch.arange(length)]
            block = min(step, nb - 1)
            extra = base + starts[block + 1] - starts[block] + torch.arange(S)  # the current block's global tokens
            if two:
                extra = torch.cat([TIME_SHIFT * HW + torch.arange(S), extra])  # and the previous block's
            keys = torch.cat([ring[tokens], extra.expand(len(chosen), -1)], 1)
            per_call = max(1, WINDOW_BYTES // (keys.shape[1] * decoder.dim * 2))
            for first in range(0, len(chosen), per_call):
                part = slice(first, first + per_call)
                self.steps[step][3].append((ring[tokens[part]], keys[part], tokens[part] - int(lows[step]) * HW))
        # no reference to the state: the state holds the parities, and a cycle would keep the clip's buffers until the next GC
        self.rope_module, self.H, self.W, self.starts, self.device = decoder.rope, H, W, starts, state.device
        self.shifted, self.shift, self.time_shift, self.tables = odd, shift, time_shift, {}

    def rope(self, b):
        """The rotary table of block b's tokens: window-local ids, so that every full block has the same one."""
        local = tuple((b * BLOCK + i + self.time_shift) % BLOCK for i in range((self.starts[b + 1] - self.starts[b]) // (self.H * self.W)))
        if local not in self.tables:
            axes = [(torch.tensor(local, dtype=torch.float32) + 0.5) / BLOCK * 2 - 1,
                    *(((torch.arange(n) + self.shift) % WINDOW + 0.5) / WINDOW * 2 - 1 for n in (self.H, self.W))]
            ids = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(1, -1, 3).to(self.device)
            self.tables[local] = self.rope_module(ids)
        return self.tables[local]


class _State:
    """What the layers of one decode share: the residual rows (the tokens, then the global tokens of each block), the ring of q / k / v
    (a block and the two latents of the previous one that shifted windows reach), the attention outputs of one step, the windows."""

    def __init__(self, decoder, z, emit):
        _, _, self.T, self.H, self.W = z.shape
        self.decoder, self.device, self.dtype, self.emit = decoder, z.device, z.dtype, emit
        self.S, self.nb, self.HW = decoder.num_register_tokens + 1, -(-self.T // BLOCK), self.H * self.W
        self.starts = [min(b * BLOCK, self.T) * self.HW for b in range(self.nb + 1)]
        self.N = self.starts[-1]
        self.parities = [_Parity(decoder, self, odd) for odd in (False, True)]
        self.qkv = torch.empty((TIME_SHIFT + BLOCK) * self.HW + 2 * self.S, decoder.heads, 3, decoder.head_dim, dtype=self.dtype, device=self.device)
        layouts = {id(groups): groups for parity in self.parities for _, _, _, groups in parity.steps}
        plans = indexed_window_plans(self.qkv, list(layouts.values()), self.S, min(self.T, BLOCK) * self.HW + self.S, WINDOW_BYTES)
        if plans is None:
            prepared = {key: [tuple(index.to(self.device) for index in group) for group in groups] for key, groups in layouts.items()}
        else:
            prepared = dict(zip(layouts, plans))
        for parity in self.parities:
            parity.indexed = plans is not None
            parity.steps = [(block, lo, hi, prepared[id(groups)]) for block, lo, hi, groups in parity.steps]
        self.attention = torch.empty(BLOCK * self.HW, decoder.dim, dtype=self.dtype, device=self.device)
        self.global_attention = torch.empty(self.nb * self.S, decoder.dim, dtype=self.dtype, device=self.device)
        self.x = torch.empty(self.N + self.nb * self.S, decoder.dim, dtype=self.dtype, device=self.device)


class MiniMaxH3SolBlock(MiniMaxH3VideoTransformerBlock):
    """An H3 decoder block (same weights, same fused projections and FFN kernels) whose forward runs its layer over the whole clip."""

    def forward(self, state, layer):
        decoder, x, S, HW, starts = state.decoder, state.x, state.S, state.HW, state.starts
        parity, last = state.parities[layer % 2], layer == len(decoder.transformer_blocks) - 1
        fused = self._fused(x[:1])
        query, key, value = state.qkv.unbind(2)
        heads, head_dim, dim, base = decoder.heads, decoder.head_dim, decoder.dim, TIME_SHIFT * HW + S
        for block, lo, hi, windows in parity.steps:
            if getattr(decoder, "_interrupt", False):
                raise GenerationInterrupted
            if block is not None:
                count = starts[block + 1] - starts[block]
                if parity.shifted and block > 0:  # the windows of this step reach into the previous block's last two latents and its global tokens
                    state.qkv[:base].copy_(state.qkv[base + (BLOCK - TIME_SHIFT) * HW:base + BLOCK * HW + S])
                self._project_qkv(x[starts[block]:starts[block + 1]], state.qkv[base:base + count], parity.rope(block), fused)
                self._project_qkv(x[state.N + block * S:state.N + (block + 1) * S], state.qkv[base + count:base + count + S], None, fused)
            global_out = state.global_attention[block * S:(block + 1) * S] if block is not None and not last else None
            if parity.indexed:
                windows(query, key, value, state.attention, base, count + S if global_out is not None else 0, global_out)
            else:
                for queries, keys, outputs in windows:
                    n, length = queries.shape
                    qkv_list = [query.index_select(0, queries.view(-1)).view(n, length, heads, head_dim),
                                *(tensor.index_select(0, keys.view(-1)).view(n, -1, heads, head_dim) for tensor in (key, value))]
                    state.attention.index_copy_(0, outputs.view(-1), pay_attention(qkv_list, recycle_q=True).reshape(n * length, dim))
                if global_out is not None:
                    rows = slice(base, base + count + S)
                    qkv_list = [query[base + count:base + count + S][None], key[rows][None], value[rows][None]]
                    global_out.copy_(pay_attention(qkv_list, force_attention="sdpa").reshape(S, dim))
            rows = slice(lo * HW, hi * HW)  # final for this layer: their windows are attended
            self._post_attention(state.attention[:(hi - lo) * HW], x[rows], fused)
            if last:
                decoder._emit(x[rows], lo, hi, state)
            if block is not None and decoder.progress is not None:
                decoder.progress.advance()
        if not last:
            self._post_attention(state.global_attention, x[state.N:], fused)

    def _fused(self, x):
        """The INT8 kernels with their input norm and residual epilogue, as H3's decoder blocks use them."""
        return (all(kernels.can_linear(layer, x, dtypes=_FUSED_DTYPES) for layer in (self.attn.to_qkv, self.attn.to_out, self.ff.w1, self.ff.w2))
                and all(p.dtype == x.dtype for p in (self.norm1.weight, self.norm2.weight, self.scale1, self.scale2)))

    def _project_qkv(self, x, qkv, rope, fused):
        """qkv (rows, heads, 3, head_dim), a ring slot, = to_qkv(norm1(x)); then q and k get their per-head norm, and the rotary table
        when there is one (tokens; not the global tokens)."""
        attn, flat = self.attn, qkv.view(qkv.shape[0], -1)
        for start in range(0, x.shape[0], CHUNK_ROWS):
            rows = slice(start, start + CHUNK_ROWS)
            if fused:
                kernels.linear(attn.to_qkv, [x[rows]], input_act="rms_norm", act_weight=self.norm1.weight, act_eps=self.norm1.eps, out=flat[rows])
            else:
                flat[rows] = attn.to_qkv(_norm(self.norm1, x[rows]))
        query, key = qkv[None, :, :, 0], qkv[None, :, :, 1]
        if rope is None:  # the global tokens: the norm alone
            for view, norm in ((query, attn.norm_q), (key, attn.norm_k)):
                view.copy_(norm(view.float()).to(view.dtype))
            return
        plan = vae_ops.rope_plan(query, attn.dim_head, rope)
        normalized = [vae_ops.normalize_projection([view], norm, rope, plan) for view, norm in ((query, attn.norm_q), (key, attn.norm_k))]
        for view, done in zip((query, key), vae_ops.finish_rope_(*normalized, rope, attn.norm_q.eps, plan)):
            if done.data_ptr() != view.data_ptr():
                view.copy_(done)

    def _post_attention(self, attention, x, fused):
        """x, residual rows, += to_out(attention) * scale1, then += FFN(norm2(x)) * scale2: in place, by chunks of rows."""
        for start in range(0, x.shape[0], CHUNK_ROWS):
            rows, out = slice(start, start + CHUNK_ROWS), x[start:start + CHUNK_ROWS]
            if fused:
                kernels.linear(self.attn.to_out, [attention[rows]], residual=out, residual_scale=self.scale1, out=out)
                self.ff([out], pre_norm=self.norm2, residual_scale=self.scale2)
                continue
            out.addcmul_(self.attn.to_out(attention[rows]), self.scale1)
            kernels.mlp([_norm(self.norm2, out)], self.ff.w1, self.ff.w2, "swiglu", residual=out, gate=self.scale2)


class MiniMaxH3SolDecoder3d(nn.Module):
    """Sol-H3-VAE's 24-layer decoder (see the module docstring).

    No tiles and no VAE Tiling choice: the residual of the whole clip is an architecture fact (shifted windows chain the blocks, 24
    layers deep), and the ring holds a block and two latents of whole frames. Measured on a RTX 5090 (PyTorch allocator, INT8
    weights 1.87 GiB), VRAM above the weights: 1.18 GiB at 1280x720 x 124 frames, 2.45 GiB at 1920x1088 x 124 (full H3 INT8 VAE: 0.47 /
    0.64 above its 2.62 GiB): about 4 KiB per token for the residual (1.18 GiB at 1080p), 12 KiB per token of 7 latents for the ring
    (0.65 GiB) and 4 KiB per token of 5 latents for a step's attention outputs (0.16 GiB), then the window gathers and the output head
    by bands (0.2 to 0.3 GiB)."""

    def __init__(self, in_channels=24, out_channels=3, patch_size=16, patch_size_t=4, num_layers=24, num_attention_heads=32,
                 attention_head_dim=64, num_register_tokens=4, ffn_mult=4, rope_theta=100.0, rope_dim_ratio=0.75, norm_eps=1e-5):
        super().__init__()
        self.dim = num_attention_heads * attention_head_dim
        self.patch_size, self.patch_size_t, self.out_channels, self.num_register_tokens = patch_size, patch_size_t, out_channels, num_register_tokens
        self.heads, self.head_dim = num_attention_heads, attention_head_dim
        self.rope = MiniMaxH3VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.x_embedder = nn.Linear(in_channels, self.dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, self.dim))
        self.transformer_blocks = nn.ModuleList([MiniMaxH3SolBlock(self.dim, num_attention_heads, attention_head_dim, ffn_mult, norm_eps)
                                                 for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(self.dim, eps=norm_eps)
        self.proj_out = nn.Linear(self.dim, out_channels * patch_size_t * patch_size * patch_size)  # stored in FP32, run in FP32
        self.progress = None

    def output_frames(self, latent_frames):
        return latent_frames * self.patch_size_t - FRAMES_TO_TRIM * -(-latent_frames // BLOCK)

    def steps(self, latent_frames):
        """The progress units of a decode: one per layer and native block."""
        return len(self.transformer_blocks) * -(-latent_frames // BLOCK)

    def _emit(self, x, lo, hi, state):
        """The final rows x of latent frames lo..hi, frame after frame and by bands of latent rows: norm_out and proj_out in FP32,
        unpatchified, the first latent of each block keeping its last frame only."""
        p, pt, H, W, HW = self.patch_size, self.patch_size_t, state.H, state.W, state.HW
        band = max(1, EMIT_BYTES // (W * EMIT_TOKEN_BYTES))
        for i in range(hi - lo):
            for top in range(0, H, band):
                bottom = min(top + band, H)
                hidden = F.layer_norm(x[i * HW + top * W:i * HW + bottom * W].float(), (self.dim,), self.norm_out.weight.float(), self.norm_out.bias.float(), self.norm_out.eps)
                pixels = self.proj_out(hidden.to(self.proj_out.weight.dtype)).view(1, bottom - top, W, self.out_channels, pt, p, p)
                del hidden
                pixels = pixels.permute(0, 3, 4, 1, 5, 2, 6).reshape(1, self.out_channels, pt, (bottom - top) * p, W * p)
                state.emit(pixels[:, :, FRAMES_TO_TRIM:] if (lo + i) % BLOCK == 0 else pixels, top * p, bottom == H)
                del pixels

    def forward(self, z, emit):
        """z: (1, channels, T, H, W) after post_quant_conv. emit(pixels (1, out_channels, frames, rows, 16W), first pixel row, last band)
        receives the decoded frames in order, latent frame after latent frame, each by bands of rows."""
        state = _State(self, z, emit)
        tokens = z[0].permute(1, 2, 3, 0).reshape(state.N, -1)
        for start in range(0, state.N, CHUNK_ROWS):
            state.x[start:min(start + CHUNK_ROWS, state.N)] = self.x_embedder(tokens[start:start + CHUNK_ROWS].to(self.x_embedder.weight.dtype))
        del tokens
        state.x[state.N:] = torch.cat([self.register_tokens[0].to(state.dtype), state.x.new_zeros(1, self.dim)]).repeat(state.nb, 1)
        for layer, block in enumerate(self.transformer_blocks):
            block(state, layer)
