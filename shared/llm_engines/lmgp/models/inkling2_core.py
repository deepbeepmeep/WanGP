# Copyright WanGP. Licensed under the WanGP License 2.0.
# Inkling 2 (looped MoE) text model: plain PyTorch forward on the checkpoint's own tensors.
# Architecture after the Hugging Face Transformers implementation (Apache 2.0) and llama.cpp PR #32694 (MIT).
"""Eager Inkling 2 forward: the PyTorch path of the engine and the numerical reference of its kernels.

The model is a decoder of `layers` blocks whose blocks `loop[0]:loop[1]` run `passes` times with shared weights. A pass of a
block has its own keys, values and convolution states: there are `slots` = layers + (passes - 1) * loop blocks of them.
Weights keep the checkpoint's names and storage: routed experts stay packed (NVFP4 or MXFP8) and are decoded per expert.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

DRAFT_FEATURES = (5, 23, 41)  # block applications whose input the DSpark drafter reads, with the final hidden states
REFERENCE_SLOTS = 1000  # offset of the slots of the blocks after the loop when they run on the first pass's states (LoopCD-Logits)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@dataclass
class Inkling2Config:
    hidden: int = 2304
    layers: int = 24
    heads: int = 24
    kv_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    window: int = 1024
    local_layers: tuple = ()
    conv_taps: int = 4
    experts: int = 128
    top_k: int = 8
    intermediate: int = 1152
    gate_softcap: float = 4.0
    up_softcap: float = 25.0
    route_scale: float = 8.0
    qk_clip: float = 22.627416997969522
    rel_clip: float = 8.0
    log_floor: int = 8192
    log_alpha: float = 0.1
    eps: float = 1e-6
    block: int = 6
    loop: tuple = (6, 18)
    passes: int = 3
    gate_rank: int = 64
    vocab_rows: int = 201024
    vocab: int = 200058
    logit_softcap: float = 30.0

    @classmethod
    def from_json(cls, config):
        t = config["text_config"]; loop = t["loop_config"]
        return cls(hidden=t["hidden_size"], layers=t["num_hidden_layers"], heads=t["num_attention_heads"], kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"],
                   d_rel=t["d_rel"], rel_extent=t["rel_extent"], window=t["sliding_window_size"], local_layers=tuple(t["local_layer_ids"]), conv_taps=t["sconv_kernel_size"],
                   experts=t["n_routed_experts"], top_k=t["num_experts_per_tok"], intermediate=t["intermediate_size"], gate_softcap=t["mlp_gate_softcap"], up_softcap=t["mlp_up_softcap"],
                   route_scale=t["route_scale"], qk_clip=t["qk_l2_clip_threshold"], rel_clip=t["rel_l2_clip_threshold"], log_floor=t["log_scaling_n_floor"], log_alpha=t["log_scaling_alpha"],
                   eps=t["rms_norm_eps"], block=t["n_layers_per_attn_res_block"], loop=(loop["core_start"], loop["core_end"]), passes=loop["num_passes"], gate_rank=loop["gate_rank"],
                   vocab_rows=t["vocab_size"], vocab=t["unpadded_vocab_size"], logit_softcap=t["final_logit_softcapping"])

    @property
    def slots(self):
        return self.layers + (self.passes - 1) * (self.loop[1] - self.loop[0])

    def schedule(self):
        """(layer, slot, loop pass or None, index in the loop) of every block application, in execution order."""
        start, end = self.loop; out = [(i, i, None, None) for i in range(start)]
        for p in range(self.passes):
            out += [(i, i if p == 0 else self.layers + (p - 1) * (end - start) + i - start, p, i - start) for i in range(start, end)]
        return out + [(i, i, None, None) for i in range(end, self.layers)]


@dataclass
class Inkling2State:
    """Eager per-sequence state: keys / values and the last inputs of the four convolutions of every slot."""
    position: int = 0
    keys: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    conv: dict = field(default_factory=dict)


def rms(x, eps, weight=None):
    y = x.float(); y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return y if weight is None else weight.float() * y


def l2_clip(x, threshold):
    return x * (threshold / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)


def softcap(x, cap):
    return torch.tanh(x / cap) * cap


def dequantize_experts(weight, scale, factor=None, dtype=torch.float32):
    """Packed experts (experts x rows x packed columns) as floats. NVFP4 (`factor` given, amax / denominator per expert): two E2M1 codes
    per byte, low nibble first, one FP8 scale per 16 values. MXFP8: FP8 values with one scale 2 ** (byte - 127) per 32 values."""
    if factor is None:
        values = _fp8(weight).unflatten(-1, (-1, 32)) * torch.exp2(scale.float() - 127)[..., None]
    else:
        table = torch.tensor(E2M1, device=weight.device); codes = weight.long()
        values = torch.stack((table[codes & 15], table[codes >> 4]), -1).unflatten(-2, (-1, 8)).flatten(-2) * _fp8(scale)[..., None]
        values = values * factor.float().reshape(-1, 1, 1, 1)
    return values.flatten(-2).to(dtype)


_FP8_TABLE = {}


def _fp8(x):
    """FP8 E4M3 bytes as FP32 (a table lookup: PyTorch has no float8 arithmetic on every device)."""
    table = _FP8_TABLE.get(x.device)
    if table is None:
        b = torch.arange(256, dtype=torch.int32); sign = 1.0 - 2.0 * (b >> 7); e = (b >> 3) & 15; m = (b & 7).float()
        table = torch.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * torch.exp2(e.float() - 7)) * sign
        _FP8_TABLE[x.device] = table = table.to(x.device)
    return table[x.view(torch.uint8).long()]


# ---- activation quantization of the routed experts' inputs (opt-in): the checkpoint's hf_quant_config.json rules, as SGLang and vLLM
# implement them on Blackwell (SGLang routed_quant.cuh / inkling2_experts_sm90.py, vLLM inkling2/nvidia/ops/nvfp4.py, mxfp8.py). FP32, dequantized.
FLT_MAX = 3.4028234663852886e38


def _e2m1(x):
    """Nearest E2M1 value, ties to the even code, saturating at 6 (cvt.rn.satfinite.e2m1x2.f32)."""
    a = x.abs(); i = (a > 0.25).long() + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0)
    return torch.tensor(E2M1[:8], device=x.device)[i].copysign(x)


def nvfp4_activation(x):
    """NVFP4 rows (float32, last dim a multiple of 16), "fouroversix_mae": a per-row (per-token) outer scale amax / 1536; per block of 16 an
    E4M3 scale mapping the block maximum to 6 or to 4 of the E2M1 grid, whichever gives the smaller sum of absolute errors (ties: 6)."""
    amax = x.abs().amax(-1, keepdim=True); encoder = (amax.new_full(amax.shape, 1536.0) / amax).clamp(max=FLT_MAX)
    decoder, outer = (1.0 / encoder)[..., None], (amax * (1.0 / 1536.0))[..., None]
    blocks = x.unflatten(-1, (-1, 16)); bmax = blocks.abs().amax(-1, keepdim=True); out = []
    for d in (1.0 / 6.0, 0.25):
        scale = (bmax * (encoder[..., None] * d)).clamp(max=448.0).to(torch.float8_e4m3fn).float()
        values = _e2m1(blocks * (1.0 / (scale * decoder)).clamp(max=FLT_MAX)) * scale
        err = ((values * outer) - blocks).abs()
        for _ in range(4): err = err[..., 0::2] + err[..., 1::2]  # the pairwise order of the reference kernels
        out.append((values, err))
    (v6, e6), (v4, e4) = out
    return (torch.where(e4 < e6, v4, v6) * outer).flatten(-2)


def mxfp8_activation(x):
    """MXFP8 rows (float32, last dim a multiple of 32): per block of 32 a power-of-two scale 2^ceil(log2(amax / 448)) (UE8M0, floor 2^-127),
    values rounded to E4M3 (nearest even) after clamping to +-448."""
    blocks = x.unflatten(-1, (-1, 32)); raw = blocks.abs().amax(-1, keepdim=True) * (1.0 / 448.0)
    e = torch.where(raw <= 2.0 ** -127, 0, (((raw.view(torch.int32) + 0x7FFFFF) & 0x7F800000) >> 23).clamp(max=254))
    scale = (e << 23).view(torch.float32).clamp(min=2.0 ** -127); inverse = torch.where(e == 0, 2.0 ** 127, 1.0 / scale)
    return ((blocks * inverse).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale).flatten(-2)


class Inkling2Core:
    """Eager forward over a dict of checkpoint tensors (`model.llm.` prefix removed), all on one device.
    act_quant: quantize the routed experts' inputs as hf_quant_config.json requires (NVFP4 / MXFP8 per layer; the inputs of w13 and w2,
    routing weight applied before w2's quantization, BF16 rounding points of the reference kernels when the tensors are BF16); off by
    default (BF16 / FP32 activations, as the Transformers reference). Its products are exact only with FP32 tensors (experts decoded to FP32)."""

    def __init__(self, config, tensors, expert_cache=64, act_quant=False):
        self.c, self.w = config, tensors; self.act_quant = act_quant
        self.schedule = config.schedule(); self.expert_cache = expert_cache; self._experts = {}
        self.picks = None  # set to a list to record (slot, expert indices) of every block application

    def new_state(self):
        return Inkling2State()

    # ---- pieces -------------------------------------------------------------------------------------------------
    def embed(self, ids):
        return rms(self.w["embed.weight"][ids], self.c.eps, self.w["embed_norm.weight"]).to(self.w["embed.weight"].dtype)

    def head(self, hidden):
        return softcap((hidden @ self.w["unembed.weight"][: self.c.vocab].T).float(), self.c.logit_softcap)

    def _mix(self, query, block_states, partial):
        """Attention of learned queries over the block states and the current state (values are the normalized states)."""
        states = rms(torch.cat([block_states, partial.unsqueeze(-2)], -2), 1e-6)
        weights = torch.softmax((query.float() @ states.transpose(-1, -2)) * self.c.hidden ** -0.5, -1)
        return weights @ states  # tokens x queries x hidden

    def _gate(self, p, hidden, message):
        hidden, message = hidden.float(), message.float(); down = self.w[p + "down_weight"]
        g = (torch.cat([rms(hidden, self.c.eps), rms(message, self.c.eps)], -1).to(down.dtype) @ down.T).float()
        return (1 - torch.tanh(g @ self.w[p + "forget_weight"].float().T)) * hidden + torch.tanh(g @ self.w[p + "gain_weight"].float().T) * message

    def _history(self, state, slot, name, x):
        """`x` (tokens x channels, FP32) preceded by the convolution's last taps - 1 inputs; keeps the new ones."""
        taps = self.c.conv_taps; old = state.conv.get((slot, name))
        if old is None: old = x.new_zeros(taps - 1, x.shape[-1])
        full = torch.cat([old, x]); state.conv[(slot, name)] = full[-(taps - 1):].clone()
        return full

    def _conv(self, state, slot, name, x, weight):
        """Static causal depthwise convolution with its residual. weight: channels x 1 x taps, the last tap on the current token."""
        x = x.float(); windows = self._history(state, slot, name, x).unfold(0, self.c.conv_taps, 1)  # tokens x channels x taps
        return ((windows * weight.float().squeeze(1)).sum(-1) + x) / math.sqrt(self.c.conv_taps + 1)

    def _dynamic_conv(self, state, slot, name, x, source, p):
        """Causal convolution whose taps are static per channel plus per token and head; tap index = backward distance."""
        c = self.c; x = x.float(); taps = c.conv_taps
        dynamic = (source @ self.w[p + "weight_proj"].T).float().view(-1, c.kv_heads, 1, taps)
        weights = self.w[p + "static_w"].float().T.reshape(c.kv_heads, -1, taps) + dynamic
        windows = self._history(state, slot, name, x).unfold(0, taps, 1).flip(-1).reshape(x.shape[0], c.kv_heads, -1, taps)
        return ((windows * weights).sum(-1).reshape(x.shape) + x) / math.sqrt(2 * taps + 1)

    def _attention(self, layer, slot, state, q_in, k_in, v_in, positions):
        c, w, p = self.c, self.w, f"layers.{layer}.attn."; full = layer not in c.local_layers; dtype = q_in.dtype; tokens = q_in.shape[0]
        tau = (1.0 + c.log_alpha * torch.log(((positions + 1).float() / c.log_floor).clamp(min=1.0)))[:, None, None] if full else 1.0
        q = (l2_clip((q_in @ w[p + "wq_du.weight"].T).view(tokens, c.heads, c.head_dim).float() * w[p + "q_l2_clip.scale"].float(), c.qk_clip).to(dtype).float() * tau).to(dtype)
        k = k_in @ w[p + "wk_dv.weight"].T; v = v_in @ w[p + "wv_dv.weight"].T
        if full:
            k = self._dynamic_conv(state, slot, "k", k, k_in, p + "k_sconv."); v = self._dynamic_conv(state, slot, "v", v, v_in, p + "v_sconv.")
        else:
            k = self._conv(state, slot, "k", k, w[p + "k_sconv.weight"]); v = self._conv(state, slot, "v", v, w[p + "v_sconv.weight"])
        k = l2_clip(k.to(dtype).view(tokens, c.kv_heads, c.head_dim).float(), c.qk_clip).to(dtype)
        v = rms(v.to(dtype).view(tokens, c.kv_heads, c.head_dim), c.eps).to(dtype)
        keys = k if slot not in state.keys else torch.cat([state.keys[slot], k]); values = v if slot not in state.values else torch.cat([state.values[slot], v])
        if not full: keys, values = keys[-(c.window - 1 + tokens):], values[-(c.window - 1 + tokens):]
        state.keys[slot], state.values[slot] = keys, values
        if not full: state.keys[slot], state.values[slot] = keys[-(c.window - 1):], values[-(c.window - 1):]
        key_positions = positions[-1] - torch.arange(keys.shape[0] - 1, -1, -1, device=keys.device)
        distance = positions[:, None] - key_positions[None, :]  # queries x keys
        rel = (q_in @ w[p + "wr_du.weight"].T).view(tokens, c.heads, c.d_rel).float() * w[p + "rel_l2_clip.scale"].float()
        rel = (l2_clip(rel, c.rel_clip).to(dtype).float() * tau).to(dtype)
        rel_logits = (rel @ w[p + "rel_logits_proj.proj"]).transpose(0, 1)  # heads x queries x extent
        inside = (distance >= 0) & (distance < c.rel_extent)
        bias = rel_logits.gather(-1, distance.clamp(0, c.rel_extent - 1).expand(c.heads, -1, -1)).masked_fill(~inside, 0.0)
        allowed = (distance >= 0) if full else (distance >= 0) & (distance < c.window)
        group = c.heads // c.kv_heads
        scores = torch.einsum("qhd,khd->hqk", q, keys.repeat_interleave(group, 1)) * c.head_dim ** -0.5 + bias
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        sinks = (q_in @ w[p + "attn_sink_proj.weight"].T).T.unsqueeze(-1)  # heads x queries x 1
        probs = torch.softmax(torch.cat([scores, sinks], -1).float(), -1)[..., :-1].to(dtype)
        out = torch.einsum("hqk,khd->qhd", probs, values.repeat_interleave(group, 1)).reshape(tokens, -1)
        gate = torch.sigmoid((q_in @ w[p + "gate_attn.weight"].T).float())
        return (out.float() * gate).to(dtype) @ w[p + "wo_ud.weight"].T

    def _expert(self, layer, index):
        key = (layer, index); hit = self._experts.pop(key, None)
        if hit is None:
            w, p = self.w, f"layers.{layer}.mlp.experts."; ref = w["embed.weight"]; hit = []
            for n in ("w13", "w2"):
                amax = w.get(p + n + "_weight_amax"); factor = None if amax is None else (amax.float() / w[p + n + "_weight_outer_scale_denominator"].float())[[index]].to(ref.device)
                hit.append(dequantize_experts(w[p + n + "_weight"][[index]].to(ref.device), w[p + n + "_weight_scale"][[index]].to(ref.device), factor, ref.dtype)[0])
            hit = tuple(hit)
            while len(self._experts) >= self.expert_cache: self._experts.pop(next(iter(self._experts)))
        self._experts[key] = hit
        return hit

    def _moe(self, layer, slot, x):
        """Routed experts. x: tokens x hidden in the model dtype. Rows of w13 alternate gate and up."""
        c, w, p = self.c, self.w, f"layers.{layer}.mlp."
        scores = torch.sigmoid(x.float() @ w[p + "gate.weight"].float().T)
        picks = torch.topk(scores + w[p + "gate.bias"].float(), c.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, picks); weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * c.route_scale
        if self.picks is not None: self.picks.append((slot, picks.cpu()))
        out = torch.zeros_like(x, dtype=torch.float32)
        quant = (nvfp4_activation if p + "experts.w13_weight_amax" in w else mxfp8_activation) if self.act_quant else None
        if quant is not None: xq = quant(x.float())  # per token, shared by its experts
        for index in picks.unique().tolist():
            token, rank = torch.where(picks == index); w13, w2 = self._expert(layer, index)
            if quant is not None:
                gate_up = (xq[token] @ w13.float().T).to(x.dtype).float(); gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
                act = (softcap(gate, c.gate_softcap) * torch.sigmoid(gate) * softcap(up, c.up_softcap) * weights[token, rank, None]).to(x.dtype).float()
                out.index_add_(0, token, (quant(act) @ w2.float().T).to(x.dtype).float())
                continue
            gate_up = (x[token] @ w13.T).float(); gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
            act = (softcap(gate, c.gate_softcap) * torch.sigmoid(gate) * softcap(up, c.up_softcap)).to(x.dtype)
            out.index_add_(0, token, ((act @ w2.T) * weights[token, rank, None]).float())
        return out.to(x.dtype)

    def _block(self, layer, slot, state, hidden, block_states, positions, messages=None):
        """One block application. Returns the new hidden states; block_states: tokens x blocks x hidden."""
        c, w, p = self.c, self.w, f"layers.{layer}."; dtype = hidden.dtype; full = layer not in c.local_layers; residual_in = hidden
        if layer > 0:
            mixed = self._mix(w[f"attn_res.attn_queries.{layer - 1}"].view(-1, c.hidden), block_states, hidden)
            q_h, k_h, v_h = mixed.unbind(-2) if full else (mixed[:, 0],) * 3
        else:
            q_h = k_h = v_h = hidden
        if messages is not None:
            q_h, k_h, v_h = (self._gate(f"{p}controllers.{n}.", h, m) for n, h, m in (("attn", q_h, messages[:, 0]), ("k", k_h, messages[:, 1]), ("v", v_h, messages[:, 2])))
        if full:
            q_in, k_in, v_in = (rms(h, c.eps, w[f"{p}{n}_input_norm.weight"]).to(dtype) for n, h in (("q", q_h), ("k", k_h), ("v", v_h)))
        elif messages is not None:
            q_in, k_in, v_in = (rms(h, c.eps, w[p + "attn_norm.weight"]).to(dtype) for h in (q_h, k_h, v_h))
        else:
            q_in = k_in = v_in = rms(q_h, c.eps, w[p + "attn_norm.weight"]).to(dtype)
        attn = self._attention(layer, slot, state, q_in, k_in, v_in, positions)
        attn = self._conv(state, slot, "attn", attn, w[p + "attn_sconv.weight"]).to(dtype).float()
        if layer % c.block == 0:
            block_states = torch.cat([block_states, residual_in.unsqueeze(-2)], -2); residual = attn
        else:
            residual = residual_in.float() + attn
        mlp_in = self._mix(w[f"attn_res.mlp_queries.{layer}"].view(-1, c.hidden), block_states, residual)[:, 0]
        if messages is not None: mlp_in = self._gate(p + "controllers.mlp.", mlp_in, messages[:, 3])
        mlp = self._moe(layer, slot, rms(mlp_in, c.eps, w[p + "mlp_norm.weight"]).to(dtype))
        mlp = self._conv(state, slot, "mlp", mlp, w[p + "mlp_sconv.weight"]).to(dtype)
        return (residual + mlp.float()).to(dtype), block_states

    def _messages(self, previous_blocks, previous_hidden, index):
        """Query, key, value and MLP messages of loop block `index` from the previous pass."""
        w = self.w; q = w["middle_cycle_cross_pass_queries"].unflatten(0, (-1, 2))[index]; kv = w["middle_cycle_cross_pass_kv_queries"].unflatten(0, (-1, 2))[index]
        return self._mix(torch.cat([q[:1], kv, q[1:]]), previous_blocks, previous_hidden)

    # ---- forward ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, embeds, state, hidden_states=None, features=None, contrast=0.0, reference=None):
        """Runs the new tokens (tokens x hidden) through every block application; returns the final normalized hidden states.
        features: a list that receives the drafter's target features, tokens x 4 x hidden (the states entering the applications
        DRAFT_FEATURES and the returned states).
        contrast: loop contrastive decoding of the hidden states (LoopCD-Hidden, arXiv 2610.02185): the blocks after the loop read its
        last pass's states extrapolated away from its first pass's, n + contrast * (n - n_first), on the normalized states they read.
        reference: a list that receives the final normalized hidden states of the blocks after the loop run on the first pass's states
        (LoopCD-Logits: their own keys, values and convolution inputs, slots offset by REFERENCE_SLOTS)."""
        c = self.c; tokens = embeds.shape[0]; start, end = c.loop
        positions = torch.arange(tokens, device=embeds.device) + state.position
        hidden = embeds; blocks = embeds.new_zeros(tokens, 0, c.hidden); pre = previous = first = None
        taps = []
        for application, (layer, slot, loop_pass, index) in enumerate(self.schedule):
            if layer == start and index == 0:
                if loop_pass == 0: pre = (hidden, blocks)
                else:
                    previous = (blocks[:, pre[1].shape[1] + 1:], hidden); hidden, blocks = pre
                    if loop_pass == 1: first = previous
            if layer == end:  # the loop's exit: the states of its last pass are blocks[:, own:] and hidden
                own = pre[1].shape[1] + 1
                if reference is not None: exit_first = (torch.cat([blocks[:, :own], first[0]], -2), first[1])
                if contrast:
                    last = rms(torch.cat([blocks[:, own:], hidden.unsqueeze(-2)], -2), 1e-6)
                    guided = last + contrast * (last - rms(torch.cat([first[0], first[1].unsqueeze(-2)], -2), 1e-6))
                    blocks, hidden = torch.cat([blocks[:, :own], guided[:, :-1].to(blocks.dtype)], -2), guided[:, -1].to(hidden.dtype)
            if features is not None and application in DRAFT_FEATURES: taps.append(hidden)
            messages = self._messages(previous[0], previous[1], index) if loop_pass else None
            new_hidden, _ = self._block(layer, slot, state, hidden, blocks, positions, messages)
            if layer % c.block == 0: blocks = torch.cat([blocks, hidden.unsqueeze(-2)], -2)
            hidden = new_hidden
            if hidden_states is not None: hidden_states.append(hidden)
        state.position += tokens
        final = self._final(blocks, hidden, embeds.dtype)
        if features is not None: features.append(torch.stack(taps + [final], 1))
        if reference is not None:
            blocks, hidden = exit_first
            for layer in range(end, c.layers):
                new_hidden, _ = self._block(layer, layer + REFERENCE_SLOTS, state, hidden, blocks, positions)
                if layer % c.block == 0: blocks = torch.cat([blocks, hidden.unsqueeze(-2)], -2)
                hidden = new_hidden
            reference.append(self._final(blocks, hidden, embeds.dtype))
        return final

    def _final(self, blocks, hidden, dtype):
        final = self._mix(self.w["attn_res.output_query"].view(-1, self.c.hidden), blocks, hidden)[:, 0]
        return rms(final, self.c.eps, self.w["norm.weight"]).to(dtype)
