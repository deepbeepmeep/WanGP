# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# Nemotron-H hybrid decoder (Mamba-2, attention without positional encoding, routed experts with squared ReLU), written for the LM
# engine after the Hugging Face Transformers implementation (Apache 2.0, models/nemotron_h/modeling_nemotron_h.py) and NVIDIA's
# NeMo checkpoints. Single sequence per recurrent state row; routed experts are MMGP expert banks (mmgp.expert_compute.RoutedExperts).

import torch
from torch import nn
from torch.nn import functional as F
from mmgp import cuda_graphs
from mmgp.expert_compute import RoutedExperts

from ..layers.attention import Attention
from ..layers.moe import SigmoidBiasRouter, routed_experts
from ..utils.context import get_context


class NemotronHRMSNorm(nn.Module):
    """float32 normalization and weight product, the result cast back to the input dtype (as the reference)."""
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


class GatedGroupRMSNorm(nn.Module):
    """RMSNorm of x * silu(gate) by groups of channels (Mamba-2's output norm), computed in float32."""
    def __init__(self, hidden_size, group_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_size, self.eps = group_size, eps

    def forward(self, x, gate):
        dtype = x.dtype
        x = x.float() * F.silu(gate.float())
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.group_size)
        x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).view(shape)
        return self.weight * x.to(dtype)


def _segment_sum(x):
    # [..., T] -> [..., T, T]: sum of x over (j, i] below the diagonal, -inf above it
    length = x.shape[-1]
    x = x[..., None].expand(*x.shape, length)
    mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    sums = torch.cumsum(x, dim=-2)
    return sums.masked_fill(~torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=0), -torch.inf)


def mamba2_chunk_scan(x, dt, A, B, C, D, chunk_size, initial_state):
    """Chunked SSD of one sequence (float32): x [T, H, P], dt [T, H] (after softplus), A [H], B/C [T, G, N], D [H], initial_state
    [H, P, N] or None. Returns y [T, H, P] and the final state [H, P, N]. The reference's chunked scan (Transformers torch_forward)."""
    length, heads, head_dim = x.shape
    groups = B.shape[1]
    B = B.float().repeat_interleave(heads // groups, dim=1)
    C = C.float().repeat_interleave(heads // groups, dim=1)
    pad = (chunk_size - length % chunk_size) % chunk_size
    D_residual = D[:, None] * x
    x = x * dt[..., None]
    A = A[None, :] * dt
    if pad:
        x, A, B, C = (torch.cat((t, t.new_zeros(pad, *t.shape[1:])), 0) for t in (x, A, B, C))
    chunks = x.shape[0] // chunk_size
    x, A, B, C = (t.view(chunks, chunk_size, *t.shape[1:]) for t in (x, A, B, C))   # c l h p / c l h / c l h n
    A = A.permute(2, 0, 1)                                                              # h c l
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(_segment_sum(A))                                                      # h c l s
    G = torch.einsum("clhn,cshn->clsh", C, B)
    M = G * L.permute(1, 2, 3, 0)
    G = L = None
    Y_diag = torch.einsum("clsh,cshp->clhp", M, x)
    M = None
    decay_states = torch.exp(A_cumsum[:, :, -1:] - A_cumsum)                           # h c l
    states = torch.einsum("clhn,clhp->chpn", B * decay_states.permute(1, 2, 0)[..., None], x)
    previous = initial_state[None].float() if initial_state is not None else torch.zeros_like(states[:1])
    states = torch.cat((previous, states), 0)                                          # (c+1) h p n
    decay_chunk = torch.exp(_segment_sum(F.pad(A_cumsum[:, :, -1], (1, 0))))           # h (c+1) (c+1)
    new_states = torch.einsum("hzc,zhpn->chpn", decay_chunk.transpose(1, 2), states)
    states, final_state = new_states[:-1], new_states[-1]
    Y_off = torch.einsum("clhn,chpn->clhp", C, states) * torch.exp(A_cumsum).permute(1, 2, 0)[..., None]
    y = (Y_diag + Y_off).reshape(-1, heads, head_dim)[:length] + D_residual
    return y, final_state


class NemotronHMamba2Mixer(nn.Module):
    layer_type = "linear_attention"

    def __init__(self, config):
        super().__init__()
        self.num_heads, self.head_dim = config.mamba_num_heads, config.mamba_head_dim
        self.hidden_size, self.state_size, self.n_groups = config.hidden_size, config.ssm_state_size, config.n_groups
        self.intermediate_size = self.num_heads * self.head_dim
        self.conv_kernel_size, self.chunk_size = config.conv_kernel, config.chunk_size
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size
        self.time_step_limit = tuple(float(v) for v in config.time_step_limit)
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size + self.conv_dim + self.num_heads, bias=config.use_bias)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1, bias=config.use_conv_bias)
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(self.num_heads, dtype=torch.float32))
        self._lock_dtype = torch.float32  # precision lock of its own float32 SSM parameters (A_log, D, dt_bias); its linears are not concerned
        self.norm = GatedGroupRMSNorm(self.intermediate_size, self.intermediate_size // self.n_groups, config.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.conv_state_buffer = torch.empty(0)
        self.recurrent_state_buffer = torch.empty(0)
        # speculative verification: the states after each of the first verify_tokens - 1 tokens (the commit copies one of them back)
        self.speculative_conv_state_buffer = torch.empty(0)
        self.speculative_recurrent_state_buffer = torch.empty(0)
        self.speculative_replay = None

    # Sequence state (one row per sequence of the batch): the last conv_kernel - 1 convolution inputs and the float32 SSM state.
    def prepare_sequence_state(self, max_batch_size, device, dtype):
        conv_shape = (int(max_batch_size), self.conv_dim, self.conv_kernel_size - 1)
        state_shape = (int(max_batch_size), self.num_heads, self.head_dim, self.state_size)
        with torch.inference_mode():
            for name, shape, kind in (("conv_state_buffer", conv_shape, dtype), ("recurrent_state_buffer", state_shape, torch.float32)):
                buffer = getattr(self, name)
                if tuple(buffer.shape) != shape or buffer.device != device or buffer.dtype != kind:
                    setattr(self, name, torch.zeros(shape, device=device, dtype=kind))
                else:
                    buffer.zero_()

    def reset_sequence_state(self):
        with torch.inference_mode():
            self.conv_state_buffer.zero_()
            self.recurrent_state_buffer.zero_()

    def snapshot_sequence_state(self):
        return {name: (getattr(self, name), getattr(self, name).to("cpu", copy=True)) for name in ("conv_state_buffer", "recurrent_state_buffer")}

    def restore_sequence_state(self, snapshot):
        for name, (tensor, copy) in (snapshot or {}).items():
            setattr(self, name, tensor)
            tensor.copy_(copy)

    def release_sequence_state(self):
        self.conv_state_buffer = torch.empty(0, device="cpu")
        self.recurrent_state_buffer = torch.empty(0, device="cpu")
        self.speculative_conv_state_buffer = torch.empty(0, device="cpu")
        self.speculative_recurrent_state_buffer = torch.empty(0, device="cpu")

    def prepare_speculative_state(self, max_verify_tokens):
        for name, live in (("speculative_conv_state_buffer", self.conv_state_buffer), ("speculative_recurrent_state_buffer", self.recurrent_state_buffer)):
            shape, buffer = (int(max_verify_tokens) - 1, *live.shape), getattr(self, name)
            if tuple(buffer.shape) != shape or buffer.device != live.device or buffer.dtype != live.dtype:
                setattr(self, name, torch.empty(shape, device=live.device, dtype=live.dtype))

    def commit_speculative_state(self, processed_tokens, verified_tokens):
        if processed_tokens != verified_tokens:
            self.conv_state_buffer.copy_(self.speculative_conv_state_buffer[int(processed_tokens) - 1])
            self.recurrent_state_buffer.copy_(self.speculative_recurrent_state_buffer[int(processed_tokens) - 1])

    def _recurrent_step(self, xBC, dt, A, weight, conv_state, state):
        """One token per row: xBC [rows, conv_dim], dt [rows, H] (raw); updates conv_state / state in place, returns y [rows, H * P] float32."""
        rows = xBC.shape[0]
        sequence = torch.cat((conv_state, xBC[:, :, None].to(conv_state.dtype)), dim=2)  # [rows, conv_dim, K]
        conv_state.copy_(sequence[:, :, 1:])
        conv = (sequence.float() * weight.float()[None]).sum(-1)
        if self.conv1d.bias is not None:
            conv += self.conv1d.bias.float()
        xBC = F.silu(conv.to(xBC.dtype))  # rounded where the reference's BF16 convolution rounds
        sequence = conv = None
        x, B, C = xBC.split([self.intermediate_size, self.n_groups * self.state_size, self.n_groups * self.state_size], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias.float())                                 # [rows, H]
        x = x.view(rows, self.num_heads, self.head_dim)
        repeat = self.num_heads // self.n_groups
        B = B.view(rows, self.n_groups, 1, self.state_size).expand(-1, -1, repeat, -1).reshape(rows, self.num_heads, self.state_size)
        C = C.view(rows, self.n_groups, 1, self.state_size).expand(-1, -1, repeat, -1).reshape(rows, self.num_heads, self.state_size)
        dA = torch.exp(dt * A[None])[..., None, None]                                       # [rows, H, 1, 1]
        dBx = (dt[..., None, None] * B[:, :, None, :].float()) * x[..., None].float()       # [rows, H, P, N]
        state.mul_(dA).add_(dBx)
        dBx = None
        y = torch.matmul(state, C.float()[..., None])[..., 0]                             # [rows, H, P], float32 as the SSM kernels
        return (y + x.float() * self.D.float()[:, None]).view(rows, -1)

    def forward(self, hidden_states):
        context = get_context()
        tokens = hidden_states.shape[0]
        projected = self.in_proj(hidden_states)
        hidden_states = None
        gate, xBC, dt = projected.split([self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
        projected = None
        A = -torch.exp(self.A_log.float())
        weight = self.conv1d.weight[:, 0]  # [conv_dim, kernel]
        if context.is_prefill and getattr(context, "speculative_verify", False):
            # verification of a draft block (one sequence): the decoding arithmetic token by token, the state after each kept for the commit
            conv_state, state = self.conv_state_buffer[:1], self.recurrent_state_buffer[:1]
            outputs = []
            for index in range(tokens):
                outputs.append(self._recurrent_step(xBC[index:index + 1], dt[index:index + 1], A, weight, conv_state, state))
                if index < tokens - 1:
                    self.speculative_conv_state_buffer[index, :1].copy_(conv_state)
                    self.speculative_recurrent_state_buffer[index, :1].copy_(state)
            y = torch.cat(outputs)
            outputs = None
        elif context.is_prefill:
            if context.cu_seqlens_q is not None and context.cu_seqlens_q.numel() > 2:
                raise RuntimeError("The Nemotron-H decoder prefills one sequence at a time")
            previous = bool(getattr(context, "has_previous_state", False))
            conv_state, state = self.conv_state_buffer[0], self.recurrent_state_buffer[0]
            history = conv_state if previous else conv_state.new_zeros(conv_state.shape)
            sequence = torch.cat((history.to(xBC.dtype), xBC.t()), dim=1)                 # [conv_dim, K - 1 + T]
            conv_state.copy_(sequence[:, -(self.conv_kernel_size - 1):])
            xBC = F.silu(F.conv1d(sequence[None], weight[:, None], self.conv1d.bias, groups=self.conv_dim)[0].t())
            sequence = None
            x, B, C = xBC.split([self.intermediate_size, self.n_groups * self.state_size, self.n_groups * self.state_size], dim=-1)
            # dt in float32 as the SSM kernels (the reference's PyTorch path adds and applies softplus in BF16)
            dt = F.softplus(dt.float() + self.dt_bias.float()).clamp(*self.time_step_limit)
            y, final_state = mamba2_chunk_scan(x.float().view(tokens, self.num_heads, self.head_dim), dt, A, B.view(tokens, self.n_groups, self.state_size),
                                               C.view(tokens, self.n_groups, self.state_size), self.D.float(), self.chunk_size, state if previous else None)
            state.copy_(final_state)
            y = y.reshape(tokens, -1)
        else:
            # decoding: one token per sequence (row) of the batch
            y = self._recurrent_step(xBC, dt, A, weight, self.conv_state_buffer[:tokens], self.recurrent_state_buffer[:tokens])
        xBC = None
        # float32 norm of the float32 scan output (the fused gated RMSNorm of the SSM kernels), rounded once before the projection
        y = self.norm(y, gate).to(gate.dtype)
        return self.out_proj(y)


class NemotronHMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(F.relu(self.up_proj(x)).square())


STREAMED_PAIRS = 512


class NemotronHMoE(nn.Module):
    layer_type = "moe"

    def __init__(self, config, use_triton=False):
        super().__init__()
        self.gate = SigmoidBiasRouter(config.hidden_size, config.n_routed_experts, config.num_experts_per_tok, config.norm_topk_prob, config.routed_scaling_factor, config.n_group)
        self.experts = RoutedExperts(config.n_routed_experts, config.hidden_size, config.moe_intermediate_size, activation="relu2", device="meta", use_triton=use_triton)
        # prompt processing of audio (thousands of rows): each expert crosses the bus once rather than computing every row on the CPU
        self.experts.streamed_pairs = STREAMED_PAIRS
        self.shared_experts = NemotronHMLP(config.hidden_size, config.moe_shared_expert_intermediate_size, config.mlp_bias)

    def forward(self, x):
        ids, weights = self.gate(x)
        return routed_experts(self.experts, x, ids, weights) + self.shared_experts(x)


class NemotronHAttention(nn.Module):
    layer_type = "full_attention"

    def __init__(self, config):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)

    def forward(self, x):
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        return self.o_proj(self.attn(q, k, v).flatten(1, -1))


_MIXER_STATES = ("conv_state_buffer", "recurrent_state_buffer", "speculative_conv_state_buffer", "speculative_recurrent_state_buffer", "speculative_replay")
MIXERS = {"mamba": NemotronHMamba2Mixer, "moe": NemotronHMoE, "attention": NemotronHAttention}


class NemotronHBlock(nn.Module):
    def __init__(self, config, block_type, use_triton=False):
        super().__init__()
        self.block_type = block_type
        self.norm = NemotronHRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mixer = NemotronHMoE(config, use_triton) if block_type == "moe" else MIXERS[block_type](config)

    @property
    def layer_type(self):
        return self.mixer.layer_type

    # the LM engine's speculative verification walks the blocks (model.blk): the Mamba mixers' states
    def prepare_speculative_state(self, max_verify_tokens):
        if self.block_type == "mamba":
            self.mixer.prepare_speculative_state(max_verify_tokens)

    def commit_speculative_state(self, processed_tokens, verified_tokens):
        if self.block_type == "mamba":
            self.mixer.commit_speculative_state(processed_tokens, verified_tokens)

    def __getattr__(self, name):
        if name in _MIXER_STATES:
            return getattr(self._modules["mixer"], name)
        return super().__getattr__(name)

    def forward(self, hidden_states):
        return hidden_states + self.mixer(self.norm(hidden_states))


def _run_block(index, block, carry):
    hidden_states = carry[0]
    carry.clear()
    return [block(hidden_states)]


class NemotronHModel(nn.Module):
    def __init__(self, config, use_triton=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NemotronHBlock(config, block_type, use_triton) for block_type in config.layers_block_type])
        self.norm = NemotronHRMSNorm(config.hidden_size, config.layer_norm_epsilon)


class NemotronHForCausalLM(nn.Module):
    """The LM engine's Nemotron-H: forward(input_ids, positions, inputs_embeds) over the flattened tokens of the pass (the engine's
    context holds the paged KV slots of the attention layers); compute_logits applies the head to the last token of each sequence."""
    _lm_engine_tower_prefix = "model.layers."

    def __init__(self, config, use_triton=False):
        super().__init__()
        self.config = config
        self.model = NemotronHModel(config, use_triton)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = None  # a block drafter (DFlash2) when installed: it reads the hidden states of its target layers
        self._block_draft = False
        self._draft_features = None

    @property
    def embed_tokens(self):
        return self.model.embed_tokens

    @property
    def layers(self):
        return self.model.layers

    @property
    def blk(self):
        return self.model.layers

    def mamba_layers(self):
        return [layer.mixer for layer in self.model.layers if layer.block_type == "mamba"]

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, **_kwargs):
        hidden_states = self.model.embed_tokens(input_ids.view(-1)) if inputs_embeds is None else inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        # MMGP: inside a graph scope, each run of blocks kept in VRAM is one graph, the streamed blocks go through their hooks
        if not self._block_draft:
            (hidden_states,) = cuda_graphs.run_blocks(self, self.model.layers, _run_block, [hidden_states])
            return self.model.norm(hidden_states)
        # the drafter's features: the hidden states after its target layers, the tower run in segments ending at them
        features, start = [], 0
        for target in self.mtp.target_layer_ids:
            (hidden_states,) = cuda_graphs.run_blocks(self, self.model.layers[start:target + 1], _run_block, [hidden_states])
            features.append(hidden_states)
            start = target + 1
        if start < len(self.model.layers):
            (hidden_states,) = cuda_graphs.run_blocks(self, self.model.layers[start:], _run_block, [hidden_states])
        self._draft_features = torch.cat(features, dim=-1)[None]
        return self.model.norm(hidden_states)

    def lm_engine_graph_forward(self, input_ids, positions, outputs, draft_features=None):
        """A decoding pass of the engine (cuda_graphs.run stage): the hidden states of input_ids written into outputs, the block
        drafter's target features into draft_features."""
        outputs.copy_(self(input_ids, positions).view(outputs.shape))
        if draft_features is not None:
            draft_features.copy_(self._draft_features.view(draft_features.shape))

    def compute_logits(self, hidden_states):
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            hidden_states = hidden_states[context.cu_seqlens_q[1:] - 1]
        return self.lm_head(hidden_states)
