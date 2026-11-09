# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# Portions adapted from flash-linear-attention (https://github.com/fla-org/flash-linear-attention),
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li, MIT License (see docs/third_party_licenses/MIT.txt).
"""Qwen GDN decode fusions, with numerical probes on other NVIDIA targets."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from fla.modules.convolution import ShortConvolution, causal_conv1d_update_kernel


class GDNShortConvolution(ShortConvolution):
    def step(self, x, residual, cache, output_final_state=False, cu_seqlens=None):
        # Installed only on the four-tap Qwen convolution. The
        # original FLA kernel's arithmetic and state update remain unchanged.
        batch, _, channels = x.shape
        if output_final_state and cache is None:
            cache = x.new_zeros((batch, channels, 4))
        y = torch.empty_like(x)
        causal_conv1d_update_kernel[(triton.cdiv(channels, 64), batch)](
            x=x, cache=cache, residual=residual, y=y, weight=self.weight,
            bias=self.bias, D=channels, W=4, BD=64, BW=4,
            ACTIVATION=self.activation, num_warps=4,
        )
        return y, cache


@triton.jit
def _prepare_kernel(Q, K, A, B, SSM_A, SSM_DT, Q_OUT, K_OUT, G, BETA,
                    Q_BATCH: tl.constexpr, Q_HEAD: tl.constexpr,
                    A_BATCH: tl.constexpr, B_BATCH: tl.constexpr,
                    HEADS: tl.constexpr, REPEAT: tl.constexpr, DIM: tl.constexpr,
                    BLOCK: tl.constexpr):
    head = tl.program_id(0)
    batch = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    source = batch * Q_BATCH + (head // REPEAT) * Q_HEAD + cols
    dest = (batch * HEADS + head) * DIM + cols
    q = tl.load(Q + source, cols < DIM, 0)
    k = tl.load(K + source, cols < DIM, 0)
    tl.store(Q_OUT + dest, q, cols < DIM)
    tl.store(K_OUT + dest, k, cols < DIM)
    a = tl.load(A + batch * A_BATCH + head).to(tl.float32)
    b = tl.load(B + batch * B_BATCH + head).to(tl.float32)
    dt = a + tl.load(SSM_DT + head).to(tl.float32)
    softplus = tl.where(dt > 20., dt, libdevice.log1p(tl.exp(dt)))
    g = tl.load(SSM_A + head).to(tl.float32) * softplus
    tl.store(G + batch * HEADS + head, g)
    tl.store(BETA + batch * HEADS + head, tl.sigmoid(b))


def prepare_decode(query, key, a, b, ssm_a, ssm_dt, num_v_heads):
    batch, _, num_k_heads, dim = query.shape
    q = query.new_empty((batch, 1, num_v_heads, dim))
    k = torch.empty_like(q)
    g = torch.empty((batch, 1, num_v_heads), device=query.device, dtype=torch.float32)
    beta = b.new_empty((batch, 1, num_v_heads))
    _prepare_kernel[(num_v_heads, batch)](
        query, key, a, b, ssm_a, ssm_dt, q, k, g, beta,
        query.stride(0), query.stride(2), a.stride(0), b.stride(0),
        num_v_heads, num_v_heads // num_k_heads, dim,
        triton.next_power_of_2(dim), num_warps=4,
    )
    return q, k, g, beta


def install_gdn_decode(model):
    """Install after checkpoint layout configuration; weights stay MMGP-owned."""
    if torch.version.hip is not None:
        return
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 8:
        return _install_turing_recurrence(model, capability)
    from shared.kernels import gdn_fused
    validated_launches = capability == (12, 0)
    count = 0
    for block in model.blk:
        # The 27B hidden-width reduction benefits from more threads for a
        # single decode row. Other widths/batches retain defaults.
        for norm in (block.attn_norm, block.post_attention_norm) if hasattr(block, "attn_norm") else ():  # models with hyper-connections have no layer norms
            if validated_launches and norm.weight.numel() == 5120:
                norm._small_batch_num_warps = 16
        if block.layer_type != "linear_attention":
            continue
        if (block._fast_recurrent_gated_delta_rule is not None
                and block._use_short_convolution
                and block.ssm_conv1d.backend == "triton"
                and block.conv_kernel_size == 4):
            if validated_launches:
                block.ssm_conv1d.__class__ = GDNShortConvolution
            block._gdn_prepare_decode = prepare_decode if validated_launches else prepare_decode_checked
            block._gdn_recurrent_raw = recurrent_raw_gates if validated_launches else recurrent_raw_gates_checked
            # Speculative verification keeps one state per layer and replays accepted prefixes with this kernel
            # instead of storing a snapshot per draft token; checked launches first verify that it is exact.
            block._gdn_replay_verification = validated_launches
            block._gdn_replay_probe = None if validated_launches else probe_replay_verification
            if validated_launches and gdn_fused.ENABLED and gdn_fused.eligible(block):
                block._gdn_fused = gdn_fused.mixer
                block._gdn_state_dtype = gdn_fused.STATE_DTYPE
            count += 1
    if count:
        mode = "Validated SM120 Launches" if validated_launches else "Numerical Checks Before Capture"
        print(f"[Qwen][GDN] Decode Fusions Available For {count} Layers ({mode}).")


# Adapted from the FLA-derived recurrent_verify_kernel in
# lmgp/layers/speculative_state.py; its MIT notice applies.
from fla.ops.utils.op import exp


@triton.jit(do_not_specialize=("T", "BATCH", "QS0", "QS1", "QS2", "KS0", "KS1", "KS2", "VS0", "VS1", "VS2", "AS0", "AS1", "BS0", "BS1", "SAVE_TOKENS"),
            do_not_specialize_on_alignment=("T", "BATCH", "QS0", "QS1", "QS2", "KS0", "KS1", "KS2", "VS0", "VS1", "VS2", "AS0", "AS1", "BS0", "BS1", "SAVE_TOKENS"))
def _recurrent_raw_kernel(Q, K, V, A, BETA_IN, SSM_A, DT, STATE, STATE_OUT, OUT, SNAPSHOTS,
                          PRE, K_SAVE, V_SAVE, A_SAVE, B_SAVE,
                          T, H: tl.constexpr, HV: tl.constexpr,
                          DK: tl.constexpr, DV: tl.constexpr,
                          QS0, QS1, QS2,
                          KS0, KS1, KS2,
                          VS0, VS1, VS2,
                          AS0, AS1,
                          BS0, BS1,
                          BATCH, BK: tl.constexpr, BV: tl.constexpr,
                          SAVE_PREFIX: tl.constexpr, SAVE_INPUTS: tl.constexpr, SAVE_TOKENS,
                          V_HEADS_TILED: tl.constexpr, SSM_PARAMS_TILED: tl.constexpr,
                          INTERLEAVE_AB: tl.constexpr, SHARED_STATE: tl.constexpr):
    # SHARED_STATE: every batch entry (a draft tree's path) starts from the one state of batch 0 and none writes a state back
    vb, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HV, bh % HV
    kh = head // (HV // H)
    tiled_head = (head % (HV // H)) * H + kh
    interleaved_head = head // 2 + (head % 2) * (HV // 2)
    value_head = tiled_head if V_HEADS_TILED else head
    gate_head = tiled_head if V_HEADS_TILED else (interleaved_head if INTERLEAVE_AB and HV % 2 == 0 else head)
    param_head = tiled_head if SSM_PARAMS_TILED else (interleaved_head if INTERLEAVE_AB and HV % 2 == 0 else head)
    keys = tl.arange(0, BK)
    values = vb * BV + tl.arange(0, BV)
    state_index = bh * DK * DV + keys[:, None] * DV + values[None, :]
    mask = (keys[:, None] < DK) & (values[None, :] < DV)
    raw_state = tl.load(STATE + (head * DK * DV + keys[:, None] * DV + values[None, :] if SHARED_STATE else state_index), mask, 0)
    state = raw_state.to(tl.float32)
    if SAVE_INPUTS:  # what a replay of any accepted prefix needs: the state before verification and every input as read
        tl.store(PRE + state_index, raw_state, mask)
    sa = tl.load(SSM_A + param_head).to(tl.float32)
    dt = tl.load(DT + param_head).to(tl.float32)
    for token in range(T):
        # Projection and replay strides contain whole heads. Keep their known
        # alignment with dynamic strides to preserve the norm reduction order.
        q_offset = tl.multiple_of(batch * QS0 + token * QS1 + kh * QS2, DK)
        k_offset = tl.multiple_of(batch * KS0 + token * KS1 + kh * KS2, DK)
        v_offset = tl.multiple_of(batch * VS0 + token * VS1 + value_head * VS2, DV)
        q = tl.load(Q + q_offset + keys, keys < DK, 0).to(tl.float32)
        raw_k = tl.load(K + k_offset + keys, keys < DK, 0)
        raw_v = tl.load(V + v_offset + values, values < DV, 0)
        raw_a = tl.load(A + batch * AS0 + token * AS1 + gate_head)
        raw_beta = tl.load(BETA_IN + batch * BS0 + token * BS1 + gate_head)
        if SAVE_INPUTS:  # stored at the indices read, so a replay reads them back through the same head mapping
            saved = batch * SAVE_TOKENS + token
            tl.store(V_SAVE + (saved * HV + value_head) * DV + values, raw_v, values < DV)
            if vb == 0:
                tl.store(A_SAVE + saved * HV + gate_head, raw_a)
                tl.store(B_SAVE + saved * HV + gate_head, raw_beta)
                if head % (HV // H) == 0:
                    tl.store(K_SAVE + (saved * H + kh) * DK + keys, raw_k, keys < DK)
        k = raw_k.to(tl.float32)
        v = raw_v.to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        q *= DK ** -0.5
        a = raw_a.to(tl.float32) + dt
        softplus = tl.where(a > 20., a, libdevice.log1p(tl.exp(a)))
        g = sa * softplus
        # Preserve the checkpoint compute dtype's sigmoid rounding before
        # FP32 recurrence, matching the existing materialized beta tensor.
        beta = tl.sigmoid(raw_beta.to(tl.float32)).to(BETA_IN.dtype.element_ty).to(tl.float32)
        state *= exp(g)
        v = beta * (v - tl.sum(state * k[:, None], 0))
        state += k[:, None] * v
        out = tl.sum(state * q[:, None], 0)
        tl.store(OUT + ((batch * T + token) * HV + value_head) * DV + values, out, values < DV)
        # Keep the optional pointer behind a constexpr-only branch: older
        # Triton versions type-check both sides of a mixed static/dynamic test.
        if SAVE_PREFIX:
            if token + 1 < T:
                tl.store(SNAPSHOTS + token * BATCH * HV * DK * DV + state_index, state, mask)
    if not SHARED_STATE:
        tl.store(STATE_OUT + state_index, state, mask)


class RecurrentReplay:
    """State before a speculative verification and the inputs the verification read. The state after any accepted prefix is
    replayed from them by the verification kernel itself, with the same arithmetic, instead of keeping one state snapshot per draft
    token: one extra state per layer instead of one per draft."""

    def __init__(self, state, heads, dim_k, tokens, input_dtype, gate_dtype):
        batch, value_heads, _, dim_v = state.shape
        self.tokens = tokens
        self.pre = torch.empty_like(state)
        self.k = state.new_empty((batch, tokens, heads, dim_k), dtype=input_dtype)
        self.v = state.new_empty((batch, tokens, value_heads, dim_v), dtype=input_dtype)
        self.a = state.new_empty((batch, tokens, value_heads), dtype=gate_dtype)
        self.b = state.new_empty((batch, tokens, value_heads), dtype=gate_dtype)
        self.out = torch.empty_like(self.v)  # replay outputs are not used

    def fits(self, state, tokens, input_dtype, gate_dtype):
        return (self.tokens >= tokens and self.pre.shape == state.shape and self.pre.dtype == state.dtype and self.pre.device == state.device
                and self.k.dtype == input_dtype and self.a.dtype == gate_dtype)


def recurrent_raw_gates(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots=None, replay=None,
                        *, v_heads_tiled=False, ssm_params_tiled=False, interleave_ab=False, shared_initial=False):
    """Read projection layouts directly; keep state and snapshots in grouped order. shared_initial: each batch entry starts from the state of
    batch 0 of `initial`, which stays as it is (the paths of a draft tree)."""
    batch, tokens, heads, dim_k = q.shape
    hv, dim_v = v.shape[-2:]
    output = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    if replay is not None and (replay.tokens < tokens or replay.k.dtype != k.dtype or replay.v.dtype != v.dtype or replay.a.dtype != a.dtype or replay.b.dtype != b.dtype):
        raise RuntimeError(f"Speculative replay buffers ({replay.tokens} tokens, {replay.k.dtype}/{replay.a.dtype}) do not match a {tokens}-token verification ({k.dtype}/{a.dtype}).")
    saving = (replay.pre, replay.k, replay.v, replay.a, replay.b) if replay is not None else (None,) * 5
    _recurrent_raw_kernel[(triton.cdiv(dim_v, 8), batch * hv)](
        q, k, v, a, b, ssm_a, ssm_dt, initial, initial, output, snapshots, *saving,
        tokens, heads, hv, dim_k, dim_v,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        *a.stride()[:2], *b.stride()[:2], batch,
        triton.next_power_of_2(dim_k), 8, snapshots is not None, replay is not None, replay.tokens if replay is not None else 0,
        v_heads_tiled, ssm_params_tiled, interleave_ab, shared_initial,
        num_warps=1, num_stages=3,
    )
    return output, initial


def recurrent_raw_replay(replay, ssm_a, ssm_dt, tokens, state, *, v_heads_tiled=False, ssm_params_tiled=False, interleave_ab=False):
    """Writes into `state` the state the last verification held after its first `tokens` inputs."""
    batch, _, heads, dim_k = replay.k.shape
    hv, dim_v = replay.v.shape[-2:]
    _recurrent_raw_kernel[(triton.cdiv(dim_v, 8), batch * hv)](
        replay.k, replay.k, replay.v, replay.a, replay.b, ssm_a, ssm_dt, replay.pre, state, replay.out, None, None, None, None, None, None,
        tokens, heads, hv, dim_k, dim_v,
        *replay.k.stride()[:3], *replay.k.stride()[:3], *replay.v.stride()[:3],
        *replay.a.stride()[:2], *replay.b.stride()[:2], batch,
        triton.next_power_of_2(dim_k), 8, False, False, 0,
        v_heads_tiled, ssm_params_tiled, interleave_ab, False,
        num_warps=1, num_stages=3,
    )


_portable_choices = {}
_MAX_PROBE_STATE_BYTES = 32 * 1024 * 1024


def _probe_key(kind, tensors, extra=()):
    return (kind, tuple(None if x is None else (x.device, x.dtype, tuple(x.shape), tuple(x.stride()))
                        for x in tensors), extra)


def _is_tracing(tensor):
    from torch._subclasses.fake_tensor import FakeTensor
    return isinstance(tensor, FakeTensor) or torch.compiler.is_compiling()


def _probe_blocked(tensor):
    return _is_tracing(tensor) or torch.cuda.is_current_stream_capturing()


def _record_choice(key, passed, reason='Numerical Check'):
    _portable_choices[key] = passed
    print(f"[Qwen][GDN] {key[0]}: {'Fused Kernel' if passed else 'Existing Kernel'} ({reason}).")


def _prepare_reference(query, key, a, b, ssm_a, ssm_dt, num_v_heads):
    repeat = num_v_heads // query.shape[2]
    return (query.repeat_interleave(repeat, 2), key.repeat_interleave(repeat, 2),
            ssm_a.float() * torch.nn.functional.softplus(a.float() + ssm_dt), b.sigmoid())


def prepare_decode_checked(query, key, a, b, ssm_a, ssm_dt, num_v_heads):
    args = (query, key, a, b, ssm_a, ssm_dt, num_v_heads)
    if _is_tracing(query):
        return _prepare_reference(*args)
    signature = _probe_key('Gate Preparation', args[:-1], (num_v_heads,))
    choice = _portable_choices.get(signature)
    if choice is True:
        return prepare_decode(*args)
    expected = _prepare_reference(*args)
    if choice is False or _probe_blocked(query):
        return expected
    # Probe only outside capture. No tensors are retained in the decision cache.
    from triton.runtime.errors import OutOfResources
    from triton.compiler.errors import CompilationError
    try:
        actual = prepare_decode(*args)
        for index, (result, reference) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(result, reference, rtol=1e-6 if index == 2 else 0,
                                       atol=1e-7 if index == 2 else 0)
    except AssertionError:
        _record_choice(signature, False)
    except (OutOfResources, CompilationError) as exc:
        _record_choice(signature, False, f'Kernel Unavailable: {exc}')
    else:
        _record_choice(signature, True)
    return expected


def _recurrent_reference(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots,
                         *, v_heads_tiled=False, ssm_params_tiled=False, interleave_ab=False):
    from shared.llm_engines.lmgp.layers.speculative_state import recurrent_verify
    # Preserve the established materialized path when a portable numerical
    # probe cannot use the direct-layout kernel, including during capture.
    from shared.llm_engines.lmgp.models.qwen3_5 import (
        _interleave_axis_halves, _reorder_v_head_axis_tiled_to_grouped,
        _reorder_v_head_axis_grouped_to_tiled, _reorder_v_heads_tiled_to_grouped,
    )
    heads, value_heads = q.shape[2], v.shape[2]
    if v_heads_tiled:
        v = _reorder_v_head_axis_tiled_to_grouped(v, 2, heads, value_heads)
        a = _reorder_v_heads_tiled_to_grouped(a, -1, heads, value_heads, 1)
        b = _reorder_v_heads_tiled_to_grouped(b, -1, heads, value_heads, 1)
    elif interleave_ab:
        a, b = (_interleave_axis_halves(x, -1) for x in (a, b))
    if ssm_params_tiled:
        ssm_a, ssm_dt = (_reorder_v_heads_tiled_to_grouped(x, 0, heads, value_heads, 1) for x in (ssm_a, ssm_dt))
    elif interleave_ab:
        ssm_a, ssm_dt = (_interleave_axis_halves(x, 0) for x in (ssm_a, ssm_dt))
    if snapshots is None and q.shape[1] > 1:
        snapshots = initial.new_empty((q.shape[1] - 1, *initial.shape))
    g = ssm_a.float() * torch.nn.functional.softplus(a.float() + ssm_dt)
    output, state = recurrent_verify(q, k, v, g, b.sigmoid(), initial, snapshots)
    if v_heads_tiled:
        output = _reorder_v_head_axis_grouped_to_tiled(output, 2, heads, value_heads)
    return output, state


def recurrent_raw_gates_checked(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots=None, replay=None,
                                *, v_heads_tiled=False, ssm_params_tiled=False, interleave_ab=False):
    if replay is not None:  # the layer passed probe_replay_verification: replays need the direct kernel's arithmetic
        return recurrent_raw_gates(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots, replay,
                                   v_heads_tiled=v_heads_tiled, ssm_params_tiled=ssm_params_tiled, interleave_ab=interleave_ab)
    args = (q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots)
    layout = dict(v_heads_tiled=v_heads_tiled, ssm_params_tiled=ssm_params_tiled, interleave_ab=interleave_ab)
    if _is_tracing(q):
        return _recurrent_reference(*args, **layout)
    signature = _probe_key('Recurrence', args, tuple(layout.values()))
    choice = _portable_choices.get(signature)
    if choice is True:
        return recurrent_raw_gates(*args, **layout)
    if choice is False or _probe_blocked(q):
        return _recurrent_reference(*args, **layout)
    prefix = snapshots[:q.shape[1] - 1] if snapshots is not None else None
    scratch_bytes = initial.numel() * initial.element_size()
    if prefix is not None:
        scratch_bytes += prefix.numel() * prefix.element_size()
    if scratch_bytes > _MAX_PROBE_STATE_BYTES:
        _record_choice(signature, False, 'Probe Memory Limit')
        return _recurrent_reference(*args, **layout)
    # Run the candidate on disposable state; only the established reference
    # advances live state during validation, including speculative prefixes.
    probe_state = probe_prefix = None
    try:
        probe_state = initial.clone()
        probe_prefix = torch.empty_like(prefix) if prefix is not None else None
    except RuntimeError as exc:  # torch.OutOfMemoryError, or the RuntimeError of the mmgp VRAM allocator (same message)
        if "CUDA out of memory" not in str(exc):
            raise
        probe_state = probe_prefix = None
        _record_choice(signature, False, 'Probe Allocation Limit')
        return _recurrent_reference(*args, **layout)
    from triton.runtime.errors import OutOfResources
    from triton.compiler.errors import CompilationError
    try:
        actual, _ = recurrent_raw_gates(*args[:7], probe_state, probe_prefix, **layout)
    except (OutOfResources, CompilationError) as exc:
        _record_choice(signature, False, f'Kernel Unavailable: {exc}')
        return _recurrent_reference(*args, **layout)
    expected, state = _recurrent_reference(*args, **layout)
    pairs = [(actual, expected), (probe_state, state)]
    if prefix is not None:
        pairs.append((probe_prefix, prefix))
    try:
        for result, reference in pairs:
            torch.testing.assert_close(result, reference, atol=2e-5,
                                       rtol=torch.finfo(reference.dtype).eps * 1.1)
    except AssertionError:
        _record_choice(signature, False)
    else:
        _record_choice(signature, True)
    return expected, state


_replay_choices = {}


def probe_replay_verification(block, max_verify_tokens):
    """Checked launches: whether a layer's speculative verification may use the direct kernel and replay accepted prefixes.
    Checked once per configuration, outside capture, with the verification's tensor layouts: the direct kernel must pass the
    numerical check of recurrent_raw_gates_checked against the established path, and replays must equal its snapshots exactly."""
    state = block.recurrent_state_buffer
    direct = getattr(block, "_gdn_direct_layout", True)
    layout = block._recurrence_layout(direct)
    projection = "qkv_gate" if block.attn_qkv_gate is not None else ("separate" if block.attn_gate_ab is None else "gate_ab")
    key = (state.device, state.dtype, int(max_verify_tokens), block.num_k_heads, block.head_k_dim, block.num_v_heads, block.head_v_dim, projection, tuple(layout.values()))
    if key in _replay_choices:
        return _replay_choices[key]  # decided on a layer of the same configuration, whether this one's weights are resident or streamed
    if block.ssm_a.device != state.device or block.ssm_dt.device != state.device:
        return False  # weights not resident yet: decided once the warm-up before graph capture has loaded them
    if key not in _replay_choices:
        _replay_choices[key] = _probe_replay(block, int(max_verify_tokens), direct, layout, projection)
        print(f"[Qwen][GDN] Speculative State Replay: {'Enabled' if _replay_choices[key] else 'Snapshots Kept'} (Numerical Check).")
    return _replay_choices[key]


def _probe_replay(block, max_verify_tokens, direct, layout, projection):
    from triton.runtime.errors import OutOfResources
    from triton.compiler.errors import CompilationError
    heads, dim_k, value_heads, dim_v, value_dim = block.num_k_heads, block.head_k_dim, block.num_v_heads, block.head_v_dim, block.value_dim
    state = block.recurrent_state_buffer[:1]
    device, dtype = state.device, state.dtype
    ssm_a, ssm_dt = block._recurrence_ssm_parameters(direct)
    generator = torch.Generator(device=device).manual_seed(0)
    replay = RecurrentReplay(state, heads, dim_k, max_verify_tokens, dtype, dtype)
    try:
        for tokens in range(2, max_verify_tokens + 1):
            # the verification's layouts: q/k/v split from the convolution output, gates from their projection
            mixed = torch.randn((1, tokens, 2 * heads * dim_k + value_heads * dim_v), device=device, dtype=dtype, generator=generator)
            q, k, v = torch.split(mixed, [heads * dim_k, heads * dim_k, value_heads * dim_v], dim=-1)
            q, k, v = q.reshape(1, tokens, heads, dim_k), k.reshape(1, tokens, heads, dim_k), v.reshape(1, tokens, value_heads, dim_v)
            if projection == "qkv_gate":
                a, b = (torch.randn((1, tokens, 2 * value_heads), device=device, dtype=dtype, generator=generator) * 2).chunk(2, dim=-1)
            elif projection == "gate_ab":
                _, a, b = torch.split(torch.randn((1, tokens, value_dim + 2 * value_heads), device=device, dtype=dtype, generator=generator) * 2, [value_dim, value_heads, value_heads], dim=-1)
            else:
                a, b = (torch.randn((1, tokens, value_heads), device=device, dtype=dtype, generator=generator) * 2 for _ in range(2))
            initial = (torch.randn(state.shape, device=device, dtype=torch.float32, generator=generator) * 0.3).to(dtype)
            expected_state, expected_prefix = initial.clone(), initial.new_empty((tokens - 1, *initial.shape))
            expected, _ = _recurrent_reference(q, k, v, a, b, ssm_a, ssm_dt, expected_state, expected_prefix, **layout)
            direct_state, prefix = initial.clone(), torch.empty_like(expected_prefix)
            actual, _ = recurrent_raw_gates(q, k, v, a, b, ssm_a, ssm_dt, direct_state, prefix, **layout)
            for result, reference in ((actual, expected), (direct_state, expected_state), (prefix, expected_prefix)):
                torch.testing.assert_close(result, reference, atol=2e-5, rtol=torch.finfo(reference.dtype).eps * 1.1)
            replayed = initial.clone()
            recurrent_raw_gates(q, k, v, a, b, ssm_a, ssm_dt, replayed, None, replay, **layout)
            if not torch.equal(replayed, direct_state):
                return False
            for accepted in range(1, tokens):
                target = torch.empty_like(initial)
                recurrent_raw_replay(replay, ssm_a, ssm_dt, accepted, target, **layout)
                if not torch.equal(target, prefix[accepted - 1]):
                    return False
    except (AssertionError, OutOfResources, CompilationError):
        return False
    return True


# Turing (sm_75): no BF16 conversion instructions, so the kernels above do not compile there. This one reads and writes BF16 tensors as
# their bits (RNE rounding as PyTorch's casts) and computes in FP32 with IEEE division, square root and exp; FP16 and FP32 tensors are
# read directly. Strides of the last q/k/v dimension are explicit: the PyTorch convolution of this route leaves them transposed.
@triton.jit
def _bf16_bits_to_fp32(bits):
    return (bits.to(tl.int32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _fp32_to_bf16_bits(x):
    bits = x.to(tl.uint32, bitcast=True)
    rounded = tl.where(x != x, 0x7FC0, (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)
    return rounded.to(tl.uint16).to(tl.int16, bitcast=True)


@triton.jit
def _load_fp32(pointer, mask, BITS: tl.constexpr):
    if BITS:
        return _bf16_bits_to_fp32(tl.load(pointer, mask, 0))
    return tl.load(pointer, mask, 0).to(tl.float32)


@triton.jit
def _load_scalar_fp32(pointer, BITS: tl.constexpr):
    if BITS:
        return _bf16_bits_to_fp32(tl.load(pointer))
    return tl.load(pointer).to(tl.float32)


@triton.jit
def _round_fp32(x, pointer, BITS: tl.constexpr):  # x rounded to the dtype of pointer's tensor
    if BITS:
        return _bf16_bits_to_fp32(_fp32_to_bf16_bits(x))
    return x.to(pointer.dtype.element_ty).to(tl.float32)


@triton.jit
def _store_fp32(pointer, value, mask, BITS: tl.constexpr):
    if BITS:
        tl.store(pointer, _fp32_to_bf16_bits(value), mask)
    else:
        tl.store(pointer, value.to(pointer.dtype.element_ty), mask)


@triton.jit(do_not_specialize=("T", "BATCH", "QS0", "QS1", "QS2", "QS3", "KS0", "KS1", "KS2", "KS3", "VS0", "VS1", "VS2", "VS3", "AS0", "AS1", "BS0", "BS1"),
            do_not_specialize_on_alignment=("T", "BATCH", "QS0", "QS1", "QS2", "QS3", "KS0", "KS1", "KS2", "KS3", "VS0", "VS1", "VS2", "VS3", "AS0", "AS1", "BS0", "BS1"))
def _recurrent_ieee_kernel(Q, K, V, A, BETA_IN, SSM_A, DT, STATE, OUT, SNAPSHOTS,
                           T, H: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                           QS0, QS1, QS2, QS3, KS0, KS1, KS2, KS3, VS0, VS1, VS2, VS3, AS0, AS1, BS0, BS1,
                           BATCH, BK: tl.constexpr, BV: tl.constexpr, SAVE_PREFIX: tl.constexpr,
                           V_HEADS_TILED: tl.constexpr, SSM_PARAMS_TILED: tl.constexpr, INTERLEAVE_AB: tl.constexpr,
                           BITS: tl.constexpr, SSM_BITS: tl.constexpr):
    vb, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HV, bh % HV
    kh = head // (HV // H)
    tiled_head = (head % (HV // H)) * H + kh
    interleaved_head = head // 2 + (head % 2) * (HV // 2)
    value_head = tiled_head if V_HEADS_TILED else head
    gate_head = tiled_head if V_HEADS_TILED else (interleaved_head if INTERLEAVE_AB and HV % 2 == 0 else head)
    param_head = tiled_head if SSM_PARAMS_TILED else (interleaved_head if INTERLEAVE_AB and HV % 2 == 0 else head)
    keys = tl.arange(0, BK)
    values = vb * BV + tl.arange(0, BV)
    state_index = bh * DK * DV + keys[:, None] * DV + values[None, :]
    mask = (keys[:, None] < DK) & (values[None, :] < DV)
    state = _load_fp32(STATE + state_index, mask, BITS)
    sa = _load_scalar_fp32(SSM_A + param_head, SSM_BITS)
    dt = _load_scalar_fp32(DT + param_head, SSM_BITS)
    for token in range(T):
        q = _load_fp32(Q + batch * QS0 + token * QS1 + kh * QS2 + keys * QS3, keys < DK, BITS)
        k = _load_fp32(K + batch * KS0 + token * KS1 + kh * KS2 + keys * KS3, keys < DK, BITS)
        v = _load_fp32(V + batch * VS0 + token * VS1 + value_head * VS2 + values * VS3, values < DV, BITS)
        a = _load_scalar_fp32(A + batch * AS0 + token * AS1 + gate_head, BITS) + dt
        raw_beta = _load_scalar_fp32(BETA_IN + batch * BS0 + token * BS1 + gate_head, BITS)
        q = tl.div_rn(q, tl.sqrt_rn(tl.sum(q * q) + 1e-6)) * DK ** -0.5
        k = tl.div_rn(k, tl.sqrt_rn(tl.sum(k * k) + 1e-6))
        g = sa * tl.where(a > 20., a, libdevice.log1p(libdevice.exp(a)))
        # the gate's sigmoid rounded to the checkpoint compute dtype, as the established paths materialize it
        beta = _round_fp32(tl.div_rn(1.0, 1.0 + libdevice.exp(-raw_beta)), BETA_IN, BITS)
        state *= libdevice.exp(g)
        v = beta * (v - tl.sum(state * k[:, None], 0))
        state += k[:, None] * v
        _store_fp32(OUT + ((batch * T + token) * HV + value_head) * DV + values, tl.sum(state * q[:, None], 0), values < DV, BITS)
        if SAVE_PREFIX:
            if token + 1 < T:
                _store_fp32(SNAPSHOTS + token * BATCH * HV * DK * DV + state_index, state, mask, BITS)
        # rounded to the state buffer's dtype after every token, as single-token decoding stores it: a verification of T tokens
        # gives the results of T decoded tokens (speculative decoding reproduces plain decoding)
        state = _round_fp32(state, STATE, BITS)
    _store_fp32(STATE + state_index, state, mask, BITS)


def recurrent_ieee(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots=None, replay=None,
                   *, v_heads_tiled=False, ssm_params_tiled=False, interleave_ab=False):
    """recurrent_raw_gates for GPUs without BF16 conversion instructions: same layouts and state handling (snapshots for the speculative
    prefixes; no replay buffers), the state updated in place."""
    if replay is not None:
        raise RuntimeError("The sm_75 recurrence keeps snapshots of speculative prefixes, not replay buffers.")
    batch, tokens, heads, dim_k = q.shape
    hv, dim_v = v.shape[-2:]
    output = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    bits = q.dtype == torch.bfloat16
    ssm_bits = ssm_a.dtype == torch.bfloat16
    as_bits = lambda x, flag=bits: None if x is None else (x.view(torch.int16) if flag else x)
    _recurrent_ieee_kernel[(triton.cdiv(dim_v, 8), batch * hv)](
        as_bits(q), as_bits(k), as_bits(v), as_bits(a), as_bits(b), as_bits(ssm_a, ssm_bits), as_bits(ssm_dt, ssm_bits), as_bits(initial),
        as_bits(output), as_bits(snapshots),
        tokens, heads, hv, dim_k, dim_v, *q.stride()[:4], *k.stride()[:4], *v.stride()[:4], *a.stride()[:2], *b.stride()[:2], batch,
        triton.next_power_of_2(dim_k), 8, snapshots is not None, v_heads_tiled, ssm_params_tiled, interleave_ab, bits, ssm_bits,
        num_warps=1, num_stages=3,
    )
    return output, initial


def _install_turing_recurrence(model, capability):
    """Decoding and speculative verification of the linear-attention layers on sm_75 (validated there only): the PyTorch recurrence
    (one launch per operation and token) replaced by _recurrent_ieee_kernel, at least as precise (FP32 throughout, the inputs read once)."""
    if capability != (7, 5):
        return
    blocks = [block for block in model.blk if block.layer_type == "linear_attention"]
    for block in blocks:
        block._gdn_recurrent_raw = recurrent_ieee
    print(f"[Qwen][GDN] Turing Recurrence Kernel For {len(blocks)} Layers (BF16 Read As Bits, FP32 IEEE Math).")


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
