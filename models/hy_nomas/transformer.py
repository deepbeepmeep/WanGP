# Adapted from Hy-Nomas (https://github.com/Tencent-Hunyuan/Hy-Nomas, hymm/models/dit/nomas_cpt.py and hymm/models/camera/action_module.py),
# Copyright (C) 2026 Tencent, Apache License 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Hy-Nomas DiT: the HunyuanVideo 1.5 double-stream topology (54 blocks, 2048 wide, 16 heads of 128, 3D RoPE) generating the latent
frames of one segment at a time. Every segment attends to keys and values of the previous segments (a WorldHistory written by an extra
forward over each finished segment), and a camera module per block adds a per-frame term to the image tokens."""
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from models.hyvideo.modules.activation_layers import get_activation_layer
from models.hyvideo.modules.embed_layers import TimestepEmbedder
from models.hyvideo.modules.mlp_layers import MLP
from models.hyvideo.modules.modulate_layers import ModulateDiT
from models.hyvideo.modules.norm_layers import RMSNorm
from models.hyvideo.modules.token_refiner import SingleTokenRefiner
from shared.attention import pay_attention
from shared.attention_kit import joint_qkv_attention, q8_prefix
from shared.kernels import ops
from shared.utils.lora_mapping import convert_lora_keys
from shared.utils.phase_progress import check_abort

HIDDEN, HEADS, HEAD_DIM, DEPTH, LATENT_CHANNELS, TEXT_DIM = 2048, 16, 128, 54, 32, 3584
ROPE_DIMS, ROPE_THETA = (16, 56, 56), 256
CAMERA_DIM, CAMERA_HIDDEN, CAMERA_ATTENTION = 64, 128, 1024


def rope_table(frames, height, width, time_offset, device):
    """(1, frames * height * width, 1, 64, 2) cos / sin of the 3D RoPE (16 / 56 / 56 channels for t / h / w), interleaved channel pairs,
    time positions from time_offset (the latent frames generated before this segment)."""
    grids = torch.meshgrid(*(torch.arange(start, start + count, device=device, dtype=torch.float32)
                             for start, count in ((time_offset, frames), (0, height), (0, width))), indexing="ij")
    angles = torch.cat([torch.outer(grid.reshape(-1), 1.0 / ROPE_THETA ** (torch.arange(0, dim, 2, device=device).float() / dim))
                        for grid, dim in zip(grids, ROPE_DIMS)], dim=1)
    return torch.stack((angles.cos(), angles.sin()), dim=-1)[None, :, None]


@dataclass
class SegmentPlan:
    """Token layout of a segment: image tokens, text tokens, the text later segments see, and the per-action attention classes of the
    current segment, (query ranges, key ranges) over [image | text] queries and [history | image | text] keys; empty for full attention."""
    image_tokens: int
    text_tokens: int
    kept_text: int
    classes: list = field(default_factory=list)


def action_classes(image_tokens, text_offsets, video_windows, tokens_per_frame, history):
    """Attention classes of the intra-segment action mask: every query sees the history, the segment's video and the head of its text
    (before the actions); the video frames of action k and the text of action k also see that action's text, other queries do not."""
    head, img_keys = text_offsets[0], (history, history + image_tokens)
    head_keys = (history + image_tokens, history + image_tokens + head)
    groups = {}
    for start, stop, action in video_windows:
        groups.setdefault(action, []).append((start * tokens_per_frame, stop * tokens_per_frame))
    groups.setdefault(-1, []).append((image_tokens, image_tokens + head))
    for action in range(len(text_offsets) - 1):
        groups.setdefault(action, []).append((image_tokens + text_offsets[action], image_tokens + text_offsets[action + 1]))
    classes = []
    for action, queries in groups.items():
        keys = [(0, history), img_keys, head_keys] if history else [img_keys, head_keys]
        if action >= 0:
            keys.append((history + image_tokens + text_offsets[action], history + image_tokens + text_offsets[action + 1]))
        classes.append(([q for q in queries if q[1] > q[0]], [k for k in keys if k[1] > k[0]]))
    return [c for c in classes if c[0]]


def _gather(tensor, ranges):
    if len(ranges) == 1:
        return tensor[:, ranges[0][0]:ranges[0][1]]
    return torch.cat([tensor[:, start:stop] for start, stop in ranges], dim=1)


def _masked_attention(classes):
    """attention_kit attention_fn of the intra-segment action mask: one attention per class of queries over its visible keys."""
    def attention(qkv_list, group):
        query, key, value = qkv_list
        qkv_list.clear()
        output = torch.empty_like(query)
        for query_ranges, key_ranges in classes:
            keys, values = _gather(key, key_ranges), _gather(value, key_ranges)
            for start, stop in query_ranges:
                output[:, start:stop] = pay_attention([query[:, start:stop], keys, values])
            del keys, values
        return output
    return attention


class NomasDoubleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for stream in ("img", "txt"):
            setattr(self, f"{stream}_mod", ModulateDiT(HIDDEN, 6, get_activation_layer("silu")))
            setattr(self, f"{stream}_norm1", nn.LayerNorm(HIDDEN, elementwise_affine=False, eps=1e-6))
            for name in ("q", "k", "v"):
                setattr(self, f"{stream}_attn_{name}", nn.Linear(HIDDEN, HIDDEN, bias=True))
            setattr(self, f"{stream}_attn_q_norm", RMSNorm(HEAD_DIM, eps=1e-6))
            setattr(self, f"{stream}_attn_k_norm", RMSNorm(HEAD_DIM, eps=1e-6))
            setattr(self, f"{stream}_attn_proj", nn.Linear(HIDDEN, HIDDEN, bias=True))
            setattr(self, f"{stream}_norm2", nn.LayerNorm(HIDDEN, elementwise_affine=False, eps=1e-6))
            setattr(self, f"{stream}_mlp", MLP(HIDDEN, 4 * HIDDEN, act_layer=get_activation_layer("gelu_tanh"), bias=True))

    def forward(self, index, img, txt, vec, rope, plan, history_kv, writer, last):
        """img (image tokens, 2048) and txt (text tokens, 2048) updated in place. history_kv: this block's quantized keys / values of the
        history (WorldHistory.block_kv), None before the first segment. writer: the WorldHistory storing this segment's keys / values (the
        history write), else None; the last block of a write skips the outputs no later block reads."""
        write = writer is not None
        img_mod, txt_mod = self.img_mod(vec).chunk(6, dim=-1), self.txt_mod(vec).chunk(6, dim=-1)
        n_img = plan.image_tokens
        img_rows, txt_rows = [(0, img.shape[0], 0)], [(0, txt.shape[0], 0)]
        streams = [([ops.norm_modulate([img], self.img_norm1, img_mod[0], img_mod[1], img_rows)], (self.img_attn_q, self.img_attn_k, self.img_attn_v)),
                   ([ops.norm_modulate([txt], self.txt_norm1, txt_mod[0], txt_mod[1], txt_rows)], (self.txt_attn_q, self.txt_attn_k, self.txt_attn_v))]

        def norm_rope(query, key, group):
            for tensor, img_norm, txt_norm in ((query, self.img_attn_q_norm, self.txt_attn_q_norm), (key, self.img_attn_k_norm, self.txt_attn_k_norm)):
                if tensor is not None:
                    ops.interleaved_norm_rope_(tensor[:, :n_img], None, rope, img_norm, None)
                    ops.interleaved_norm_rope_(tensor[:, n_img:], None, None, txt_norm, None)
            if write and key is not None:
                writer.store(index, key, group.kv)

        def keep_value(value, group):
            writer.store(index, value, group.kv, values=True)

        attention = joint_qkv_attention(streams, HEADS, HEAD_DIM, norm_rope, kv_prefix=None if history_kv is None else q8_prefix(*history_kv), norm_v=keep_value if write else None,
                                        attention_fn=_masked_attention(plan.classes) if plan.classes else None)
        if last and write:
            return
        attention = attention.reshape(attention.shape[1], HIDDEN)
        ops.gated_residual_(img, [self.img_attn_proj(attention[:n_img])], img_mod[2], fused=True)
        ops.mlp([ops.norm_modulate([img], self.img_norm2, img_mod[3], img_mod[4], img_rows)], self.img_mlp.fc1, self.img_mlp.fc2, "gelu",
                residual=img, gate=img_mod[5])
        ops.gated_residual_(txt, [self.txt_attn_proj(attention[n_img:])], txt_mod[2], fused=True)
        del attention
        ops.mlp([ops.norm_modulate([txt], self.txt_norm2, txt_mod[3], txt_mod[4], txt_rows)], self.txt_mlp.fc1, self.txt_mlp.fc2, "gelu",
                residual=txt, gate=txt_mod[5])


class NomasCameraModule(nn.Module):
    """The camera conditioning of one block (upstream ActionModule, extrinsics_attn). Upstream cross-attends the image tokens of each latent
    frame to that frame's camera token alone (camera chunks of one frame): the softmax is 1 and the attention output is the token's value,
    so the module adds proj_keyboard(value(camera token)) to every image token of the frame. The query projection, the q / k norms and
    the RoPE of that attention do not change the result and are not loaded."""

    def __init__(self):
        super().__init__()
        self.keyboard_embed = nn.Sequential(nn.Linear(CAMERA_DIM, CAMERA_HIDDEN), nn.SiLU(), nn.Linear(CAMERA_HIDDEN, CAMERA_HIDDEN))
        self.keyboard_attn_kv = nn.Linear(CAMERA_HIDDEN, 2 * CAMERA_ATTENTION, bias=False)
        self.proj_keyboard = nn.Linear(CAMERA_ATTENTION, HIDDEN, bias=False)

    def forward(self, camera):
        """camera tokens (frames, 64) -> the term added to the image tokens of each frame (frames, 2048)."""
        return self.proj_keyboard(self.keyboard_attn_kv(self.keyboard_embed(camera))[..., CAMERA_ATTENTION:])


class NomasFinalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_final = nn.LayerNorm(HIDDEN, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(HIDDEN, LATENT_CHANNELS, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(HIDDEN, 2 * HIDDEN, bias=True))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear(ops.norm_modulate([x], self.norm_final, shift, scale, [(0, x.shape[0], 0)]))


class NomasPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(LATENT_CHANNELS, HIDDEN, kernel_size=1, stride=1)


class HYNomasTransformer(nn.Module):
    # the camera modules (54 small ones, ~260 MB in all), run between the double blocks while a camera is driven, stay in VRAM with the
    # base: streamed as blocks of their own, each one interrupted the transfers ahead of the double blocks (profiles that stream)
    _offload_base_modules = ("action_modules", "extrinsics_embedding_camctrl", "c2ws_hidden_states_layer1", "c2ws_hidden_states_layer2")

    def __init__(self):
        super().__init__()
        self.img_in = NomasPatchEmbed()
        self.txt_in = SingleTokenRefiner(TEXT_DIM, HIDDEN, HEADS, depth=2)
        self.time_in = TimestepEmbedder(HIDDEN, get_activation_layer("silu"))
        self.double_blocks = nn.ModuleList(NomasDoubleBlock() for _ in range(DEPTH))
        self.final_layer = NomasFinalLayer()
        self.action_modules = nn.ModuleList(NomasCameraModule() for _ in range(DEPTH))
        self.extrinsics_embedding_camctrl = nn.Linear(CAMERA_DIM, CAMERA_DIM)
        self.c2ws_hidden_states_layer1 = nn.Linear(CAMERA_DIM, CAMERA_DIM)
        self.c2ws_hidden_states_layer2 = nn.Linear(CAMERA_DIM, CAMERA_DIM)
        # These hold no weights themselves: their locks keep MMGP from rounding the float32 timestep to BF16 before the FP32 sinusoids,
        # and the root's lock keeps the RoPE table FP32 (read in FP32 by the norm + RoPE op, as upstream); latents and camera poses are cast in forward
        for module in (self, self.time_in, self.txt_in):
            module._lock_dtype = torch.float32

    @staticmethod
    def _auto_preload_workload(args, kwargs):
        """Workload of MMGP's automatic VRAM preload (arguments of the first double block): the image tokens of the segment. The history
        and the visible text change a little at every move, not enough to measure the activations again (with the budget's preload)."""
        return args[1].shape[0]

    def preprocess_loras(self, model_type, state_dict):
        # adapters of the camera modules' query projection cannot change the result (see NomasCameraModule), which is not loaded
        state_dict = {key: value for key, value in state_dict.items() if not ("action_modules." in key and ".img_attn_q." in key)}
        return convert_lora_keys(state_dict, dict(self.named_modules()))

    def camera_tokens(self, camera):
        """Normalized camera-to-world poses (4 per latent frame, 4, 4) -> one camera token per latent frame (latent frames, 64), the poses
        flattened by groups of 4 (source _preprocess_camera)."""
        tokens = self.extrinsics_embedding_camctrl(camera.to(self.extrinsics_embedding_camctrl.weight.dtype).reshape(-1, 4 * 16))
        return tokens + self.c2ws_hidden_states_layer2(torch.nn.functional.silu(self.c2ws_hidden_states_layer1(tokens)))

    def forward(self, latents, timestep, text_states, rope, plan, history, camera=None, write=False):
        """latents (1, 32, frames, height, width) of the current segment -> the predicted velocity (same shape), or None with write, where
        the forward only stores the segment's keys and values in the history record opened by the caller. text_states (text tokens, 3584): the valid tokens of the
        segment's prompt. camera: the segment's normalized camera poses (4 per latent frame, 4, 4), None without camera control."""
        _, _, frames, height, width = latents.shape
        dtype = self.img_in.proj.weight.dtype
        img = self.img_in.proj(latents.to(dtype)).flatten(2)[0].transpose(0, 1).contiguous()
        vec = self.time_in(timestep)
        txt = self.txt_in(text_states[None].to(dtype), timestep)[0]
        camera = None if camera is None else self.camera_tokens(camera)
        for index, block in enumerate(self.double_blocks):
            block(index, img, txt, vec, rope, plan, history.block_kv(index), history if write else None, last=index == DEPTH - 1)
            if write:
                history.flush(index)
            if camera is not None and not (write and index == DEPTH - 1):
                img.view(frames, height * width, HIDDEN).add_(self.action_modules[index](camera)[:, None])
            check_abort()
        if write:
            return None
        out = self.final_layer(img, vec)
        return out.transpose(0, 1).reshape(1, LATENT_CHANNELS, frames, height, width)
