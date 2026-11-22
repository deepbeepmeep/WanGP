# Copyright WanGP. Licensed under the WanGP License 2.0.
# DSpark block drafter of Inkling 2: plain PyTorch forward on the checkpoint's own tensors (mtp.safetensors).
# Algorithm after vLLM's Inkling 2 DSpark support (Apache 2.0), NVIDIA NeMo Automodel (Apache 2.0) and llama.cpp (MIT).
"""Eager DSpark drafter: the numerical reference of the engine's drafter.

A draft is one parallel pass over a block of 8 rows (the last accepted token and 7 masked slots) that attends the target's features
of every processed position, then a small recurrent head that turns the 7 slots into tokens one after the other."""
import torch

from .inkling2_core import rms, softcap


def _rope(x, positions, theta):
    """Half-split rotation of x [rows, heads, dim] by absolute positions."""
    half = x.shape[-1] // 2; inv = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angle = positions.float()[:, None, None] * inv; cos, sin = angle.cos(), angle.sin(); x = x.float(); a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)


class Inkling2DraftCore:
    def __init__(self, config, tensors, target):
        """config: the checkpoint's mtp_config dict; tensors: mtp.safetensors on one device; target: the Inkling2Core whose embedding and head it shares."""
        self.c, self.w, self.target = config, tensors, target
        self.heads, self.kv_heads, self.dim = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        self.block, self.window, self.local = config["block_size"], config["sliding_window_size"], set(config["local_layer_ids"])
        self.eps, self.theta = config["rms_norm_eps"], config["rope_theta"]
        self.alpha = torch.softmax(tensors["fuse_weight"].float(), -1)

    def _mixed(self, row, features, norm):
        return rms((self.alpha[row][:, None] * features.float()).sum(-2), self.eps, self.w[norm])

    def _normed(self, x, heads, weight, positions, dtype):
        return _rope(rms(x.view(-1, heads, self.dim), self.eps, self.w[weight]).to(dtype), positions, self.theta).to(dtype)

    @torch.no_grad()
    def hidden(self, anchor, features):
        """Block states [8, hidden] after the three layers. features: positions x 4 x hidden of the positions the target processed."""
        w, dtype = self.w, self.w["mask_embed"].dtype; p = features.shape[0]; rows = self.block
        h = torch.cat((self.target.embed(torch.tensor([anchor], device=features.device)), w["mask_embed"].expand(rows - 1, -1))).to(dtype)
        positions = p + torch.arange(rows, device=h.device); group = self.heads // self.kv_heads
        for i in range(self.c["num_hidden_layers"]):
            l = f"layers.{i}."; a = l + "self_attn."; first = max(0, p - self.window) if i in self.local else 0
            ctx = self._mixed(i, features[first:], l + "dflare_norm.weight").to(dtype); ctx_positions = first + torch.arange(ctx.shape[0], device=h.device)
            x = rms(h, self.eps, w[l + "input_layernorm.weight"]).to(dtype)
            q = self._normed(x @ w[a + "q_proj.weight"].T, self.heads, a + "q_norm.weight", positions, dtype)
            k = torch.cat((self._normed(ctx @ w[a + "k_proj_ctx.weight"].T, self.kv_heads, a + "k_norm_ctx.weight", ctx_positions, dtype), self._normed(x @ w[a + "k_proj.weight"].T, self.kv_heads, a + "k_norm.weight", positions, dtype)))
            v = torch.cat((ctx @ w[a + "v_proj_ctx.weight"].T, x @ w[a + "v_proj.weight"].T)).view(-1, self.kv_heads, self.dim)
            probs = torch.softmax((torch.einsum("qhd,khd->hqk", q, k.repeat_interleave(group, 1)) * self.dim ** -0.5).float(), -1).to(dtype)
            h = h + torch.einsum("hqk,khd->qhd", probs, v.repeat_interleave(group, 1)).reshape(rows, -1) @ w[a + "o_proj.weight"].T
            x = rms(h, self.eps, w[l + "post_attention_layernorm.weight"]).to(dtype)
            act = (torch.nn.functional.silu((x @ w[l + "mlp.gate_proj.weight"].T).float()) * (x @ w[l + "mlp.up_proj.weight"].T).float()).to(dtype)
            h = h + act @ w[l + "mlp.down_proj.weight"].T
        return rms(h, self.eps, w["norm.weight"]).to(dtype)

    @torch.no_grad()
    def draft(self, anchor, features, tokens=None, pick=None):
        """Drafted tokens (at most block - 1) after `anchor`, and their logits. pick: logits -> token (argmax by default)."""
        w = self.w; hid = self.hidden(anchor, features).float(); f_rnn = self._mixed(3, features[-1], "markov_head.dflare_norm.weight")
        rank = self.c["markov_rank"]; state = hid.new_zeros(rank); prev = anchor; out, logits = [], []
        proj, bias = w["markov_head.rnn_proj.weight"].float(), w["markov_head.rnn_proj.bias"].float(); unembed = self.target.w["unembed.weight"]
        for j in range(1, 1 + (tokens or self.block - 1)):
            raw = proj @ torch.cat((state, w["markov_head.markov_w1.weight"][prev].float(), hid[j], f_rnn)) + bias
            gate, cand, latent = torch.sigmoid(raw[:rank]), torch.tanh(raw[rank:2 * rank]), torch.tanh(raw[2 * rank:])
            state = gate * state + (1 - gate) * cand
            row = hid[j] + latent @ w["markov_head.hidden_out.weight"].float().T + w["markov_head.hidden_out.bias"].float()
            lg = softcap((row.to(unembed.dtype) @ unembed[: self.target.c.vocab].T).float(), self.target.c.logit_softcap)
            prev = int(lg.argmax()) if pick is None else int(pick(lg)); out.append(prev); logits.append(lg)
        return out, torch.stack(logits)
