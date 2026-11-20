import importlib.metadata
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib.metadata import version
from shared.sageattention_patch import sageattention_version

def clear_list(l):
    for i in range(len(l)):
        l[i] = None

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None

try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

try:
    from sageattention import sageattn_varlen
    def sageattn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ):
        return sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
except ImportError:
    sageattn_varlen_wrapper = None

try:
    from sageattention import sageattn
    @torch.compiler.disable()
    def sageattn_wrapper(
            qkv_list,
            attention_length
        ):
        q,k, v = qkv_list
        padding_length = q.shape[1] -attention_length
        q = q[:, :attention_length, :, : ]
        k = k[:, :attention_length, :, : ]
        v = v[:, :attention_length, :, : ]

        o = sageattn(q, k, v, tensor_layout="NHD")
        del q, k ,v
        clear_list(qkv_list)

        if padding_length > 0:
            o = torch.cat([o, torch.empty( (o.shape[0], padding_length, *o.shape[-2:]), dtype= o.dtype, device=o.device  ) ], 1)

        return o

except ImportError:
    sageattn = None


def get_attention_modes():
    ret = ["sdpa", "auto"]
    if flash_attn != None:
        ret.append("flash")
    if memory_efficient_attention != None:
        ret.append("xformers")
    if sageattn_varlen_wrapper != None:
        ret.append("sage")
    if sageattn != None and sageattention_version().startswith("2") :
        ret.append("sage2")

    return ret



MEMORY_LAYOUT = {
    "sdpa": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "xformers": (
        lambda x: x,
        lambda x: x,
    ),
    "sage2": (
        lambda x: x,
        lambda x: x,
    ),
    "sage": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}

@torch.compiler.disable()
def sdpa_wrapper(
        qkv_list,
        attention_length
    ):
    q,k, v = qkv_list
    padding_length = q.shape[2] -attention_length
    q = q[:, :, :attention_length, :]
    k = k[:, :, :attention_length, :]
    v = v[:, :, :attention_length, :]

    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=False
    )
    del q, k ,v
    clear_list(qkv_list)

    if padding_length > 0:
        o = torch.cat([o, torch.empty( (*o.shape[:2], padding_length, o.shape[-1]), dtype= o.dtype, device=o.device  ) ], 2)

    return o

def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def attention(
    qkv_list,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q , k , v = qkv_list
    clear_list(qkv_list)
    del qkv_list
    padding_length = 0
    # if attn_mask == None and mode == "sdpa": 
    #     padding_length  = q.shape[1] - cu_seqlens_q
    #     q = q[:, :cu_seqlens_q, ... ]
    #     k = k[:, :cu_seqlens_kv, ... ]
    #     v = v[:, :cu_seqlens_kv, ... ]

    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
        
    elif mode == "sdpa":
        # if attn_mask is not None and attn_mask.dtype != torch.bool:
        #     attn_mask = attn_mask.to(q.dtype)            
        # x = F.scaled_dot_product_attention(
        #     q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        # )
        assert attn_mask==None
        qkv_list = [q, k, v]
        del q, k , v
        x = sdpa_wrapper( qkv_list, cu_seqlens_q )

    elif mode == "xformers":
        x = memory_efficient_attention(
            q, k, v , attn_bias= attn_mask
        )

    elif mode == "sage2":
        qkv_list = [q, k, v]
        del q, k , v
        x = sageattn_wrapper(qkv_list, cu_seqlens_q)

    elif mode == "sage":
        x = sageattn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        x = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # reshape x to [b, s, a, d]      

    elif mode == "flash":
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        x = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert (
                attn_mask is None
            ), "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    if padding_length > 0 :
        out = torch.cat([out, torch.empty( (out.shape[0], padding_length, out.shape[2]), dtype= out.dtype, device=out.device  ) ], 1)

    return out


def parallel_attention(
    hybrid_seq_parallel_attn,
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    cu_seqlens_q,
    cu_seqlens_kv
):
    attn1 = hybrid_seq_parallel_attn(
        None,
        q[:, :img_q_len, :, :],
        k[:, :img_kv_len, :, :],
        v[:, :img_kv_len, :, :],
        dropout_p=0.0,
        causal=False,
        joint_tensor_query=q[:,img_q_len:cu_seqlens_q[1]],
        joint_tensor_key=k[:,img_kv_len:cu_seqlens_kv[1]],
        joint_tensor_value=v[:,img_kv_len:cu_seqlens_kv[1]],
        joint_strategy="rear",
    )
    if flash_attn.__version__ >= '2.7.0':
        attn2, *_ = _flash_attn_forward(
            q[:,cu_seqlens_q[1]:],
            k[:,cu_seqlens_kv[1]:],
            v[:,cu_seqlens_kv[1]:],
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    else:
        attn2, *_ = _flash_attn_forward(
            q[:,cu_seqlens_q[1]:],
            k[:,cu_seqlens_kv[1]:],
            v[:,cu_seqlens_kv[1]:],
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    attn = torch.cat([attn1, attn2], dim=1)
    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn
