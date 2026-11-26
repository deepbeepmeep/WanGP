"""
https://github.com/jt-zhang/Sparse_SageAttention_API

Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from shared import sparge_core


def sparse_sageattn(qkv_list, mask_id=None, is_causal=False, tensor_layout="HND"):
    """Compatibility entry point for the bundled Triton block-map kernel."""
    if tensor_layout == "HND":
        return sparge_core.attention(qkv_list, mask_id, mode="sparge_triton", causal=is_causal)
    if tensor_layout != "NHD":
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    hnd = [x.transpose(1, 2) for x in qkv_list]
    qkv_list.clear()
    return sparge_core.attention(hnd, mask_id, mode="sparge_triton", causal=is_causal).transpose(1, 2)
