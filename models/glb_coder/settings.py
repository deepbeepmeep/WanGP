# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
KV_LAYOUTS = {"int8": "INT8", "q6_q4": "Q6/Q4", "q4": "Q4"}
MIN_CONTEXT, MAX_CONTEXT = 1024, 262144
DEFAULT_SETTINGS = {"kv_cache_layout": "int8", "kv_cache_size": 64000, "save_source": 0}

CUSTOM_SETTINGS = [
    {"id": "kv_cache_layout", "name": "KV Cache Quantization", "label": "KV Cache Quantization", "type": "dropdown",
     "default": DEFAULT_SETTINGS["kv_cache_layout"], "choices": [(label, value) for value, label in KV_LAYOUTS.items()],
     "info": "INT8 keeps the most precision. Q6/Q4 uses 6-bit keys and 4-bit values. Q4 uses the least memory."},
    {"id": "kv_cache_size", "name": "KV Cache Size", "label": "KV Cache Size (tokens)", "type": "int",
     "default": DEFAULT_SETTINGS["kv_cache_size"], "min": MIN_CONTEXT, "max": MAX_CONTEXT, "inc": 256,
     "info": "Total tokens for the prompt, reference image and generated code. A larger window reserves more VRAM."},
    {"id": "save_source", "name": "Save Three.js Source", "label": "Save Three.js Source", "type": "dropdown",
     "default": DEFAULT_SETTINGS["save_source"], "choices": [("Off", 0), ("On", 1)],
     "info": "Include the exported scene's editable HTML. Batch sources are collected in one ZIP."},
]
