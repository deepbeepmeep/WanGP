# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import enum
from tqdm import tqdm

__version__ = "4.5.0"

class profile_type(int, enum.Enum): 
    @staticmethod
    def tostr(v):
        if v == 0:
            s = "AutoRAM_AutoVRAM"
        elif v == 1:
            s= "HighRAM_HighVRAM" 
        elif v == 2:
            s ="HighRAM_LowVRAM"
        elif v == 3: 
            s = "LowRAM_HighVRAM"
        elif v == 4:
            s = "LowRAM_LowVRAM"
        else:
            s = "VerylowRAM_LowVRAM"
        return s
    
    AutoRAM_AutoVRAM  = 0
    HighRAM_HighVRAM  = 1
    HighRAM_LowVRAM  = 2
    LowRAM_HighVRAM  = 3
    LowRAM_LowVRAM  = 4
    VerylowRAM_LowVRAM  = 5

_welcome_message = f" Memory Management for the GPU Poor (mmgp {__version__}) - Copyright (c) 2024-2026 DeepBeepMeep "
tqdm.write(f"\033[1m\033[95m{_welcome_message:*^100}\033[0m")

import os as _os
if _os.environ.get("MMGP_VIRTUAL_VRAM_GB") or _os.environ.get("MMGP_VIRTUAL_RAM_GB"):  # a smaller GPU or machine emulated for simulations (mmgp.virtual_gpu)
    from . import virtual_gpu
    virtual_gpu.install_from_environment()

