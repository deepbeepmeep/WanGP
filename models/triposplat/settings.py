"""TripoSplat tuning settings, shared by the handler (UI declarations) and the pipeline (defaults)."""

DEFAULT_GAUSSIAN_COUNT = 262144


def gaussian_count_setting(idol):
    choices = [("32K (Lightest File)", 32768), ("64K", 65536), ("128K", 131072), ("192K", 196608)]
    choices += [("256K", 262144), ("512K", 524288), ("1M (Most Detail)", 1048576)] if idol else [("256K (Most Detail)", 262144)]
    return {"id": "num_gaussians", "label": "Gaussian Count", "name": "Gaussians", "type": "dropdown", "default": DEFAULT_GAUSSIAN_COUNT,
            "choices": choices, "info": "More Gaussians keep finer details. Fewer give lighter files that display faster. "
                                       "Only the final decoding depends on the count."}
