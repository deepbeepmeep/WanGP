# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Kernel self-tests started as WanGP starts, one after the other in a thread of their own: they load the kernel libraries and run them
on the GPU (about 1.7 s) while WanGP goes on loading, and the backend choices read their results. A later call runs its probe again."""
import concurrent.futures

_pending = {}


def start():
    from shared.kernels import fp8_backend, int8_backend
    from shared.llm_engines.lmgp import vllm_support
    pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="kernel_probes")
    for name, probe in (("triton", vllm_support._run_triton_runtime_smoke), ("int8", int8_backend._selftest_wgpkernels), ("fp8", fp8_backend._selftest_wgp_fp8)):
        _pending[name] = pool.submit(probe)
    pool.shutdown(wait=False)


def result(name, probe):
    """The result of the probe started at startup, else probe()."""
    pending = _pending.pop(name, None)
    return probe() if pending is None else pending.result()
