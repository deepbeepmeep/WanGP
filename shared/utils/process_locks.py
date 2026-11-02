# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import time
import threading
import torch
from shared.utils.generation_state import set_generation_state

gen_lock = threading.Lock()
_MAIN_PROCESS_RUNNING_KEY = "main_process_running"
_PROCESS_NAMES_KEY = "process_names"

def get_gen_info(state):
    cache = state.get("gen", None)
    if cache == None:
        cache = dict()
        state["gen"] = cache
    return cache


def _main_generation_active_locked(gen):
    return bool(gen.get(_MAIN_PROCESS_RUNNING_KEY, False))


def set_main_generation_running(state, running):
    gen = get_gen_info(state)
    with gen_lock:
        if running:
            gen[_MAIN_PROCESS_RUNNING_KEY] = True
        else:
            gen.pop(_MAIN_PROCESS_RUNNING_KEY, None)

def main_generation_running(state):
    gen = get_gen_info(state)
    with gen_lock:
        return _main_generation_active_locked(gen)

def any_GPU_process_running(state, process_id, ignore_main = False, *, wait_for_lock = False):
    gen = get_gen_info(state)
#"process:" + process_id
    if not wait_for_lock and gen_lock.locked():
        return True
    with gen_lock:
        process_status = gen.get("process_status", None)
        if process_status == "process:main" and not _main_generation_active_locked(gen):
            return False
        return process_status is not None and not (process_status =="process:main" and ignore_main)


def _get_gpu_residents(gen):
    residents = gen.get("gpu_residents", None)
    if residents is None:
        residents = {}
        gen["gpu_residents"] = residents
    return residents


def _drop_gpu_resident_locked(gen, process_id):
    _get_gpu_residents(gen).pop(process_id, None)


def _collect_resident_release_actions_locked(gen, requester_id = None):
    release_actions = []
    residents = _get_gpu_residents(gen)
    for resident_id, resident_info in list(residents.items()):
        if resident_id == requester_id:
            residents.pop(resident_id, None)
            continue
        if not bool(resident_info.get("force_release_on_acquire", False)):
            continue
        release_callback = resident_info.get("release_vram_callback", None)
        if not callable(release_callback):
            residents.pop(resident_id, None)
            continue
        release_actions.append((resident_id, resident_info.get("process_name", resident_id), release_callback))
        residents.pop(resident_id, None)
    return release_actions


def _run_release_actions(release_actions):
    for resident_id, process_name, release_callback in release_actions:
        try:
            release_callback()
        except Exception as exc:
            print(f"[GPU] Unable to release resident VRAM for {process_name} ({resident_id}): {exc}")
    if len(release_actions) > 0 and torch.cuda.is_available():
        torch.cuda.synchronize()


def register_GPU_resident(state, process_id, process_name, release_vram_callback = None, force_release_on_acquire = True):
    gen = get_gen_info(state)
    with gen_lock:
        _get_gpu_residents(gen)[process_id] = {
            "process_name": process_name,
            "release_vram_callback": release_vram_callback,
            "force_release_on_acquire": bool(force_release_on_acquire),
        }


def unregister_GPU_resident(state, process_id):
    gen = get_gen_info(state)
    with gen_lock:
        _drop_gpu_resident_locked(gen, process_id)


def force_release_GPU_resident(state, process_id):
    gen = get_gen_info(state)
    release_callback = None
    with gen_lock:
        resident_info = _get_gpu_residents(gen).pop(process_id, None)
        if resident_info is not None:
            release_callback = resident_info.get("release_vram_callback", None)
    if callable(release_callback):
        release_callback()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def acquire_main_GPU_ressources(state):
    """Claim the GPU for the main generation. False when the user aborts while another process holds it."""
    gen = get_gen_info(state)
    release_actions = []
    waiting_on = None
    while True:
        with gen_lock:
            process_status = gen.get("process_status", None)
            if process_status is None or process_status == "process:main":
                release_actions = _collect_resident_release_actions_locked(gen, requester_id="main")
                set_generation_state(gen, process_status="process:main")
                break
            if gen.get("abort", False):
                return False
            if process_status != waiting_on:
                process_id = process_status.split(":", 1)[1]
                wait_msg = f"Media generation is waiting for {gen[_PROCESS_NAMES_KEY][process_id]} to release GPU resources..."
                waiting_on = process_status
            set_generation_state(gen, status=wait_msg)  # restores it when a queued status message has replaced it
        time.sleep(0.1)
    _run_release_actions(release_actions)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return True

def try_acquire_GPU_ressources(state, process_id, process_name):
    """Atomically claim an idle GPU, without suspending an active generation."""
    gen = get_gen_info(state)
    with gen_lock:
        status = gen.get("process_status")
        if status is not None and not (status == "process:main" and not _main_generation_active_locked(gen)):
            return False
        release_actions = _collect_resident_release_actions_locked(gen, requester_id=process_id)
        gen.setdefault("process_hierarchy", {})[process_id] = None
        gen.setdefault(_PROCESS_NAMES_KEY, {})[process_id] = process_name
        set_generation_state(gen, process_status="process:" + process_id)
    _run_release_actions(release_actions)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return True


def acquire_GPU_ressources(state, process_id, process_name, gr = None, custom_pause_msg = None, custom_wait_msg = None, parks = False):
    """parks: a generation running suspends at its next step with its models parked (their VRAM is reclaimed only as this process
    allocates it, see mmgp offload.park) rather than unloaded; for processes whose VRAM goes through PyTorch's allocator."""
    gen = get_gen_info(state)
    original_process_status = None
    release_actions = []
    while True:
        with gen_lock:
            process_hierarchy = gen.get("process_hierarchy", None)
            if process_hierarchy is None:
                process_hierarchy = dict()
                gen["process_hierarchy"]= process_hierarchy
            process_names = gen.get(_PROCESS_NAMES_KEY, None)
            if process_names is None:
                process_names = dict()
                gen[_PROCESS_NAMES_KEY] = process_names
            process_names[process_id] = process_name

            process_status = gen.get("process_status", None)
            if process_status is None:
                _drop_gpu_resident_locked(gen, process_id)
                original_process_status = None
                release_actions = _collect_resident_release_actions_locked(gen, requester_id=process_id)
                set_generation_state(gen, process_status="process:" + process_id)
                break
            elif process_status == "request:" + process_id and not _main_generation_active_locked(gen):
                _drop_gpu_resident_locked(gen, process_id)
                original_process_status = None
                release_actions = _collect_resident_release_actions_locked(gen, requester_id=process_id)
                set_generation_state(gen, process_status="process:" + process_id)
                break
            elif process_status == "process:main":
                if not _main_generation_active_locked(gen):
                    _drop_gpu_resident_locked(gen, process_id)
                    original_process_status = None
                    release_actions = _collect_resident_release_actions_locked(gen, requester_id=process_id)
                    set_generation_state(gen, process_status="process:" + process_id)
                    break
                original_process_status = process_status
                gen["park_request"] = parks
                set_generation_state(gen, process_status="request:" + process_id)

                gen["pause_msg"] = custom_pause_msg if custom_pause_msg is not None else f"Generation Suspended while using {process_name}" 
                break
            elif process_status == "process:" + process_id:
                _drop_gpu_resident_locked(gen, process_id)
                break
        time.sleep(0.1)

    _run_release_actions(release_actions)

    if original_process_status is not None:
        total_wait = 0
        wait_time = 0.1
        wait_msg_displayed = False
        while True:
            with gen_lock:
                process_status = gen.get("process_status", None)
                if process_status == "process:" + process_id:
                    break
                if process_status is None or (process_status == "request:" + process_id and not _main_generation_active_locked(gen)):
                    # handle case when main process has finished at some point in between the last check and now
                    set_generation_state(gen, process_status="process:" + process_id)
                    break

            total_wait += wait_time
            if round(total_wait,2) >= 5 and gr is not None and not wait_msg_displayed:
                wait_msg_displayed = True
                if custom_wait_msg is None:
                    gr.Info(f"Process {process_name} is Suspended while waiting that GPU Ressources become available")
                else:
                    gr.Info(custom_wait_msg)

            time.sleep(wait_time)

    with gen_lock:
        process_hierarchy[process_id] = original_process_status
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def release_GPU_ressources(state, process_id, keep_resident = False, process_name = None, release_vram_callback = None, force_release_on_acquire = True):
    gen = get_gen_info(state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with gen_lock:
        if keep_resident:
            _get_gpu_residents(gen)[process_id] = {
                "process_name": process_name or process_id,
                "release_vram_callback": release_vram_callback,
                "force_release_on_acquire": bool(force_release_on_acquire),
            }
        else:
            _drop_gpu_resident_locked(gen, process_id)
        process_hierarchy = gen.get("process_hierarchy", {})
        restore_status = process_hierarchy.pop(process_id, None)
        if restore_status == "process:main" and not _main_generation_active_locked(gen):
            restore_status = None
        set_generation_state(gen, process_status=restore_status)
        gen.get(_PROCESS_NAMES_KEY, {}).pop(process_id, None)
