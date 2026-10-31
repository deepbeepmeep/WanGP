# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from shared.lora_paths import load_lora_config
from shared.mcp.config import server_port


def _arg_provided(argv: Sequence[str], name: str) -> bool:
    return any(one_arg == name or str(one_arg).startswith(f"{name}=") for one_arg in argv)


def _arg_on_off(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in ("on", "yes", "true", "1"):
        return True
    if value in ("off", "no", "false", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected on, off, yes or no, got {raw!r}")


def parse_wgp_args(config_filename: str, argv: Sequence[str] | None = None):
    from shared.deepy.config import normalize_deepy_voice_language
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt or image using Gradio")
    add = parser.add_argument

    add("--save-masks", action="store_true", help="save proprocessed masks for debugging or editing")
    add("--save-speakers", action="store_true", help="save proprocessed audio track with extract speakers for debugging or editing")
    add("--debug-gen-form", action="store_true", help="View form generation / refresh time")
    add("--betatest", action="store_true", help="test unreleased features")
    add("--vram-safety-coefficient", type=float, default=0.85, help="max VRAM (between 0 and 1) that should be allocated to preloaded models")
    add("--share", action="store_true", help="Create a shared URL to access webserver remotely")
    add("--lock-config", action="store_true", help="Prevent modifying the configuration from the web interface")
    add("--lock-model", action="store_true", help="Prevent switch models")
    add("--save-quantized", action="store_true", help="Save a quantized version of the current model")
    add("--convrot", action="store_true", help="Save INT8 ConvRot instead of Quanto INT8 (requires --save-quantized)")
    add("--test", action="store_true", help="Load the model and exit generation immediately")
    add("--preload", type=str, default="0", help="Megabytes of each model to preload in VRAM, at least (initial budget for dynamic profiles 1/3; overrides the Configuration preload of every kind of output)")
    add("--multiple-images", action="store_true", help="Allow inputting multiple images with image to video")
    add("--loras", type=str, default="", help="Root folder for LoRAs (default: loras)")
    add("--lora-config", type=load_lora_config, default={}, metavar="FILE", help="JSON mapping LoRA subfolder names to directory paths")
    add("--check-loras", action="store_true", help="Filter Loras that are not valid")
    add("--lora-preset", type=str, default="", help="Lora preset to preload")
    add("--settings", type=str, default="settings", help="Path to settings folder")
    add("--config", type=str, default="", help=f"Path to config folder for {config_filename} and queue.zip")
    add("--profile", type=str, default=-1, help="Profile No")
    add("--verbose", type=str, default=1, help="Verbose level")
    add("--log-timestamp", dest="log_timestamp", type=_arg_on_off, nargs="?", const=True, default=False, metavar="on|off",
        help="Stamp every line WanGP prints to the console with the time of day, for instance '2026-10-12 15:06:41 | '. "
             "Default: off in every mode - a plain run prints exactly what it printed before. 'on' stamps this "
             "run's console, and the flag on its own means on. Nothing is written "
             "to a file: pipe it ('wgp.py ... 2>&1 | tee job.log') to keep a stamped log. "
             "WAN2GP_LOG_TIMESTAMP=1 sets the default to on for every run")
    add("--log-timestamp-format", dest="log_timestamp_format", type=str, default=None, metavar="FMT",
        help="Stamp format: full (the default, 'YYYY-MM-DD HH:MM:SS'), clock, date, ms, uptime, clock+uptime, "
             "any strftime pattern (e.g. the full preset) or a token mix (e.g. \"{time}.{ms} +{uptime}\"). "
             "The pattern is rendered literally, so brackets are available by writing them: \"[%%H:%%M:%%S]\"")
    add("--log-timestamp-sep", dest="log_timestamp_sep", type=str, default=None, metavar="SEP",
        help="Separator between the stamp and the message (default: pipe). 'none' separates with a single space")
    add("--debug-deepy", type=str, default=None, help="Enable Deepy verbose debug logging and write it to the given folder")
    add("--deepy-sessions-dir", type=str, default="", metavar="FOLDER", help="Store persistent Deepy sessions in FOLDER (default: ./deepy_sessions)")
    add("--workspaces-dir", type=str, default="", metavar="FOLDER", help="Workspace manifests and dedicated media folders (overrides Configuration / Outputs; default: ./workspaces)")
    add("--llm-io", type=str, default="", metavar="FOLDER", help="Write a plain-text transcript of all local and remote LLM input/output to FOLDER")
    add("--steps", type=int, default=0, help="default denoising steps")
    add("--frames", type=int, default=0, help="default number of frames")
    add("--seed", type=int, default=-1, help="default generation seed")
    add("--advanced", action="store_true", help="Access advanced options by default")
    add("--fp16", action="store_true", help="For using fp16 transformer model")
    add("--bf16", action="store_true", help="For using bf16 transformer model")
    add("--server-port", type=str, default=0, help="Server port")
    add("--theme", type=str, default="", help="set UI Theme")
    add("--perc-reserved-mem-max", type=float, default=0, help="Share of RAM that pinning may lock, as a fraction, e.g. 0.4 (default: the Configuration setting, else 0.4 on Windows and 0.6 on Linux)")
    # Development tools, not listed: reports of the VRAM alive at each phase's peak (allocations of MIN_MB and more) after each generation in
    # <outputs>/vram_debug, and of where the RAM goes (MIN_MB 16 when omitted) after each generation and model release in <outputs>/ram_debug
    add("--vram-debug", type=float, default=0, metavar="MIN_MB", help=argparse.SUPPRESS)
    add("--ram-debug", type=float, nargs="?", const=16, default=0, metavar="MIN_MB", help=argparse.SUPPRESS)
    add("--ram-allocator", choices=["default", "mmgp"], default=None, help="RAM allocator for CPU tensors, applied at startup: mmgp (MMGP RAM Allocator, the default: the RAM of freed tensors goes back to the system when the queue is done, a model is released or the RAM runs short) or default (PyTorch's, which keeps it for reuse). Overrides the Configuration setting")
    add("--max-vram", type=float, default=None, metavar="GB", help="Emulates a GPU with GB of VRAM, to try WanGP as on a smaller GPU: every VRAM figure is that card's and WanGP reserves at most GB - 1.2 (what a card leaves beside the driver and the desktop). The VRAM other programs use counts only when they leave less than that")
    add("--max-ram", type=float, default=None, metavar="GB", help="A RAM budget for WanGP: the RAM it reserves for fast transfers, keeps free and takes back from the models that are not running is planned within GB, counting what WanGP already uses. The RAM other programs use counts only when they leave less. The reserved RAM stays a share of the PC's real RAM, within the budget. A planning budget, not a hard limit")
    # Development tools, not listed: a PC with GB of RAM / a GPU with GB of VRAM under real pressure (helper processes lock or hold the rest),
    # OTHERS GB of it used by the system and other programs (6 GB of RAM, 1.5 GB of VRAM on Windows by default)
    add("--ram-stress", type=str, default=None, metavar="GB[:OTHERS]", help=argparse.SUPPRESS)
    add("--vram-stress", type=str, default=None, metavar="GB[:OTHERS]", help=argparse.SUPPRESS)
    add("--vram-allocator", choices=["default", "vmm", "vmm_spill"], default=None, help="VRAM allocator, applied at startup: vmm_spill (MMGP Optimized VRAM Allocator, the default, with RAM spilling: a generation slightly too large for the VRAM can finish, slowly), vmm (the same, an out of memory error when no VRAM is left) or default (PyTorch's). Overrides the Configuration setting")
    add("--prevent-power-throttling", action=argparse.BooleanOptionalAction, default=None, help="Windows: override Configuration / Performance to keep full CPU speed in the background (enabled by default; --no-prevent-power-throttling lets Windows save power instead)")
    add("--server-name", type=str, default="", help="Server name")
    add("--gpu", type=str, default="", help="Default GPU Device")
    add("--open-browser", action="store_true", help="open browser")
    # Deprecated model-selection shortcuts: keep accepting existing launch scripts.
    add("--t2v", action="store_true", help=argparse.SUPPRESS)
    add("--i2v", action="store_true", help=argparse.SUPPRESS)
    add("--t2v-14B", action="store_true", help=argparse.SUPPRESS)
    add("--t2v-1-3B", action="store_true", help=argparse.SUPPRESS)
    add("--vace-1-3B", action="store_true", help=argparse.SUPPRESS)
    add("--i2v-1-3B", action="store_true", help=argparse.SUPPRESS)
    add("--i2v-14B", action="store_true", help=argparse.SUPPRESS)
    add("--compile", action="store_true", help="Enable pytorch compilation")
    add("--listen", action="store_true", help="Server accessible on local network")
    add("--attention", type=str, default="", help="attention mode")
    add("--vae-config", type=str, default="", help="vae config mode")
    add("--process", type=str, default="", help="Process a saved queue (.zip) or settings file (.json) without launching the web UI")
    add("--offline", action="store_true", help="Never download anything: use only the files already on this computer; a missing file stops the operation with an error")
    add("--deepy-server", action="store_true", help="Serve the Deepy chat and media galleries without the Gradio UI")
    from shared.authentication import add_arguments
    add_arguments(parser)
    add("--deepy-voice-language", default=None, type=normalize_deepy_voice_language, help="Override the configured Whisper dictation language, e.g. fr, en or auto (default: Deepy configuration)")
    add("--ask-deepy", action="store_true", help="Start an interactive Deepy console session without launching the web UI")
    add("--mcp", action="store_true", help="Start WanGP as an MCP server without launching the web UI")
    add("--mcp-webui", action=argparse.BooleanOptionalAction, default=None, help="Override Configuration > Agents MCP autostart for this launch; --mcp-webui enables it and --no-mcp-webui disables it")
    add("--mcp-transport", type=str, default="", help="MCP transport: stdio (default with --mcp), sse, or streamable-http (default with the web UI)")
    add("--mcp-host", type=str, default="", help="Optional MCP host for non-stdio transports")
    add("--mcp-port", type=server_port, default=None, help="Override the Agents configuration MCP server port for HTTP transports (default: 8000)")
    add("--mcp-api-version", choices=("1", "2", "2.1"), default="2.1", help="WanGP MCP API version (latest: 2.1); pin 1 or 2 for earlier contracts")
    add("--mcp-async", action="store_true", help="Allow asynchronous generation/post-processing in MCP API v2/v2.1; disabled by default")
    add("--mcp-console-output", action="store_true", help="Mirror WanGP stdout/stderr while serving MCP requests")
    add("--mcp-allow-read-file-system", action="store_true", help="Allow MCP agents to reference arbitrary server filesystem paths; disabled by default")
    add("--dry-run", action="store_true", help="Validate file without generating (use with --process)")
    add("--output-dir", type=str, default="", help="Override output directory for CLI processing (use with --process)")
    add("--refresh-catalog", action="store_true", help="Refresh local plugin metadata for installed external plugins")
    add("--refresh-full-catalog", action="store_true", help="Refresh local plugin metadata for all catalog plugins")
    add("--merge-catalog", action="store_true", help="Merge plugins_local.json into plugins.json and remove plugins_local.json")

    args = parser.parse_args(argv)
    if args.llm_io:
        from shared.llm_io import configure_llm_io

        configure_llm_io(args.llm_io)
    if args.ask_deepy and not _arg_provided(argv, "--verbose"):
        args.verbose = "0"
    return args
