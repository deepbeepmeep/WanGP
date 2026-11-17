# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import os


def use_selector_event_loop() -> None:
    if os.name != "nt":
        return
    import asyncio
    import uvicorn.loops.asyncio as uvicorn_asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Uvicorn 0.36+ creates its loop directly, bypassing the application's policy.
    if hasattr(uvicorn_asyncio, "asyncio_loop_factory"):
        uvicorn_asyncio.asyncio_loop_factory = lambda use_subprocess=False: asyncio.SelectorEventLoop


def silence_proactor_connection_reset() -> None:
    if os.name != "nt":
        return

    from asyncio import proactor_events as _proactor

    transport_cls = _proactor._ProactorBasePipeTransport
    if getattr(transport_cls, "_wan2gp_patch", False):
        return

    original = transport_cls._call_connection_lost

    def _call_connection_lost(self, exc):
        if isinstance(exc, ConnectionResetError):
            exc = None
        try:
            return original(self, exc)
        except ConnectionResetError:
            return None

    transport_cls._call_connection_lost = _call_connection_lost
    transport_cls._wan2gp_patch = True
