# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence


@dataclass(slots=True)
class BackendEvent:
    kind: str
    text: str = ""
    data: dict[str, Any] | None = None


EventCallback = Callable[[BackendEvent], None]
ToolCallback = Callable[[str, dict[str, Any]], dict[str, Any]]
StopCallback = Callable[[], bool]


class RemoteBackend(Protocol):
    engine: str

    def run_turn(self, text: str, *, system_prompt: str, tools: Sequence[dict[str, Any]], images: Sequence[str], on_event: EventCallback, call_tool: ToolCallback, should_stop: StopCallback) -> str: ...
    def one_shot(self, text: str, *, system_prompt: str, images: Sequence[str], max_output_tokens: int, should_stop: StopCallback | None = None) -> str: ...
    def interrupt(self) -> None: ...
    def close(self) -> None: ...
