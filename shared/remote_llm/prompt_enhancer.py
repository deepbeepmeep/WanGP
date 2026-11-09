# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from typing import Any, Sequence

from .images import temporary_image_paths
from .registry import create_backend
from shared.prompt_enhancer.templates import render_system_prompt
from shared.utils.cancellation import check_cancelled


IMAGE_ROLES = "\n\nThe labeled images belong only to this generation window. Respect the start and end anchors and the times of injected frames. Reference images guide appearance, not chronological order. Keep these roles distinct and do not invent absent anchors."


def enhance_prompt(engine: str, server_config: dict[str, Any], prompts: Sequence[str], images: Sequence[Any], *, instructions: str, max_output_tokens: int, image_contexts=None, prompt_durations=None, enhancement_progress=None, template_context=None, inject_duration=True, duration_instructions="") -> list[str]:
    """image_contexts (one per prompt) give each prompt its own images and their labels, named in the request in attachment order; without them every prompt receives images."""
    results = []
    for index, prompt in enumerate(prompts):
        if enhancement_progress is not None:
            enhancement_progress.prompt(index, len(prompts), max_output_tokens)
        labelled = [] if image_contexts is None else [(image, label) for image, label in zip(image_contexts[index].images, image_contexts[index].labels) if image is not None]
        backend = create_backend(engine, server_config)
        try:
            duration = prompt_durations[index] if prompt_durations is not None else None
            system_prompt = render_system_prompt(str(instructions or ""), template_context, duration, inject_duration=inject_duration, duration_instructions=duration_instructions)
            text = str(prompt or "")
            if labelled:
                system_prompt += IMAGE_ROLES
                text = "Attached images, in order: " + "; ".join(f"{number}. {label}" for number, (_, label) in enumerate(labelled, 1)) + "\n\n" + text
            def should_stop():
                try:
                    check_cancelled()
                except InterruptedError:
                    return True
                return False

            check_cancelled()
            with temporary_image_paths(images if image_contexts is None else [image for image, _ in labelled]) as image_paths:
                results.append(backend.run_turn(text, system_prompt=system_prompt, tools=[], images=image_paths, on_event=lambda event: None, call_tool=lambda name, args: {}, should_stop=should_stop))
            check_cancelled()
        finally:
            backend.close()
    return results
