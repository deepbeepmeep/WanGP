# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Prompt-enhancer input/output routing and comma-separated chaining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


COMBINED_INPUT_SYSTEM_SUFFIX = (
    "The user input contains a JSON object with `prompt`, `alt_prompt`, and `requested_output`. "
    "Treat `prompt` and `alt_prompt` as input data, use both when relevant, and return only the new "
    "value for the field named by `requested_output`. Do not return JSON, field names, or an explanation."
)
ALT_PROMPT_INHERITANCE_ERROR = "This prompt enhancer chain produces one alt prompt per prompt, but the model does not declare alt_prompt_inherits_prompt_paragraphs=True."


@dataclass(frozen=True)
class PromptEnhancerStep:
    mode: str
    input_field: str
    output_field: str


@dataclass(frozen=True)
class PromptEnhancerResult:
    prompts: list[str]
    alt_prompts: list[str]
    prompt_changed: bool
    alt_prompt_changed: bool


def without_thinking(mode: str) -> str:
    return str(mode or "").replace("K", "")


def with_thinking(mode: str, enabled: bool) -> str:
    mode = without_thinking(mode)
    return f"{mode}K" if enabled and mode else mode


def normalize_choice(value: str, choices: list[str], default: str = "", require_choice: bool = False) -> str:
    mode = without_thinking(value)
    if mode in choices and (mode or not require_choice):
        return mode
    # Keep selections saved while resolution context used a mode flag.
    legacy_mode = mode.replace("R", "")
    if legacy_mode != mode and legacy_mode in choices and (legacy_mode or not require_choice):
        return legacy_mode
    if default in choices and (default or not require_choice):
        return default
    return next((choice for choice in choices if choice or not require_choice), "")


def split_mode(mode: str) -> list[str]:
    thinking = "K" in str(mode or "")
    parts = [without_thinking(part.strip()) for part in str(mode or "").split(",") if without_thinking(part.strip())]
    return [with_thinking(part, thinking) for part in parts]


def parse_mode(mode: str) -> list[PromptEnhancerStep]:
    steps = []
    for part in split_mode(mode):
        input_letters = [letter for letter in "TLB" if letter in part]
        if len(input_letters) > 1:
            raise ValueError(f"Prompt enhancer step '{without_thinking(part)}' must use only one of T, L, or B.")
        input_field = {"L": "alt_prompt", "B": "both"}.get(input_letters[0] if input_letters else "T", "prompt")
        steps.append(PromptEnhancerStep(part, input_field, "alt_prompt" if "O" in part else "prompt"))
    return steps


def uses_routing(mode: str) -> bool:
    mode = without_thinking(mode)
    return "," in mode or any(letter in mode for letter in "LBO")


def profile_suffix(mode: str) -> str:
    return next((letter for letter in str(mode or "") if letter.isdigit()), "0")


def originals_for_outputs(originals: list[str], output_count: int) -> list[str]:
    return originals * output_count if len(originals) == 1 else originals


def repeat_evenly(values: list[str], output_count: int) -> list[str]:
    if len(values) > output_count:
        raise ValueError(f"Cannot distribute {len(values)} alt prompts across {output_count} prompts.")
    repeats, remainder = divmod(output_count, len(values))
    return [value for index, value in enumerate(values) for _ in range(repeats + (remainder if index == len(values) - 1 else 0))]


def requires_alt_prompt_inheritance(mode: str) -> bool:
    return any(step.output_field == "alt_prompt" and step.input_field in ("prompt", "both") for step in parse_mode(mode))


def validate_alt_prompt_inheritance(mode: str, prompt_count: int, enabled: bool) -> str:
    return ALT_PROMPT_INHERITANCE_ERROR if prompt_count > 1 and requires_alt_prompt_inheritance(mode) and not enabled else ""


def validate_alt_prompt_count(prompt_count: int, alt_prompt_count: int, prompt_label: str, alt_prompt_label: str) -> str:
    return f'"{alt_prompt_label}" contains {alt_prompt_count} paragraphs, but "{prompt_label}" contains only {prompt_count}.' if alt_prompt_count > prompt_count else ""


def append_combined_input_contract(instructions: str | None, mode: str) -> str | None:
    if "B" not in str(mode or ""):
        return instructions
    return f"{str(instructions or '').rstrip()}\n\n{COMBINED_INPUT_SYSTEM_SUFFIX}".strip()


def _combined_inputs(prompts: list[str], alt_prompts: list[str], output_field: str) -> list[str]:
    count = max(len(prompts), len(alt_prompts))
    if len(alt_prompts) != count:
        alt_prompts = repeat_evenly(alt_prompts, count)
    if len(prompts) not in (1, count):
        raise ValueError(f"Cannot combine {len(prompts)} prompts with {len(alt_prompts)} alt prompts.")
    return [json.dumps({"prompt": prompts[0 if len(prompts) == 1 else index], "alt_prompt": alt_prompts[0 if len(alt_prompts) == 1 else index], "requested_output": output_field}, ensure_ascii=False) for index in range(count)]


def run_chain(mode: str, prompts: list[str], alt_prompts: list[str], alt_prompt_inherits_prompt_paragraphs: bool, enhance: Callable[[PromptEnhancerStep, list[str]], list[str]]) -> PromptEnhancerResult:
    steps = parse_mode(mode)
    current_prompts = list(prompts)
    current_alt_prompts = list(alt_prompts) or [""]
    prompt_changed = alt_prompt_changed = False
    inheritance_error = validate_alt_prompt_inheritance(mode, len(current_prompts), alt_prompt_inherits_prompt_paragraphs)
    if inheritance_error:
        raise ValueError(inheritance_error)

    for step in steps:
        if step.input_field == "prompt":
            inputs = current_prompts
        elif step.input_field == "alt_prompt":
            inputs = current_alt_prompts
        else:
            inputs = _combined_inputs(current_prompts, current_alt_prompts, step.output_field)
        outputs = list(enhance(step, inputs))
        if len(outputs) != len(inputs):  # not usable: the step leaves its field as it was
            print(f"[Prompt Enhancer] Step '{without_thinking(step.mode)}' returned {len(outputs)} outputs for {len(inputs)} inputs: its enhanced {step.output_field.replace('_', ' ')} is not usable, the previous one is kept.")
            continue
        if step.output_field == "prompt":
            current_prompts, prompt_changed = outputs, True
        else:
            current_alt_prompts, alt_prompt_changed = outputs, True

    return PromptEnhancerResult(current_prompts, current_alt_prompts, prompt_changed, alt_prompt_changed)
