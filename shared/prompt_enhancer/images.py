# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Window inputs and command preservation for prompt enhancement."""

from dataclasses import dataclass

from shared.remote_llm import is_remote_engine, resolve_role_engine
from shared.utils.frame_scheduler import SLASH_BLOCK_RE, build_default_window_plan, build_frame_scheduler, fit_video_length, floor_frame_count


MULTI_IMAGE_PROMPT_ENHANCEMENT = True
_florence_image_warning_shown = False


@dataclass
class ImageContext:
    images: list
    labels: list[str]
    duration_seconds: float | None = None


def enabled(server_config, mode):
    """Whether the enhancer receives every image of a prompt with its label: the local Qwen3.5 enhancers caption them, remote engines receive them."""
    enhancer = server_config.get("enhancer_enabled", 0)
    return MULTI_IMAGE_PROMPT_ENHANCEMENT and "I" in mode and (enhancer in (3, 4, 5, 6, 7) or enhancer > 0 and is_remote_engine(resolve_role_engine(server_config, "prompt_enhancer")))


def image_list(images):
    return [] if images is None else images if isinstance(images, list) else [images]


def control_image_input(inputs):
    return inputs.get("image_guide") if inputs["image_mode"] > 0 and "V" in inputs["video_prompt_type"] else None


def reference_name(number, main_reference=False):
    return "Main Ref. Image" if number == 1 and main_reference else f"Image reference no {number}"


def resolve_continue_video(video_source, image_prompt_type, file_list, output_dir):
    if "L" not in image_prompt_type:
        return video_source if "V" in image_prompt_type else None
    if file_list:
        return file_list[-1]
    import glob
    import os

    files = [path for extension in ("mp4", "mov", "mkv") for path in glob.glob(os.path.join(output_dir, f"*.{extension}"))]
    return max(files, key=os.path.getmtime) if files else None


def continuation_start_image(video_source, keep_frames, fps, max_frames):
    """Read only the last frame selected by generation's source resampler."""
    from PIL import Image
    from shared.utils.video_decode import _resample_frame_indices, decode_video_frames_ffmpeg, probe_video_stream_metadata

    metadata = probe_video_stream_metadata(video_source)
    if metadata is None:
        raise ValueError(f"Unable to read continuation video: {video_source}")
    limit = int(keep_frames) if str(keep_frames or "") else max_frames
    if limit < 0:
        limit = int(max(metadata["frame_count"] / metadata["fps_float"] * fps + limit, 0)) if metadata["fps_float"] > 0 else 0
    source_fps = metadata["fps"] if metadata["fps"] > 0 else max(1, round(metadata["fps_float"] or 0))
    indices = _resample_frame_indices(source_fps, metadata["frame_count"], limit, fps, 0)
    if not indices:
        raise ValueError("No frames remain in the continuation video after source trimming.")
    frames = decode_video_frames_ffmpeg(video_source, indices[-1], 1, bridge="numpy")
    if len(frames) == 0:
        raise ValueError("Unable to read the last retained continuation frame.")
    frame = frames[0]
    if frame.dtype.kind == "f":
        import torch
        from shared.utils.hdr import linear_to_srgb

        frame = linear_to_srgb(torch.from_numpy(frame)).mul(255).round_().clamp_(0, 255).to(torch.uint8).numpy()
    return Image.fromarray(frame)


def resolve_injected_positions(positions, image_count, *, windows, source_frames, source_overlap, window_size, discard_frames, reuse_frames, alignment_shift):
    """Resolve numeric frames and L/X against the actual generation window plan."""
    endpoints, guide_start = [], source_frames
    for index, window in enumerate(windows):
        overlap = source_overlap if index == 0 else window["overlap_frames"]
        end = guide_start - overlap + window["frame_num"] - window["discard_last_frames"] - window["trim_last_frames"] - 1
        endpoints.append(end)
        guide_start = end + 1
    result, window_no = [], 0
    for position in (positions or "").replace(",", " ").split():
        if position.upper() in ("L", "X"):
            # Explicit extra L/X slots refer to extension windows, which use the
            # configured default geometry after the planned windows finish.
            while window_no >= len(endpoints):
                endpoints.append(endpoints[-1] + window_size - reuse_frames - discard_frames)
            if position.upper() == "L":
                result.append(endpoints[window_no])
            window_no += 1
        else:
            result.append(int(position) - 1 + alignment_shift)
    return result[:image_count]


def window_contexts(windows, image_start, image_end, image_refs, *, fps, window_size, positions="", source_frames=0, reset_alignment=False, extract_from_window_start=False, discard_frames=0, reuse_frames=0, main_reference=False, control_image=None, continuation_image=None, picture_labels=False):
    references, ends = image_list(image_refs), image_list(image_end)
    if image_start is not None and windows[0].get("new_shot", False):
        source_frames = 0
    source_overlap = min(source_frames, max(1, windows[0]["overlap_frames"])) if not windows[0].get("new_shot", False) else 0
    alignment_shift = source_frames if reset_alignment else 0
    positions = resolve_injected_positions(positions, len(references), windows=windows, source_frames=source_frames, source_overlap=source_overlap, window_size=window_size, discard_frames=discard_frames, reuse_frames=reuse_frames, alignment_shift=alignment_shift)
    last_injection = {position: index for index, position in enumerate(positions)}
    contexts, end_index = [], 0
    guide_start, previous_end = source_frames, None
    for index, window in enumerate(windows):
        overlap = source_overlap if index == 0 else window["overlap_frames"]
        start_frame = guide_start - overlap
        end_frame = start_frame + window["frame_num"] - 1
        kept_end = end_frame - window["discard_last_frames"] - window["trim_last_frames"]
        start = (image_start if image_start is not None else continuation_image) if index == 0 else previous_end
        if window.get("new_shot", False):
            start = None
        no_end = window.get("no_end_image", False)
        end = None
        if not no_end and end_index < len(ends):
            end = ends[end_index]
            end_index += 1
        images, labels = ([control_image], ["Control Image"]) if control_image is not None else ([], [])
        if picture_labels:
            # Label every image with the <Picture N> the model receives, in its order: start / continuation frame, end image,
            # injected frames by time, reference images. Frames inside the overlap belong to the continuation and are dropped.
            time_origin, middle = start_frame + max(overlap - 1, 0), []
            injected = sorted(((position, image) for ref_no, (image, position) in enumerate(zip(references, positions), 1)
                               if last_injection[position] == ref_no - 1 and guide_start <= position <= end_frame), key=lambda item: item[0])
            for position, image in injected:
                if position == start_frame:  # a start image already owns frame 0
                    start = image if start is None else start
                elif position in (kept_end, end_frame) and end is None:
                    end = image
                else:
                    middle.append((image, f"injected frame at {(position - time_origin) / fps:g}s"))
            entries = ([(start, "start image")] if start is not None else []) + ([(end, "end image")] if end is not None else []) + middle
            entries += [(image, reference_name(ref_no, main_reference)) for ref_no, image in enumerate(references[len(positions):], len(positions) + 1)]
            # without a visible start image, <Picture 1> is still the frame continued from the previous window (or source video)
            first = 2 if start is None and not window.get("new_shot", False) and (index > 0 or source_overlap > 0) else 1
            images += [image for image, _ in entries]
            labels += [f"<Picture {first + number}>: {label}" for number, (_, label) in enumerate(entries)]
            contexts.append(ImageContext(images, labels, window["frame_num"] / fps))
            previous_end = end
            guide_start += window["frame_num"] - overlap - window["discard_last_frames"] - window["trim_last_frames"]
            continue
        first_injected = start_frame if extract_from_window_start else guide_start
        for ref_no, (image, position) in enumerate(zip(references, positions), 1):
            if last_injection[position] != ref_no - 1 or not first_injected <= position <= end_frame:
                continue
            timing = f"Frame {(position - start_frame) / fps:g}s/{window_size / fps:g}s"
            if position in (kept_end, end_frame):
                if end is None:
                    end = image
                    continue
                label = f"end image ({timing})"
            elif position == start_frame:
                start = image
                continue
            else:
                label = f"{reference_name(ref_no, main_reference)} ({timing})"
            images.append(image)
            labels.append(label)
        for ref_no, image in enumerate(references[len(positions):], len(positions) + 1):
            images.append(image)
            labels.append(reference_name(ref_no, main_reference))
        if start is not None:
            images.insert(0, start)
            labels.insert(0, "start image")
        if end is not None:
            images.append(end)
            labels.append("end image")
        contexts.append(ImageContext(images, labels, window["frame_num"] / fps))
        previous_end = end
        guide_start += window["frame_num"] - overlap - window["discard_last_frames"] - window["trim_last_frames"]
    return contexts


def prepare_auto(prompts, image_start, image_end, image_refs, *, windows, fps, window_size, positions, source_frames, reset_alignment, model_def, discard_frames, reuse_frames, multi_prompt_output, video=True, video_prompt_type="", control_image=None, continuation_image=None):
    if model_def.get("sliding_window", False) and video and not multi_prompt_output:
        contexts = window_contexts(windows, image_start, image_end, image_refs, fps=fps, window_size=window_size, positions=positions, source_frames=source_frames, reset_alignment=reset_alignment, extract_from_window_start=model_def.get("extract_guide_from_window_start", False), discard_frames=discard_frames, reuse_frames=reuse_frames, main_reference="K" in video_prompt_type, control_image=control_image, continuation_image=continuation_image, picture_labels=model_def.get("prompt_enhancer_picture_labels", False))
        # Repeated prompt text still needs the distinct anchors of each window.
        prompts = [prompts[index] if index < len(prompts) else strip_window_commands([prompts[-1]])[0][0] for index in range(len(windows))]
    else:
        first_window = [windows[0]]
        contexts = window_contexts(first_window, image_start, image_end, image_refs, fps=fps, window_size=window_size, positions=positions, source_frames=source_frames, reset_alignment=reset_alignment, extract_from_window_start=model_def.get("extract_guide_from_window_start", False), discard_frames=discard_frames, reuse_frames=reuse_frames, main_reference="K" in video_prompt_type, control_image=control_image, continuation_image=continuation_image, picture_labels=model_def.get("prompt_enhancer_picture_labels", False)) * len(prompts)
    if not video:
        for context in contexts:
            context.duration_seconds = None
    return prompts, contexts


def window_plan(prompts, inputs, model_def, *, fps, source_frames, has_start):
    """Sliding-window plan of a video generation: its windows (frames, overlap, trims), each window's prompt, the window size, default overlap and discarded frames."""
    minimum, step = model_def.get("frames_minimum", 5), model_def.get("frames_steps", 4)
    latent_size, offset = model_def.get("latent_size", step), model_def.get("frames_offset", 1)
    total_frames = fit_video_length(inputs["video_length"], minimum, latent_size, offset)
    window_size = floor_frame_count(inputs["sliding_window_size"], minimum, latent_size, offset)
    default_overlap = min(window_size - latent_size, inputs["sliding_window_overlap"])
    defaults = model_def.get("sliding_window_defaults", {})
    discard = inputs["sliding_window_discard_last_frames"]
    geometry = dict(minimum=minimum, step=step, frame_offset=offset, overlap_offset=defaults.get("overlap_offset", 1), max_overlap=defaults.get("overlap_max"), preserve_exact_output_frames=model_def.get("image_end_frame_position", False), output_frame_policy=model_def.get("frame_scheduler_output_policy"))
    scheduler, error = build_frame_scheduler(prompts, total_frames=total_frames, fps=fps, window_size=window_size, default_overlap=inputs["sliding_window_overlap"], supported_model_commands=model_def.get("prompt_slash_commands", []), allow_new_shot="T" in model_def.get("image_prompt_types_allowed", ""), first_window_overlap_frames=source_frames, initial_shared_frames=int(source_frames > 0), discard_last_frames=discard, **geometry)
    if error is not None:
        raise ValueError(error)
    if scheduler["active"]:
        windows = scheduler["windows"]
        expanded = [prompts[index] if index < len(prompts) else window["prompt"] for index, window in enumerate(windows)]
    else:
        first_overlap = min(1 if has_start else max(1, default_overlap), source_frames)
        windows = build_default_window_plan(total_frames=total_frames, window_size=window_size, default_overlap=default_overlap, discard_last_frames=discard, first_window_overlap=first_overlap, first_window_available_overlap=source_frames, initial_shared_frames=int(first_overlap > 0), **geometry)
        expanded = [prompts[min(index, len(prompts) - 1)] for index in range(len(windows))]
    return windows, expanded, window_size, default_overlap, discard


def prepare_manual(prompts, inputs, model_def, *, fps, source_frames, open_image, multi_prompt_output, continuation_image=None):
    window_mode = "W" in inputs["multi_prompts_gen_type"]
    if not window_mode and len(prompts) > 1:
        contexts = []
        for index, prompt in enumerate(prompts):
            current = dict(inputs)
            for key, flag in (("image_start", "S"), ("image_end", "E")):
                gallery_items = inputs[key] if flag in inputs["image_prompt_type"] else None
                if gallery_items and len(gallery_items) > 1:
                    if inputs["multi_images_gen_type"] != 1 or len(gallery_items) != len(prompts):
                        raise ValueError("Multiple anchor images require matching images and text prompts with equal counts.")
                    current[key] = gallery_items[index:index + 1]
            _, per_prompt = prepare_manual([prompt], current, model_def, fps=fps, source_frames=source_frames, open_image=open_image, multi_prompt_output=multi_prompt_output, continuation_image=continuation_image)
            contexts.extend(per_prompt)
        return prompts, contexts

    def gallery(key, active):
        return [open_image(item[0]) for item in inputs[key]] if active and inputs[key] else []

    starts = gallery("image_start", "S" in inputs["image_prompt_type"])
    ends = gallery("image_end", "E" in inputs["image_prompt_type"])
    references = gallery("image_refs", "I" in inputs["video_prompt_type"])
    if window_mode and len(starts) > 1:
        raise ValueError("Only one Start Image is supported in Sliding Window prompt modes.")
    sliding = model_def.get("sliding_window", False) and inputs["image_mode"] == 0 and not model_def.get("audio_only", False)
    start = starts[0] if starts else None
    if model_def.get("fake_start_image", False):
        start, source_frames = None, 0
    if sliding:
        windows, expanded, window_size, default_overlap, discard = window_plan(prompts, inputs, model_def, fps=fps, source_frames=source_frames, has_start=start is not None)
    else:
        window_size = fit_video_length(inputs["video_length"], model_def.get("frames_minimum", 5), model_def.get("latent_size", model_def.get("frames_steps", 4)), model_def.get("frames_offset", 1))
        default_overlap = discard = 0
        windows = [dict(frame_num=window_size, output_frames=window_size, overlap_frames=0, discard_last_frames=0, trim_last_frames=0)]
        expanded = prompts
    positions = inputs["frames_positions"] if "F" in inputs["video_prompt_type"] else ""
    control_image = control_image_input(inputs)
    common = dict(fps=fps, window_size=window_size, positions=positions, source_frames=source_frames, reset_alignment="T" in inputs["video_prompt_type"], extract_from_window_start=model_def.get("extract_guide_from_window_start", False), discard_frames=discard, reuse_frames=default_overlap, main_reference="K" in inputs["video_prompt_type"], control_image=open_image(control_image) if control_image is not None else None, continuation_image=continuation_image, picture_labels=model_def.get("prompt_enhancer_picture_labels", False))
    if sliding:
        contexts = window_contexts(windows, start, ends, references, **common)
        if multi_prompt_output or not window_mode:
            return prompts[:1], contexts[:1]
        return expanded, contexts
    if len(starts) > 1 and (inputs["multi_images_gen_type"] != 1 or len(starts) != len(prompts)):
        raise ValueError("Multiple Start Images require matching images and text prompts with equal counts.")
    contexts = [window_contexts(windows, starts[min(index, len(starts) - 1)] if starts else None, ends[index:index + 1] if len(ends) > 1 else ends, references, **common)[0] for index in range(len(prompts))]
    if inputs["image_mode"] > 0 or model_def.get("audio_only", False):
        for context in contexts:
            context.duration_seconds = None
    return prompts, contexts


def contexts_for_prompts(contexts, count):
    # Chained fields can contain one shared value or one value per window.
    if len(contexts) == count:
        return contexts
    if count == 1:
        return contexts[:1]
    if len(contexts) == 1:
        return contexts * count
    raise ValueError("Prompt and image-window counts do not match.")


def select_images(mode, image_start, image_refs, batch_images, contexts):
    if "I" not in mode:
        return [], None
    if contexts is not None:
        return [image for context in contexts for image in context.images], contexts
    images = list(image_list(image_start))
    if not batch_images:
        images = images[:1]
    if not batch_images or not images:
        images += image_list(image_refs)[:1]
    return [image for image in images if image is not None], None


def caption_input_info(image_start, image_end, image_refs, control_image=None, main_reference=False):
    starts, ends, refs = [[image for image in image_list(values) if image is not None] for values in (image_start, image_end, image_refs)]
    return dict(count=len(starts) + len(ends) + len(refs) + int(control_image is not None), anchor="Start Image" if starts else "End Image" if ends else "Control Image", main_reference=main_reference)


def warn_florence_images(server_config, mode, image_start, image_refs, batch_images, prompt_count, input_info=None):
    global _florence_image_warning_shown
    if _florence_image_warning_shown or "I" not in mode or server_config.get("enhancer_enabled", 0) not in (1, 2) or is_remote_engine(resolve_role_engine(server_config, "prompt_enhancer")):
        return
    info = input_info or caption_input_info(image_start, None, image_refs)
    if info["count"] <= 1:
        return
    anchors = [info["anchor"] if image is not None else None for image in image_list(image_start)]
    refs = [reference_name(index + 1, info.get("main_reference", False)) if image is not None else None for index, image in enumerate(image_list(image_refs))]
    selected, _ = select_images(mode, anchors, refs, batch_images, None)
    selected = list(dict.fromkeys(selected[:prompt_count]))
    if selected:
        print(f"Warning: Florence uses only one image caption per prompt. Selected: {', '.join(selected)}. Other images for the same prompt are ignored.")
        _florence_image_warning_shown = True


def strip_window_commands(prompts):
    return [SLASH_BLOCK_RE.sub("", prompt).strip() for prompt in prompts], [" ".join(match.group(0) for match in SLASH_BLOCK_RE.finditer(prompt)) for prompt in prompts]


def restore_window_commands(outputs, commands, mode="G"):
    separator = "\n" if "P" in mode or mode == "FG" else " "
    return [f"{command}{separator}{output}".strip() for command, output in zip(commands, outputs)]


def batch_kwargs(kwargs, index, batch_windows, count=1):
    if kwargs is None or batch_windows:
        return kwargs
    kwargs = dict(kwargs)
    if "image_contexts" in kwargs:
        kwargs["image_contexts"] = kwargs["image_contexts"][index:index + 1]
    callbacks = dict(kwargs.get("generation_callbacks") or {})
    if callbacks.get("enhancement_progress") is not None:
        callbacks["enhancement_progress"] = callbacks["enhancement_progress"].for_prompt(index, count)
        kwargs["generation_callbacks"] = callbacks
    return kwargs
