# Adapted from Hy-Nomas (https://github.com/Tencent-Hunyuan/Hy-Nomas, hymm/pipeline/causal_ar.py), Copyright (C) 2026 Tencent, Apache License
# 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Hy-Nomas generation: the segments of a world are denoised one after the other (8 Euler steps of the distilled DiT, no guidance), each
attending to the keys and values of the previous segments; after a segment, one more forward over its latents lightly re-noised writes
its keys and values to the history (see history.py). The latents of the clip are then decoded by the PixelUMM decoder.

A world is returned with its .wgpworld file (world_file.py) and stays open as a session. Given a world file, generation continues that
world with the prompt's segments, or serves its interactive exploration (exploration.py): one move (held keys and / or a text event) at
a time, from the world's end, until the viewer leaves. A world that is not the open session is rebuilt from its file by replaying the
history writes of its segments."""
import copy
import io
import queue
import re
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import torch
from accelerate import init_empty_weights
from mmgp import offload
from PIL import Image
from tqdm import tqdm

from models.hyvideo.text_encoder.text_encoder_1_5 import PROMPT_TEMPLATE, TextEncoder as QwenTextEncoder
from shared.image_vae.output import float_to_video_bytes
from shared.utils import files_locator as fl
from shared.utils.phase_progress import check_abort, generation_progress, set_phase_status
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.world_exploration import EXPLORE_PROMPT, exploration_for
from shared.utils.world_instructions import split_instructions
from . import prompts as world_prompts
from . import world_file
from .camera import keyboard_poses, normalize_segment, resample, world_camera
from .exploration import MOVE_SECONDS, move_frames
from .history import HISTORY_FRAMES, WorldHistory
from .pixelumm import OVERLAP, PixelUMMDecoder
from .transformer import HEAD_DIM, HEADS, LATENT_CHANNELS, HYNomasTransformer, SegmentPlan, action_classes, rope_table

DECODER_FILE = "hy_nomas/pixelumm_decoder_fp8.safetensors"
TEXT_LENGTH = 1280  # text tokens after the template
FLOW_SHIFT = 5.0
CACHE_NOISE_SIGMA = 0.206897  # noise level of the latents that write a finished segment to the cache
SPATIAL_RATIO, TEMPORAL_RATIO, FPS = 32, 4, 24
UNCHANGED_SCENE = "Compared to the previous segment, the visible scene, background, and other-object layout remain unchanged."
CHANGE_PREFIX = "Compared to the previous segment, "
KEEP_DECODED_FRAMES = True  # the open world's decoded frames (its video, then each move) stay in RAM as uint8: Replay and Record reuse them


def _to_uint8(video):
    """(3, frames, height, width) in [-1, 1] -> uint8 (frames, height, width, 3) on the CPU."""
    return ((video.float() + 1) * 127.5).round().clamp_(0, 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


def _video_and_world_bytes(frames):
    """Clamped frames (3, n, height, width) as (6, n, height, width) bytes: WanGP's bytes of the video (shared.image_vae.output.
    float_to_video_bytes: the conversion WanGP applies to a float video), then the rounded bytes the world keeps (_to_uint8)."""
    return torch.cat([float_to_video_bytes(frames), ((frames.float() + 1) * 127.5).round().clamp_(0, 255).to(torch.uint8)])


def flow_sigmas(steps):
    sigmas = torch.linspace(1, 0, steps + 1, device="cpu")
    return FLOW_SHIFT * sigmas / (1 + (FLOW_SHIFT - 1) * sigmas)


def scene_change(text):
    """A text event as a segment's description change."""
    text = text.strip()
    return UNCHANGED_SCENE if not text else text if text.startswith(CHANGE_PREFIX) else CHANGE_PREFIX + text


def subject_action(text, subject):
    """A text event naming the world's subject (its display name, or the last word of it, with an article) as an action of that subject,
    the name replaced by the subject token as upstream's turns write actions; None when the event does not name it."""
    name = subject["display_name"]
    names = [re.escape(name)] + ([re.escape(name.split()[-1])] if " " in name else [])
    action, count = re.subn(r"\b(?:the |a |an )?(?:" + "|".join(names) + r")\b", subject["subject_id"], text.strip().rstrip("."), flags=re.IGNORECASE)
    return action if count else None


@dataclass
class WorldSession:
    """An open world: its manifest, history (archive in RAM, working set in a state ring while in use), and on the CPU the latents, text
    embeddings and camera poses of its stored segments; moves made since it was opened are not part of its file (Save World writes a
    new world file with them: _session_world_file)."""
    manifest: dict
    history: WorldHistory
    latents: torch.Tensor
    text_states: list
    camera: torch.Tensor | None
    moves: int = 0
    decoded_tail: torch.Tensor | None = None  # pixels of the overlap of the last move's last decoder window (the next move's head)
    world_latents: int = 0  # latent frames of the world's own video (the moves follow)
    world_frames: np.ndarray | None = None  # its decoded frames (KEEP_DECODED_FRAMES), uint8 (frames, height, width, 3)
    move_frames: list = field(default_factory=list)  # the frames of each move as shown (KEEP_DECODED_FRAMES)
    move_records: list = field(default_factory=list)  # each move as a segment of a saved world (_session_world_file)
    last_frame: torch.Tensor | None = None  # the last frame shown (3, 1, height, width) in [-1, 1] on the CPU: a saved world's still

    @property
    def frames(self):
        return self.latents.shape[2]


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, text_encoder_filename, save_quantized=False):
        self.device = torch.device("cuda")
        self.model_type = model_type
        model_filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        with init_empty_weights():
            transformer = HYNomasTransformer()
        # the camera modules' unused query projection and q / k norms are skipped (see NomasCameraModule)
        offload.load_model_data(transformer, model_filename, writable_tensors=False, default_dtype=torch.bfloat16, ignore_unused_weights=True)
        self.transformer = transformer.eval().requires_grad_(False)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, model_filename, torch.bfloat16, None)
        with init_empty_weights():
            decoder = PixelUMMDecoder()
        offload.load_model_data(decoder, fl.locate_file(DECODER_FILE), writable_tensors=False, default_dtype=torch.bfloat16)
        self.decoder = decoder.eval().requires_grad_(False)
        templates = copy.deepcopy(PROMPT_TEMPLATE)
        self.text_encoder = QwenTextEncoder(text_encoder_type="llm", max_length=TEXT_LENGTH, text_encoder_precision="bf16", tokenizer_type="llm",
                                            tokenizer_path=fl.locate_folder("Qwen2.5-VL-7B-Instruct"), prompt_template=templates["li-dit-encode-image-json"],
                                            prompt_template_video=templates["li-dit-encode-video-json"], hidden_state_skip_layer=2, apply_final_norm=False,
                                            reproduce=True, device="cpu", text_encoder_path=text_encoder_filename)
        self.text_cache = TextEncoderCache()
        self.session = None  # the open world (the last generated, continued or explored one)
        self._pending_write = None  # the history write of the last move, made after its clip is out (finish_move)
        self._interrupt = False

    def _token_ids(self, text):
        """Token ids of the valid text of a prompt after the template (the positions the DiT's text tokens have)."""
        tokens = self.text_encoder.text2tokens(text, data_type="video", max_length=TEXT_LENGTH)
        crop = self.text_encoder.prompt_template_video["crop_start"]
        return tokens["input_ids"][0, crop:][tokens["attention_mask"][0, crop:].bool()].tolist()

    def _encode(self, texts):
        tokens = self.text_encoder.text2tokens(texts[0], data_type="video", max_length=TEXT_LENGTH)
        output = self.text_encoder.encode(tokens, data_type="video", device=self.device)
        valid = int(output.attention_mask[0].sum())
        return [output.hidden_state[0, :valid].to(torch.bfloat16)]

    def _segment_layout(self, text, frames):
        """(text offsets of the actions, video windows of the actions) of a segment prompt, both None without per-action masking."""
        offsets = world_prompts.action_text_offsets(text, self._token_ids)
        return offsets, world_prompts.action_video_windows(text, frames) if offsets is not None else None

    @staticmethod
    def _plan(frames, states, offsets, windows, tokens_per_frame, history):
        image_tokens, text_tokens = frames * tokens_per_frame, len(states)
        classes = action_classes(image_tokens, offsets, windows, tokens_per_frame, history.length) if windows is not None else []
        return SegmentPlan(image_tokens, text_tokens, text_tokens if offsets is None else offsets[0], classes)

    def _denoise(self, latents, states, plan, rope, history, camera, sigmas, on_step):
        for step in tqdm(range(len(sigmas) - 1)):
            timestep = (sigmas[step:step + 1] * 1000).to(self.device)
            velocity = self.transformer(latents, timestep, states, rope, plan, history, camera=camera)
            latents = (latents.float() + velocity.float() * (sigmas[step + 1] - sigmas[step])).to(torch.bfloat16)
            del velocity
            on_step(step, latents)
            check_abort()
        return latents

    def _write(self, latents, eps, start, states, plan, rope, history, camera):
        """Stores a finished segment in the history: one forward over its latents re-noised to CACHE_NOISE_SIGMA."""
        cache_latents = ((1.0 - CACHE_NOISE_SIGMA) * latents.float() + CACHE_NOISE_SIGMA * eps.float()).to(torch.bfloat16)
        history.write_from(start, latents.shape[2], plan.image_tokens, plan.image_tokens + plan.kept_text)
        self.transformer(cache_latents, torch.full((1,), CACHE_NOISE_SIGMA * 1000.0, device=self.device), states, rope, plan, history, camera=camera, write=True)
        history.commit()

    def _decode_tail(self, latents, frames, head):
        """The pixels of a move, as the source's streaming decoder shows them: its last `frames` latent frames are decoded with the
        OVERLAP frames before them (windows of the timeline's stride grid, cross-faded with `head`, the pixels of the previous move's
        last window over these frames); the last frame, at the right edge of its window, comes out over-sharp and is held back until
        the next move, which shows it first. Returns (pixels (3, frames, height, width) in [-1, 1], the next move's head)."""
        start = max(0, latents.shape[2] - frames - OVERLAP)
        raw = self.decoder.decode(latents[:, :, start:].to(self.device), offset=start, head=head, clamp=False)
        first = (latents.shape[2] - frames - start - (head is not None)) * TEMPORAL_RATIO
        return raw[:, first:-TEMPORAL_RATIO].clamp(-1, 1), raw[:, -OVERLAP * TEMPORAL_RATIO:].clone()

    def _render(self, latents, step=4 * OVERLAP):
        """The whole video of latents as uint8 frames (frames, height, width, 3) on the CPU, decoded run by run as the explorer streams
        its moves (_decode_tail carries each run's last window into the next): the decoder never holds the whole video, and the last
        run keeps the frames a move holds back."""
        parts, head, end = [], None, 0
        while end < latents.shape[2]:
            frames = min(step, latents.shape[2] - end)
            video, head = self._decode_tail(latents[:, :, :end + frames], frames, head)
            parts.append(_to_uint8(video))
            end += frames
            check_abort()
        parts.append(_to_uint8(head[:, -TEMPORAL_RATIO:].clamp(-1, 1)))
        return np.concatenate(parts)

    def _render_session(self):
        """The whole video of the open world (its own video, then the moves) as uint8 frames: with KEEP_DECODED_FRAMES, the frames kept
        so far (the world's video is decoded once if it was not, and kept), otherwise everything decoded again."""
        session = self.session
        if not KEEP_DECODED_FRAMES:
            return self._render(session.latents)
        if session.world_frames is None:
            session.world_frames = self._render(session.latents[:, :, :session.world_latents])
        if not session.move_frames:
            return session.world_frames
        # the last move holds its last frames back for the next one: they end the video
        return np.concatenate([session.world_frames, *session.move_frames, _to_uint8(session.decoded_tail[:, -TEMPORAL_RATIO:].clamp(-1, 1))])

    @staticmethod
    def _last_frame_jpeg(video, size=None):
        """The clip's last frame as JPEG, within size x size (None: the world's resolution); video in [-1, 1], or its rounded bytes."""
        frame = video[:, -1] if video.dtype == torch.uint8 else ((video[:, -1].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        frame = frame.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(frame)
        if size is not None:
            image.thumbnail((size, size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85 if size else 92)
        return buffer.getvalue()

    def close_session(self):
        """Releases the open world (its history archive and working set)."""
        if self.session is not None:
            self.session.history.release_working_set()
        self.session = self._pending_write = None

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, seed, sampling_steps, frame_num, height, width, input_custom=None, callback=None, world_output=None, **kwargs):
        if input_custom is not None and input_prompt.strip() == EXPLORE_PROMPT:
            exploration = exploration_for(input_custom)
            return None if exploration is None else self._explore(exploration, callback)
        device = self.device
        base = self.open_world(input_custom) if input_custom is not None else None
        self.close_session()  # a continuation extends the base history: an interrupted one must not leave it as the open session
        if base is not None:
            height, width = base.manifest["height"], base.manifest["width"]
        description, instructions = split_instructions(input_prompt)
        if base is None and not description:
            raise ValueError("A new world needs a description before its exploration instructions")
        world = (world_prompts.parse_world(description) or world_prompts.plain_world(description)) if description else {"segments": []}
        if base is not None:  # the prompt continues the world: its segments are later segments, a long description becomes a change
            world = {**world, "segments": [{**segment, "description_change": segment.get("description_change") or scene_change(segment.get("visual_description_long") or "")}
                                           for segment in world["segments"]]}
        if world["segments"]:  # segments without timestamps share the selected video length
            pixel_boundaries = world_prompts.segment_boundaries(world) or [round(index * frame_num / len(world["segments"])) for index in range(len(world["segments"]) + 1)]
            pixel_boundaries = [boundary - pixel_boundaries[0] for boundary in pixel_boundaries]
            frame_num = pixel_boundaries[-1]
            local_boundaries = world_prompts.latent_boundaries(pixel_boundaries, (frame_num - 1) // TEMPORAL_RATIO + 1)
            segment_frames = [stop - start for start, stop in zip(pixel_boundaries, pixel_boundaries[1:])]
        else:
            frame_num, local_boundaries, segment_frames = 0, [0], []
        # each exploration instruction is one more segment, made as the explorer makes a move
        subject = ((world if base is None else base.manifest["world"]).get("subject_list") or [None])[0]
        for instruction in instructions:
            frames = self._move_frames(instruction["seconds"] or MOVE_SECONDS, instruction["event"])
            segment = {**self._move_segment(instruction["event"], subject, frames * TEMPORAL_RATIO / FPS), "instruction": instruction["text"]}
            if instruction["keys"]:
                segment["camera"] = {"keys": "+".join(instruction["keys"])}
            world["segments"].append(segment)
            local_boundaries.append(local_boundaries[-1] + frames)
            segment_frames.append(frames * TEMPORAL_RATIO)
            frame_num += frames * TEMPORAL_RATIO
        texts = world_prompts.segment_prompts(world) if base is None else world_prompts.segment_prompts({**world, "segments": [{}] + world["segments"]})[1:]
        offset = 0 if base is None else base.frames
        latent_frames, latent_height, latent_width = local_boundaries[-1], height // SPATIAL_RATIO, width // SPATIAL_RATIO
        boundaries = [offset + boundary for boundary in local_boundaries]
        tokens_per_frame = latent_height * latent_width
        camera, camera_on = world_camera(world["segments"], boundaries, segment_frames)
        if camera is not None:
            camera = camera.to(device)

        text_states = self.text_cache.encode(self._encode, texts, device=device)
        layouts = [self._segment_layout(text, boundaries[index + 1] - boundaries[index]) for index, text in enumerate(texts)]
        check_abort()

        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, LATENT_CHANNELS, latent_frames, latent_height, latent_width), generator=generator, dtype=torch.bfloat16, device="cpu").to(device)
        history = WorldHistory(self.transformer.double_blocks, HEADS, HEAD_DIM, device) if base is None else base.history
        try:
            if base is not None:
                history.load_working_set()
            sigmas = flow_sigmas(sampling_steps)
            if callback is not None:
                callback(-1, None, True, override_num_inference_steps=sampling_steps * len(texts))
            latents_out = torch.empty_like(noise)
            for index, (states, (offsets, windows)) in enumerate(zip(text_states, layouts)):
                set_phase_status(f"Generating Segment {index + 1}/{len(texts)}")
                start, frames = boundaries[index], boundaries[index + 1] - boundaries[index]
                local = start - offset
                plan = self._plan(frames, states, offsets, windows, tokens_per_frame, history)
                rope = rope_table(frames, latent_height, latent_width, start, device)
                segment_camera = camera[4 * local:4 * (local + frames)] if camera is not None and camera_on[index] else None

                def on_step(step, latents, index=index, local=local, frames=frames):
                    latents_out[:, :, local:local + frames] = latents
                    if callback is not None:
                        callback(index * sampling_steps + step, latents_out[0, :, :local + frames], False)

                latents = self._denoise(noise[:, :, local:local + frames].clone(), states, plan, rope, history, segment_camera, sigmas, on_step)
                # every segment, the last one included, goes to the history: the world can be explored from its end
                eps = torch.randn(latents.shape, generator=generator, dtype=torch.bfloat16, device="cpu").to(device)
                self._write(latents, eps, start, states, plan, rope, history, segment_camera)
                del rope, latents, eps, segment_camera
        except BaseException:  # interrupted: the working set's state ring leaves the blocks
            history.release_working_set()
            raise
        del noise

        world_output = world_output or "world"
        rounded = None
        latents_out = latents_out if base is None else torch.cat([base.latents.to(device), latents_out], dim=2)
        if world_output == "world":  # no video: only its last frames, for the world's still
            start = max(0, latents_out.shape[2] - 2 * OVERLAP)
            video = self.decoder.decode(latents_out[:, :, start:], offset=start)
        elif base is not None:  # a continuation's video is its new part, decoded after the continued world's last frames (no seam)
            video, head = self._decode_tail(latents_out, latent_frames, None)
            video = torch.cat([video, head[:, -TEMPORAL_RATIO:].clamp(-1, 1)], dim=1)[:, :frame_num]
            del head
        else:  # the whole video, to RAM window after window: WanGP's bytes, and the rounded bytes the world keeps (Replay / Record)
            decoded = self.decoder.decode_to_ram(latents_out, _video_and_world_bytes)
            video, rounded = decoded[:3], decoded[3:]
            del decoded
        # the world's whole video, kept for Replay / Record (before the length trim); a continuation's is decoded when they need it
        world_frames = rounded.permute(1, 2, 3, 0).numpy() if KEEP_DECODED_FRAMES and world_output == "world_video" and base is None else None
        history.release_working_set()
        if world_output == "video":
            self.close_session()
            return {"x": video[:, :frame_num]}
        generation = 0 if base is None else len(base.manifest["generations"])
        segments = [{"latent_start": boundaries[index], "latent_frames": boundaries[index + 1] - boundaries[index], "video_frames": segment_frames[index],
                     "text_offsets": offsets, "action_windows": windows, "camera": bool(camera_on and camera_on[index]), "generation": generation}
                    for index, (offsets, windows) in enumerate(layouts)]
        generations = [{"seed": seed, "latent_frames": latent_frames}]  # each generation drew its clip noise, then its history writes' noise
        text_states = [states.cpu() for states in text_states]
        camera = None if camera is None else camera.cpu()
        if base is None:
            video = video if world_output == "world" else video[:, :frame_num]
            manifest = {"model_type": self.model_type, "width": width, "height": height, "fps": FPS, "steps": sampling_steps, "world": world,
                        "generations": generations, "segments": segments, "history_frames": history.past_frames}
        else:
            manifest = {**base.manifest, "world": {**base.manifest["world"], "segments": base.manifest["world"]["segments"] + world["segments"]},
                        "generations": base.manifest["generations"] + generations, "segments": base.manifest["segments"] + segments}
            text_states = base.text_states + text_states
            if camera is not None or base.camera is not None:  # identity poses (camera off) where a part had none
                parts = [base.camera if base.camera is not None else torch.eye(4, device="cpu").expand(4 * base.frames, 4, 4),
                         camera if camera is not None else torch.eye(4, device="cpu").expand(4 * (latents_out.shape[2] - base.frames), 4, 4)]
                camera = torch.cat(parts)
        manifest["world_id"] = uuid.uuid4().hex
        latents_out = latents_out.cpu()
        # a small thumbnail for the gallery strip, the full resolution frame for the viewer's still
        last = video if rounded is None else rounded[:, :frame_num]  # (the rounded bytes of a whole video: the still's pixels as before)
        side_file = world_file.build(manifest, latents_out, text_states, camera, self._last_frame_jpeg(last, 320), self._last_frame_jpeg(last))
        self.session = WorldSession(manifest, history, latents_out, text_states, camera, world_latents=latents_out.shape[2], world_frames=world_frames)
        return {"x": video, "side_files": {world_file.EXTENSION: side_file}}

    @torch.inference_mode()
    def open_world(self, path):
        """Makes the world of a .wgpworld file the open session, rebuilding its history unless it is the open session already (a world
        just generated, not moved since): one history write per segment from its stored latents, text embeddings and camera poses,
        with the noise of its generation."""
        manifest, tensors = world_file.read(path)
        session = self.session
        if session is not None and session.manifest["world_id"] == manifest["world_id"] and session.moves == 0:
            return session
        self.close_session()
        device, latents = self.device, tensors["latents"]
        latent_height, latent_width = latents.shape[3:]
        tokens_per_frame = latent_height * latent_width
        camera = tensors.get("camera")
        text_states = [tensors[f"text_{index}"] for index in range(len(manifest["segments"]))]
        # worlds saved before the key existed were generated with the default history
        history = WorldHistory(self.transformer.double_blocks, HEADS, HEAD_DIM, device, manifest.get("history_frames", HISTORY_FRAMES))
        generator = generation = None
        try:
            for index, segment in enumerate(manifest["segments"]):
                set_phase_status(f"Preparing World {index + 1}/{len(manifest['segments'])}")
                start, frames = segment["latent_start"], segment["latent_frames"]
                if segment["generation"] != generation:  # each generation drew its clip noise, then the noise of its history writes
                    generation = segment["generation"]
                    drawn = manifest["generations"][generation]
                    generator = torch.Generator(device="cpu").manual_seed(drawn["seed"])
                    torch.randn((1, LATENT_CHANNELS, drawn["latent_frames"], latent_height, latent_width), generator=generator, dtype=torch.bfloat16, device="cpu")
                states = text_states[index].to(device)
                plan = self._plan(frames, states, segment["text_offsets"], segment["action_windows"], tokens_per_frame, history)
                rope = rope_table(frames, latent_height, latent_width, start, device)
                segment_camera = camera[4 * start:4 * (start + frames)].to(device) if segment["camera"] else None
                segment_latents = latents[:, :, start:start + frames].to(device)
                eps = torch.randn(segment_latents.shape, generator=generator, dtype=torch.bfloat16, device="cpu").to(device)
                self._write(segment_latents, eps, start, states, plan, rope, history, segment_camera)
                del states, rope, segment_camera, segment_latents, eps
                check_abort()
        except BaseException:  # interrupted: the working set's state ring leaves the blocks
            history.release_working_set()
            raise
        self.session = WorldSession(manifest, history, latents, text_states, camera, world_latents=latents.shape[2])
        return self.session

    def _move_states(self, prompt):
        return self.text_cache.encode(self._encode, [prompt], device=self.device)[0]

    _move_frames = staticmethod(lambda seconds, text: move_frames(seconds, bool(text)))  # exploration.move_frames

    @staticmethod
    def _move_segment(text, subject, seconds):
        """A move as a world segment: an unchanged scene, or a typed event. As upstream's turns keep the subject's actions apart from the
        scene changes, an event naming the world's subject becomes that subject's action over the whole move (`seconds`), another event
        a scene change."""
        action = subject_action(text, subject) if text and subject is not None else None
        actions = [] if action is None else [{"start_time": 0.0, "end_time": seconds, "action_description": f"<Action1>{action}</Action1>."}]
        return {"description_change": scene_change("" if action else text), "contained_actions": actions}

    @classmethod
    def _move_prompt(cls, text, subject=None, seconds=0.0):
        """The prompt of a move (see _move_segment)."""
        return world_prompts.segment_prompts({"segments": [{}, cls._move_segment(text, subject, seconds)]})[1]

    @torch.inference_mode()
    def move(self, keys, text="", seconds=1.0, on_step=None, steps=None, memory=None):
        """Continues the open world by one move: `seconds` of video where the camera follows held keys (camera.keyboard_poses spec,
        None for a still camera) and the scene follows an optional text event, in `steps` denoising steps (the distilled DiT supports
        4, 6 and 8; default: the world's), seeing `memory` history frames from now on (default: unchanged). Returns the new frames
        (3, frames, height, width) in [-1, 1]: those of the move but its last latent frame, after the frame the previous move held back
        (_decode_tail). Moves are not stored in the world file (a new one can be saved with them). The move's history write is left for
        finish_move (made by the next move otherwise), so that the clip can be shown first."""
        self.finish_move()
        session, device = self.session, self.device
        manifest = session.manifest
        latent_height, latent_width = manifest["height"] // SPATIAL_RATIO, manifest["width"] // SPATIAL_RATIO
        # whole decoder strides (5 latent frames, 0.83 s): each one costs exactly one more decoder window
        frames = self._move_frames(seconds, text)
        subject, length = (manifest["world"].get("subject_list") or [None])[0], frames * TEMPORAL_RATIO / FPS
        prompt = self._move_prompt(text, subject, length)
        states = self._move_states(prompt)
        offsets, windows = self._segment_layout(prompt, frames)
        start, history = session.frames, session.history
        if memory:
            history.set_past_frames(memory)
        history.load_working_set()
        plan = self._plan(frames, states, offsets, windows, latent_height * latent_width, history)
        rope = rope_table(frames, latent_height, latent_width, start, device)
        camera = None
        if keys:
            poses = normalize_segment(resample(keyboard_poses(keys, frames * TEMPORAL_RATIO), frames * TEMPORAL_RATIO))
            camera = torch.from_numpy(poses).to(device, torch.float32)
        seed = (manifest["generations"][-1]["seed"] * 1_000_003 + session.moves + 1) % (1 << 63)  # a saved world's last generation can be a move
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, LATENT_CHANNELS, frames, latent_height, latent_width), generator=generator, dtype=torch.bfloat16, device="cpu")
        latents = self._denoise(noise.to(device), states, plan, rope, history, camera, flow_sigmas(steps or manifest["steps"]), on_step or (lambda step, latents: None))
        eps = torch.randn(latents.shape, generator=generator, dtype=torch.bfloat16, device="cpu").to(device)
        self._pending_write = (latents, eps, start, states, plan, rope, history, camera)
        # in a saved world, the move is a segment of a generation of its own: its seed draws the same noise, then the same eps for its write
        session.move_records.append({"segment": {**self._move_segment(text, subject, length), **({"camera": keys} if keys else {})}, "states": states.cpu(),
                                     "offsets": offsets, "windows": windows, "camera": None if camera is None else camera.cpu(), "start": start, "frames": frames,
                                     "seed": seed})
        session.latents = torch.cat([session.latents, latents.cpu()], dim=2)
        session.moves += 1
        del rope, eps, camera
        video, session.decoded_tail = self._decode_tail(session.latents, frames, session.decoded_tail)
        session.last_frame = video[:, -1:].cpu()
        if KEEP_DECODED_FRAMES:
            session.move_frames.append(_to_uint8(video))
        return video

    def _session_world_file(self):
        """The open world as explored so far, as a new world file to explore or continue later: its segments, then each move as a segment
        of a generation of its own (its seed replays the noise of the move's history write), seeing the Memory chosen last; its still is
        the last frame shown."""
        session = self.session
        manifest, records, first = session.manifest, session.move_records, len(session.manifest["generations"])
        segments = [{"latent_start": record["start"], "latent_frames": record["frames"], "video_frames": record["frames"] * TEMPORAL_RATIO, "text_offsets": record["offsets"],
                     "action_windows": record["windows"], "camera": record["camera"] is not None, "generation": first + index} for index, record in enumerate(records)]
        manifest = {**manifest, "world": {**manifest["world"], "segments": manifest["world"]["segments"] + [record["segment"] for record in records]},
                    "generations": manifest["generations"] + [{"seed": record["seed"], "latent_frames": record["frames"]} for record in records],
                    "segments": manifest["segments"] + segments, "history_frames": session.history.past_frames, "world_id": uuid.uuid4().hex}
        camera = session.camera
        if camera is not None or any(record["camera"] is not None for record in records):  # identity poses where a part had none
            def identity(latent_frames):
                return torch.eye(4, device="cpu").expand(4 * latent_frames, 4, 4)
            camera = torch.cat([camera if camera is not None else identity(session.world_latents)]
                               + [record["camera"] if record["camera"] is not None else identity(record["frames"]) for record in records])
        return world_file.build(manifest, session.latents, session.text_states + [record["states"] for record in records], camera,
                                self._last_frame_jpeg(session.last_frame, 320), self._last_frame_jpeg(session.last_frame))

    @torch.inference_mode()
    def finish_move(self):
        """Makes the history write of the last move, if not done yet."""
        if self._pending_write is not None:
            pending, self._pending_write = self._pending_write, None
            self._write(*pending)

    @torch.inference_mode()
    def explore(self, world_path):
        """Serves the exploration of a world outside the queue (the Worlds gallery took the GPU from a running generation)."""
        exploration = exploration_for(world_path)
        if exploration is not None:
            self._explore(exploration, None)

    def _move_stats(self, seconds, frames, request):
        """Statistics the explorer shows (G): the whole GPU's VRAM use and load (NVML when available), this model's VRAM peak during the
        move, the system's and WanGP's RAM, the move's timing, and in seconds of video, video frames and moves of the length chosen in
        the viewer: what was explored (the world's own video, then the moves) and what the memory holds of it, against its size."""
        from shared.utils.stats import live_stats
        history = self.session.history
        default_move = move_frames(request["seconds"])

        def amount(latent_frames):  # latent frames as seconds, video frames and default moves
            return {"seconds": round(latent_frames * TEMPORAL_RATIO / FPS, 1), "frames": latent_frames * TEMPORAL_RATIO, "moves": round(latent_frames / default_move)}

        world = sum(segment["latent_frames"] for segment in self.session.manifest["segments"])
        # the move just made counts: its history write waits for the next move (finish_move), which sees it
        explored = history.total + (0 if self._pending_write is None else self._pending_write[0].shape[2])
        stats = {"move_seconds": round(seconds, 2), "video_seconds": round(frames / FPS, 2), "steps": request["steps"], "moves": self.session.moves,
                 "peak_gb": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 1), **live_stats(),
                 "world": amount(world), "explored_moves": amount(explored - world), "explored": amount(explored), "memory": amount(history.past_frames),
                 "memory_full": explored > history.past_frames, "room_moves": max(0, history.past_frames - explored) // default_move}
        return stats

    def _explore(self, exploration, callback):
        """Serves the moves of an exploration until the viewer leaves (or the task is aborted); nothing is saved."""
        try:
            set_phase_status("Preparing World")
            self.open_world(exploration.world_path)
            self._move_states(self._move_prompt(""))  # the text of moves without a typed event, before the first move
            exploration.publish("ready")
            while True:
                set_phase_status("Exploring World - Waiting for a Move")
                try:
                    request = exploration.next_request(0.25)
                except queue.Empty:
                    check_abort()
                    continue
                if request is None:
                    break
                if "render" in request:
                    set_phase_status("Exploring World - Rendering the Video")
                    exploration.publish("rendered", frames=self._render_session(), fps=FPS, record=request["render"], moves=self.session.moves)
                    continue
                if "save" in request:
                    set_phase_status("Exploring World - Saving the World")
                    exploration.publish("saved", data=self._session_world_file(), moves=self.session.moves)
                    continue
                set_phase_status(f"Exploring World - Move {self.session.moves + 1}")
                if callback is not None:
                    callback(-1, None, True, override_num_inference_steps=request["steps"])
                on_step = None if callback is None else lambda step, latents: callback(step, None, False)
                torch.cuda.reset_peak_memory_stats(self.device)
                started = time.perf_counter()
                video = self.move(request["keys"], request["text"], request["seconds"], on_step, request["steps"], request["memory"])
                frames = ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
                exploration.publish("clip", frames=frames, fps=FPS, stats=self._move_stats(time.perf_counter() - started, len(frames), request))
                del video
                self.finish_move()  # while the clip plays
        finally:
            exploration.publish("ended")
            self.close_session()
        return None
