# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import VIDEO_PROMPT, VIDEO_PROMPT_SHORT, help_text, specialities


def longcat_help(architecture, version=None, one_speaker_only=False, multi_speakers_only=False):
    avatar = architecture != "longcat_video"
    info = "Generate from text, animate a start image, or continue a source video. **Sliding Window** builds a longer video in successive sections. Keep subject and scene descriptions consistent, and inspect joins before extending further. **Auto** sampling selects the continuation behavior when needed."
    compact = "Text/image-to-video and source-video continuation. Sliding windows extend output; inspect joins and drift. Auto sampling chooses continuation behavior."
    if avatar:
        info = "Animate a character using **Voice to Follow**. The recording supplies the words and timing; the prompt describes appearance, expression, gestures and the scene."
        if multi_speakers_only:
            info += " This version is for two speakers. Choose how their voices are supplied and use **Speakers Locations** to match each voice to the correct person."
        elif not one_speaker_only:
            info += " For two speakers, choose a two-voice mode and use **Speakers Locations** to match each voice to the correct person."
        info += "\n\n### Start Image and Reference Image\n\n**Start Image** sets the opening frame and composition.\n\n**Anchor Reference Image** provides appearance guidance. Used on its own, it also becomes the opening frame. With a separate start image or source video, it adds visual guidance alongside that starting scene.\n\nUse **one reference image**. LongCat does not combine several portraits, outfits or viewpoints."
        if not one_speaker_only:
            info += " For two speakers, use one image showing both people and set their speaker locations; separate portraits are not combined automatically."
        info += "\n\n### Longer Videos\n\n**Sliding Window** extends the performance in successive sections. Each new section starts from the ending frames of the preceding section. Your original reference image remains the appearance guide throughout the video. If you do not provide one, the first frame becomes the appearance guide. Supply audio for the full duration, keep the scene description consistent, and check the joins and the characters' appearance."
        compact = "Audio-driven character video; audio supplies words/timing, prompt visual behavior. Optional image anchor/start frame; assign two voices to their speakers. Sliding windows support long performances."
        compact += " One reference image only; no multi-image fusion or portrait-per-speaker inputs. Reference alone supplies the opening frame; with a start image/source video it provides additional appearance guidance. An explicit reference remains the anchor for all windows; without one, the first frame is reused. For two speakers use one image containing both people."
        if one_speaker_only:
            compact += " This variant supports one speaker only."
        elif multi_speakers_only:
            compact += " This variant requires two speakers."
        if version == "1.6":
            compact += " Avatar 1.6: 8 steps, sample_solver='distill', guidance_scale=audio_guidance_scale=1; 25 fps, sliding_window_size=93, sliding_window_overlap=13, discard=0. The supplied reference, or the first latent frame when absent, anchors later windows; overlap latents are reused. 15001 frames is about 10 minutes. Supply enough driving audio. Built-in DMD steps 1-4 then OPSD steps 5-8; do not add the 1.5 adapter."
        elif architecture == "longcat_avatar_v1_5":
            compact += " Avatar 1.5: use its eight-step Distill preset."
    title = f"LongCat Avatar {version or ('1.5' if architecture == 'longcat_avatar_v1_5' else '1.0')}" if avatar else "LongCat Video"
    result = help_text(title, info, VIDEO_PROMPT, compact, VIDEO_PROMPT_SHORT + (" The driving recording determines dialogue; do not substitute new words in the prompt." if avatar else ""))
    result["infos"] += "\n\n### Speed and Memory\n\n**KV Cache** can speed up generation when using an image or continuing a video. **Streamed from RAM** options use less graphics memory. **Q4** saves more memory than **INT8**, but may change fine details. Leave the cache disabled if you prefer the standard behavior.\n\n[Longcat repo](https://github.com/meituan-longcat/LongCat-Video)"
    result["deepy_infos"] += " custom_settings.longcat_kv_cache: Disabled (default), INT8 or Q4 (full VRAM), INT8_RAM or Q4_RAM (MMGP streaming). Q4 trades cache precision for about half the INT8 storage. Applies to start/reference/overlap frames; rebuilt on adapter changes and per window."
    result["specialities"] = specialities("lip sync") if avatar else []
    return result
