# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import os
from pathlib import Path

from shared.utils import files_locator as fl
from .composition import SOURCE_LABELS, migrate_source, read_abc, validate_source
from .prompt_enhancers import LYRICS_SYSTEM_PROMPT, STYLE_SYSTEM_PROMPT, INSTRUMENTAL_PLAN_SYSTEM_PROMPT, INSTRUMENTAL_STYLE_SYSTEM_PROMPT


ARCHITECTURE = "yue2"
HUM_ARCHITECTURE = "yue2_hum"
REPO_ID = "DeepBeepMeep/TTS"
TEXT_ENCODER_FOLDER = "YuE2_AR"
ASSETS = ["vae_config.json", "YuE2_VAE_bf16.safetensors"]
SCORING_CHECKPOINT = "SheetSage2_MERT2_bf16.safetensors"
HUM_ENCODER = "YuE2_VAE_Encoder.safetensors"
HUM_INFOS = """**Turn a hummed melody into a new song.** Upload a clear human hum (10–30 seconds is a useful starting point), enter the words to sing in **Lyrics**, and describe the genre, instruments and voice in **Music Style**.

**Continue Hum** uses your melody as the opening of a longer composition. **Hum Only** keeps the transcribed score without extending it; this does not guarantee an exact waveform duration. Both modes also use the hum's pitch contour and timing during audio synthesis. Transcription errors can change notes. This creates a new performance; it does not clone your voice or preserve the input recording.

Start with 32 synthesis steps and guidance 1. Guidance controls the lyrics/style conditioning, not a separate hum strength. Maximum Song Duration caps the result; it does not force that length. The song can end sooner, and a short cap can cut it off. Change the seed for another interpretation. Use an isolated hum rather than a full mixed song. No manual ABC score or direct-generation mode is offered for this finetune.

**Save ABC and MIDI Score** exports the composition used for generation. For Continue Hum, this includes the generated continuation. Abort cancels; Early Stop renders the audio tokens already composed. The included acoustic LoRA is required and already incorporates Mothersuperior's real-audio v4 LoRA; do not add that LoRA again. Prompt enhancement is optional and off by default; it edits lyrics/style only. **CC BY-NC 4.0: non-commercial use.**
"""
HUM_PROMPT_INFOS = """Write the actual words to sing, with short lines and section labels such as `[Verse]` and `[Chorus]`. Repeat chorus words explicitly. For example:
```text
[Verse]
Morning light across the bay
We watch the shadows drift away

[Chorus]
Stay with me until the dawn
Let our little song go on
```
Music Style example: `English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM`.
Your recording supplies the melody and phrasing, not a transcript of lyrics. Match the opening lyric phrases to the hum; Continue Hum can compose additional sections. Hum Only works best with lyrics sized for the hummed phrase. Avoid competing style instructions. The optional Lyrics and Music Style enhancers do not process or alter the hum.
"""
HUM_DEEPY_INFOS = """YuE2 Hum-to-Song: required `audio_guide` = clear human hum; `audio_prompt_type=\"A\"`. `prompt` = actual lyrics; `alt_prompt` = music style. `model_mode=0` continues the hummed melody into a longer score; `1` uses only its transcribed score. No direct generation or manual ABC. Outputs a new 48 kHz stereo song, not voice cloning or waveform-preserving editing. Start with 10–30 s of isolated humming, 32 steps and guidance 1 (lyrics/style CFG). Duration is an upper limit, not a target. `custom_settings.save_score=1` exports the final ABC/MIDI composition. Built-in hum LoRA already includes real-audio v4; do not stack it again. Abort cancels; Early Stop renders composed tokens. Enhancer off by default, edits lyrics/style only. CC BY-NC 4.0.
"""
PROMPT = """[Verse]
I left my keys beside your coffee
Caught the first bus out of town
Watched the harbor through the window
Till the morning mist came down

[Chorus]
Leave a light on by the water
Let it shine across the blue
Every road can take me farther
Every road leads back to you

[Verse]
City rain against my collar
Your old song inside my head
I could hear you in the silence
Of the words we never said

[Chorus]
Leave a light on by the water
Let it shine across the blue
Every road can take me farther
Every road leads back to you

[Bridge]
Now the last train crosses over
And the rooftops come in view
I have found the words I needed
I am bringing them to you

[Chorus]
Leave a light on by the water
Let it shine across the blue
Every road can take me farther
Every road leads back to you

[Outro]
There's a light on by the water
And I'm coming home to you"""
STYLE = "English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM"

INFOS = """**Turn your lyrics into a complete song** with a singing voice and accompaniment, in stereo.

- **Lyrics** supplies the words to sing. **Music Style** describes how the song should sound. Use both for your first song.
- **Melody and chords** is the recommended starting point: YuE2 plans the tune and its harmony before making the recording.
- **Melody only** plans the tune while giving the accompaniment more freedom. Use this when a supplied melody should keep its shape but the arrangement can change.
- **Direct generation** skips the written composition plan. Try it for an alternative interpretation of the same lyrics and style.

**Start with the defaults.** Write one or two short verses and a repeated chorus, choose a clear style, and allow enough time for the words. **Maximum Song Duration is an upper limit, not a requested song length. YuE2 can stop earlier when it considers the song finished.** Increasing the limit allows more time but does not force a longer song; a limit that is too short can cut it off. A very large limit is capped by the model's capacity, which also depends on the length of your lyrics and score. Change the seed to try another performance. More synthesis steps cost more time; 32 is the recommended starting point.

**Abort** cancels the generation without saving a partial song. **Early Stop** stops audio-token generation and renders what has already been composed.

**Optional prompt enhancer:** disabled by default. Choose Lyrics to turn an idea into singable words or tidy existing lyrics; choose Music Style to clarify the sound, or Lyrics then Music Style to prepare both. Review the result before generating. The enhancer does not edit your ABC score.

**Composition Source** chooses one input independently of Composition Planning:

- **New Composition** plans from your lyrics and style; retained uploads are ignored.
- **Use ABC Score** renders the uploaded UTF-8 `.abc` notation as supplied.
- **Continue ABC Score** keeps the supplied notation and asks the planner to continue it before rendering.
- **Use Score from Audio** transcribes a source recording with SheetSage2/MERT2 and renders that composition.
- **Continue Score from Audio** transcribes the recording, continues its notation, then renders the complete composition.

Only the selected source is used. ABC modes require an ABC file; audio modes require source audio. A hidden upload is ignored. Use Melody and chords, Melody only, or Instrumental planning with supplied sources; Direct generation supports New Composition only. Compatible ABC uses Vocal and Ins voices. Melody-only scores must omit chord symbols. Keep notation out of the lyrics and style fields.

**Save ABC and MIDI Score:** turn this on to save both `.abc` and `.mid` beside the song, using the same filename. This exports the complete composition used for synthesis, including any continuation, not a transcription of the finished performance. The score can be longer than an early-stopped song. Direct generation has no score to export. Off by default.

Changing lyrics, style or score creates a **new recording**. Audio conditioning does not preserve the original singer or backing track. Supply lyrics separately: transcription extracts notes, not words. Match section order and phrasing to your source. Transcription and continuation can make musical mistakes; review the saved score. Maximum duration and context limits still apply, even when extending a composition. The model weights are **CC BY-NC 4.0 (non-commercial use)**.


### Instrumental Music

Select **Instrumental - Melody and Chords**, put `[Instrumental]` in **Lyrics**, and describe the instruments and mood in **Music Style**, for example `instrumental, ambient, piano, soft strings, reflective`. For a chosen structure, use section tags such as `[Intro]`, `[Verse 1]` and `[Chorus]` as shown in Prompt Help. Leave source audio and the ABC file empty for automatic composition.

The instrumental adapter downloads on first use and loads automatically at strength 1; no manual LoRA selection is needed. It plans melody and chords before composing audio. Section order and optional times guide the result but do not guarantee exact transitions. Use a short duration cap for previews or a larger cap to allow a longer piece and ending.

### Optional Prompt Enhancement

Enhancement is **off by default**. Choose what you want to edit:

| Songs with vocals | Instrumentals | Updates |
| --- | --- | --- |
| Lyrics | Instrumental Section Plan | Lyrics field only |
| Music Style | Instrumental Music Style | Music Style only, using the other field as context |
| Lyrics then Music Style | Instrumental Plan then Music Style | Both fields, in that order |

For finished lyrics or a finished section plan, choose style only. To start from an idea, choose lyrics/plan or both, click **Enhance**, and review the result before generation. Select an instrumental enhancer explicitly: Composition Planning does not switch it automatically. Enhancers prepare text inputs; they do not write or edit ABC scores.

### LoRAs and License

Compatible YuE2 **AR** and **acoustic/diffusion** LoRAs can be selected together, each with its own multiplier. AR LoRAs affect composition and conditioning and require a constant multiplier. Acoustic LoRAs affect rendering and can use step schedules. Start at 1 and lower it for a weaker effect. To change the built-in instrumental adapter's strength, select the same adapter manually; your multiplier replaces the automatic copy.

The model weights and built-in instrumental adapter use **CC BY-NC 4.0 (non-commercial use)**.
"""
PROMPT_INFOS = """### Lyrics for Classic YuE2

Classic modes (**Melody and chords**, **Melody only**, **Direct generation**) accept section labels as ordinary text, not a fixed list of control tokens. Use these standard song sections:

| Label | Purpose |
| --- | --- |
| `[Intro]` | Opening section |
| `[Verse]` | Main lyric verses |
| `[Pre-Chorus]` | Build-up before a chorus |
| `[Chorus]` | Repeated hook or refrain |
| `[Post-Chorus]` | Follow-up after a chorus |
| `[Bridge]` | Contrasting section |
| `[Interlude]` | Instrumental passage between sung sections |
| `[Solo]` | Featured instrumental passage |
| `[Outro]` | Closing section |

Numbered labels such as `[Verse 1]` and `[Verse 2]`, and descriptive variants such as `[Final Chorus]` or `[Saxophone Solo]`, can also express structure. These are musical cues, not guaranteed controls; there is no exhaustive supported-label whitelist in classic mode. Leave instrumental passages without lyric lines and keep detailed sound instructions in Music Style. Instrumental mode uses the same capitalization and numbering conventions, with the restricted set of section names listed below. YuE2 handles its internal formatting automatically.

Write the actual words to sing, not a request to write a song. Put labels on their own lines, use short singable lines, and leave a blank line between sections:

```text
[Verse]
Morning light across the bay
We watch the shadows drift away

[Chorus]
Stay with me until the dawn
Let our little song go on
```

Repeat the actual chorus words when you want it again. Keep instruments, production notes and explanations out of the lyrics. With **Lyrics** or **Lyrics then Music Style** enhancement, you can instead enter a songwriting idea, click **Enhance**, then review the words before generating.

### Music Style for Songs

Describe the language, genre, main instruments, vocal character and mood. Add tempo when it matters:

```text
English acoustic pop, warm female vocal, fingerpicked guitar,
gentle drums, hopeful, 90 BPM
```

Use a few compatible ideas rather than conflicting styles. **Music Style** enhancement uses the lyrics as context while leaving them unchanged. A voice description guides vocal character; it does not guarantee a specific singer's identity.

### Instrumental Prompts

Only for **Instrumental - Melody and Chords**, replace sung lyrics with `[Instrumental]` to let the model choose the structure, or use one section tag per line (the same title-case format as vocal songs):

```text
[Intro]
[Verse]
[Chorus]
[Bridge]
[Chorus]
[Outro]
```

Supported tags: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`. Numbered sections such as `[Verse 1]` and `[Chorus 2]` are accepted. Capitalization is flexible; YuE2 handles it and removes section numbers internally without changing section order or repetition. Optional times go inside the tags, for example `[Intro 0:00-0:15]` followed by `[Verse 1 0:15-0:45]` on the next line. They guide structure without guaranteeing exact timing. Use real line breaks, no sung words or bracketed production notes, and do not combine `[Instrumental]` with section tags. Invalid plans report the faulty line number and its original content.

Put the sound description in **Music Style**:

```text
instrumental, ambient, piano, soft strings, reflective, 80 BPM
```

Use **Instrumental Music Style** to enhance only that description. **Instrumental Section Plan** turns a structural brief such as “intro, verse, chorus twice, then outro” into tags; **Instrumental Plan then Music Style** prepares both fields. With no requested structure, plan enhancement returns `[Instrumental]`. Select the instrumental option explicitly, click **Enhance**, and review the plan before generating.

### When Using Source Audio or a Score

Choose the composition input before generating. For an ABC file, select **Use ABC Score** or **Continue ABC Score**, then upload the notation. For audio-based composition, provide the recording and enter lyrics separately, following its musical structure; transcription supplies musical notes. Instrumental mode takes a section plan instead of lyrics. Keep ABC notation out of the Lyrics and Music Style fields.
"""

DEEPY_INFOS = """**YuE2 song generation:** `prompt` = lyrics; `alt_prompt` = music style. Outputs 48 kHz stereo vocals and accompaniment.
- `model_mode`: **0** melody+chords (recommended), **1** melody only/free accompaniment, **2** direct generation.
- `audio_prompt_type`: `0` new composition; `1` use supplied ABC; `12` continue supplied ABC; `A` transcribe source audio and use its score; `A2` transcribe and continue the score. Modes 0/1/3 support all five; direct mode 2 requires `0`. Only the selected source is used.
- `custom_guide` is required for `1`/`12`: a UTF-8 `.abc` file with Vocal/Ins voices; mode 1 omits chord symbols. `audio_guide` is required for `A`/`A2`. Hidden inputs and old `custom_settings.abc` text are ignored. Continuation preserves the supplied score before adding model-generated notation.
- `custom_settings.save_score`: **0** off (default), **1** export both `.abc` and `.mid` with the song's filename. Exports the complete conditioning composition, including continuation, not the finished performance; it may outlast a truncated song. Mode 2 exports neither. API artifacts return these side files in memory.
- **`duration_seconds` is an upper limit, not a requested song length. YuE2 can stop earlier when it considers the song finished.** Increasing it does not force a longer song; too short can cut it off. Very large limits are capped by the remaining model context after lyrics/score. Start with 32 steps and guidance 1; change seed for another take.
- Abort cancels without audio. Early Stop renders existing audio tokens; during score planning it finishes the score then makes an approximately eight-second preview, capped by duration. Acoustic synthesis/decoding still finish; previews may end mid-phrase.
- Prompt enhancement is off by default: `T1` prepares lyrics, `B2O` prepares style, `T1,B2O` prepares lyrics then style informed by them. It does not modify ABC.
- Audio sources use SheetSage2/MERT2 in modes 0/1/3. Supply lyrics separately; notes are transcribed, not words. Regeneration creates a new recording; no waveform-preserving edits. **CC BY-NC 4.0 weights: non-commercial use.**


### Instrumentals
Mode 3 uses `[Instrumental]` or a section-only plan such as `[Intro]`, `[Verse 1]`, `[Chorus]` in `prompt`, with instrumental `alt_prompt`. Use title-case names as for classic songs; YuE2 normalizes case and removes section numbers internally while preserving times and repetition. Only the six section names listed in prompt help are allowed; instruments belong in `alt_prompt`. Its built-in AR LoRA downloads just in time at strength 1; no manual selection needed. Selecting the same adapter manually uses your multiplier instead. Times/order guide rather than guarantee transitions. Use short caps for previews or larger caps to allow longer pieces/endings.

### LoRAs and License
`activated_loras`/`loras_multipliers` accept compatible YuE2 AR and acoustic/diffusion adapters together with independent strengths. Keys route each component; AR affects composition/conditioning and requires constant strength, acoustic affects rendering and supports step schedules. Start at 1 and lower for a weaker effect. Native/ComfyUI fused adapters are accepted; full replacement weights require base-relative diff tensors. Model and instrumental adapter: CC BY-NC 4.0, non-commercial use.
"""
DEEPY_PROMPT_INFOS = """Prepare generation-ready `prompt` and `alt_prompt` directly from the user's request using the rules below.

### Classic Lyrics (Modes 0/1/2)
Use actual short, singable lines, one bracketed section label per line, and blank lines between sections. Standard sections: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`, `[Interlude]`, `[Solo]`, `[Outro]`. Numbered `[Verse 1]`/`[Verse 2]` and descriptive `[Final Chorus]`/`[Saxophone Solo]` are also text cues. Classic mode has no exhaustive label whitelist; labels do not guarantee behavior. Leave instrumental passages without sung lines. Repeat chorus words explicitly; keep production notes out of lyrics. When the user supplies a songwriting brief, write the finished lyrics yourself before submitting `prompt`; never submit the brief as sung text. Preserve supplied lyrics, language, meaning and section order unless the user asks for revisions. Return only lyrics and section labels in this field, without a title, explanations, Markdown fences or JSON.

### Vocal Music Style
`alt_prompt`: language + genre + instruments + vocal character + mood, optionally tempo. Example: `English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM`. Write a concise style description yourself, using the lyrics as context. Preserve the user's instruments, language, mood, exclusions and supplied tempo; add only compatible details and do not invent an exact BPM, key or meter. Avoid conflicting styles and keep lyrics, section tags and ABC notation out of this field. Vocal character does not guarantee singer identity. Change style for a new arrangement, seed for a new take; increase duration or shorten lyrics if cut off.

### Instrumental Mode 3
`prompt`: `[Instrumental]` alone or `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`, one per real newline. Use the same title-case format as classic songs. Numbers and times are accepted, for example `[Verse 1 0:15-0:45]`. Do not mix `[Instrumental]` with tags, add sung words, or put production notes in brackets. Unlike classic mode, only these six section names are allowed: do not emit instrument tags such as `[Guitar]` or `[Piano]`, or classic-only variants such as `[Solo]` or `[Final Chorus]`. Put instruments in `alt_prompt`. YuE2 handles capitalization and removes section numbers internally, preserving order, repeated sections and times. Times guide rather than guarantee transitions or duration. Example style: `instrumental, ambient, piano, soft strings, reflective, 80 BPM`. Convert a structural brief into valid tags yourself, preserving requested order, repetitions and explicit times. If no structure is requested, use `[Instrumental]`; do not invent timestamps or an arrangement. Preserve an existing valid plan. Write `alt_prompt` beginning with instrumental, describing genre, instruments, mood and arrangement; preserve explicit constraints and never introduce singing, speech, humming, choir or backing vocals. The AR LoRA loads automatically. If validation fails, use the reported line number and original content to correct that line; do not remove valid times or rewrite unrelated sections.

### Source Audio and Scores
Choose audio_prompt_type=1 to render custom_guide ABC, or 12 to continue it. A uses audio_guide transcription; A2 continues that transcription. Inactive uploads are ignored. Supply aligned lyrics or an instrumental section plan separately. save_score=1 exports the conditioning score as ABC and MIDI.
"""


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE, HUM_ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "tts"

    @staticmethod
    def query_family_infos():
        return {"music": (2195, "Music"), "tts": (2200, "TTS")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return ARCHITECTURE

    @staticmethod
    def query_model_def(base_model_type, model_def):
        definition = {
            "group": "music", "audio_only": True, "image_outputs": False, "sliding_window": False, "supports_early_stop": True, "device_explicit": True,
            "enabled_audio_lora": True,
            "guidance_max_phases": 1, "no_negative_prompt": True, "inference_steps": True,
            "temperature": True, "top_k_slider": True, "top_p_slider": True, "embedded_guidance": False,
            "image_prompt_types_allowed": "", "profiles_dir": [ARCHITECTURE], "compile": False,
            "lm_decoder": True, "prompt_class": "Lyrics", "prompt_enhancer_button_label": "Enhance",
            "text_encoder_URLs": [f"https://huggingface.co/{REPO_ID}/resolve/main/{TEXT_ENCODER_FOLDER}/YuE2_AR_{precision}.safetensors" for precision in ("bf16", "int8_convrot")],
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "yue2_lora_instrumental": f"https://huggingface.co/{REPO_ID}/resolve/main/yue2/ar_lora_inst_v3abc.bf16.safetensors",
            "prompt_enhancer_def": {
                "selection": ["T1", "B2O", "T1,B2O", "B3", "B4O", "B3,B4O"],
                "labels": {"T1": "Lyrics", "B2O": "Music Style", "T1,B2O": "Lyrics then Music Style", "B3": "Instrumental Section Plan", "B4O": "Instrumental Music Style", "B3,B4O": "Instrumental Plan then Music Style"},
                "default": "",
            },
            "text_prompt_enhancer_instructions1": LYRICS_SYSTEM_PROMPT, "text_prompt_enhancer_max_tokens1": 1536,
            "text_prompt_enhancer_instructions2": STYLE_SYSTEM_PROMPT, "text_prompt_enhancer_max_tokens2": 384,
            "text_prompt_enhancer_instructions3": INSTRUMENTAL_PLAN_SYSTEM_PROMPT, "text_prompt_enhancer_max_tokens3": 512,
            "text_prompt_enhancer_instructions4": INSTRUMENTAL_STYLE_SYSTEM_PROMPT, "text_prompt_enhancer_max_tokens4": 384,
            "alt_prompt_inherits_prompt_paragraphs": True,
            "alt_prompt": {"label": "Music Style", "placeholder": "Language, genre, instruments, mood, vocal character and tempo", "lines": 3},
            "model_modes": {"choices": [("Melody and chords", 0), ("Melody only", 1), ("Direct generation", 2), ("Instrumental - Melody and Chords", 3)], "default": 0, "label": "Composition Planning"},
            "any_audio_prompt": True, "audio_prompt_choices": True, "audio_guide_label": "Source Song (Music to Transcribe)",
            "audio_prompt_type_sources": {"selection": list(SOURCE_LABELS), "labels": SOURCE_LABELS, "default": "0", "label": "Composition Source", "letters_filter": "012A", "custom_flags": {"0": "New Composition", "1": "ABC Score", "2": "Score Continuation"}},
            "custom_guide": {"id": "custom_guide", "name": "ABC Score", "label": "ABC Score (.abc)", "type": "file", "default": None, "required": True, "file_types": [".abc"], "audio_prompt_type": "1"},
            "custom_settings": [
                {"id": "save_score", "name": "Save Score", "label": "Save ABC and MIDI Score", "type": "dropdown", "choices": [("Off", 0), ("On", 1)], "default": 0},
            ],
            "duration_slider": {"label": "Maximum Song Duration (seconds)", "name": "Maximum Song Duration", "min": 1, "max": 600, "increment": 1, "default": 120},
            "infos": INFOS,
            "prompt_infos": PROMPT_INFOS,
            "deepy_infos": DEEPY_INFOS,
            "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
        }
        if base_model_type == HUM_ARCHITECTURE:
            definition["prompt_enhancer_def"]["selection"] = ["T1", "B2O", "T1,B2O"]
            definition["prompt_enhancer_def"]["labels"] = {key: value for key, value in definition["prompt_enhancer_def"]["labels"].items() if key in definition["prompt_enhancer_def"]["selection"]}
            definition.update({
                "parent_model_type": ARCHITECTURE,
                "model_modes": {"choices": [("Continue Hum", 0), ("Hum Only", 1)], "default": 0, "label": "Hum Melody"},
                "audio_guide_label": "Hummed Melody",
                "audio_prompt_type_sources": {"selection": ["A"], "labels": {"A": "Hummed Melody to Song"}, "default": "A", "label": "Source Audio", "letters_filter": "A"},
                "infos": HUM_INFOS, "prompt_infos": HUM_PROMPT_INFOS,
                "deepy_infos": HUM_DEEPY_INFOS, "deepy_prompt_infos": HUM_PROMPT_INFOS,
                "specialities": [{"name": "hum to song", "aliases": ["humming to song"], "description": "Create a song from a hummed melody, lyrics and music style."}],
            })
            definition.pop("custom_guide")
        from models.help_shared import merge_help, specialities
        return merge_help(definition, {"specialities": specialities("music generation")})

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        files = [{"repoId": REPO_ID, "sourceFolderList": ["yue2", TEXT_ENCODER_FOLDER, "sheetsage2"], "fileList": [ASSETS, ["qwen.tiktoken"], [SCORING_CHECKPOINT]]}]
        if base_model_type == HUM_ARCHITECTURE:
            files.append({"repoId": REPO_ID, "sourceFolderList": [""], "fileList": [[HUM_ENCODER]]})
        return files

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, dtype=None, VAE_dtype=None, save_quantized=False, profile=0, lm_features=None, text_encoder_filename=None, **kwargs):
        from .pipeline import YuE2Pipeline
        paths = {name: fl.locate_file(os.path.join("yue2", name)) for name in ASSETS}
        if base_model_type == HUM_ARCHITECTURE:
            acoustic_weights, hum_weights = model_filename
            hum_encoder_weights = fl.locate_file(HUM_ENCODER)
        else:
            acoustic_weights, = model_filename
            hum_weights = hum_encoder_weights = None
        tokenizer_path = fl.locate_file(os.path.join(TEXT_ENCODER_FOLDER, "qwen.tiktoken"))
        pipeline = YuE2Pipeline(text_encoder_filename, acoustic_weights, tokenizer_path, paths["YuE2_VAE_bf16.safetensors"], paths["vae_config.json"], dtype, VAE_dtype, lm_features, scoring_checkpoint=fl.locate_file(os.path.join("sheetsage2", SCORING_CHECKPOINT)), hum_weights=hum_weights, hum_encoder_weights=hum_encoder_weights)
        if save_quantized:
            from wgp import save_quantized_model
            directory = Path(__file__).parent
            save_quantized_model(pipeline.transformer, model_type, acoustic_weights, dtype, str(directory / "yue2.json"), submodel_no=1)
        pipe = {"text_encoder": pipeline.text_encoder, "transformer": pipeline.transformer, "vae": pipeline.vae}
        if pipeline.hum is not None:
            pipe.update(hum_projections=pipeline.hum, hum_encoder=pipeline.hum_encoder)
        return pipeline, {"pipe": pipe, "loras": ["text_encoder"], "cudaGraphs": ["text_encoder"] if lm_features.graphs else False}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"prompt": PROMPT, "alt_prompt": STYLE, "audio_prompt_type": "0", "duration_seconds": 120, "video_length": 0, "num_inference_steps": 32, "guidance_scale": 1.0, "temperature": 1.0, "top_k": 100, "top_p": 0.95, "model_mode": 0, "custom_guide": None, "custom_settings": {"save_score": 0}, "prompt_enhancer": "", "negative_prompt": "", "repeat_generation": 1, "multi_prompts_gen_type": "FG"})
        if base_model_type == HUM_ARCHITECTURE:
            ui_defaults["audio_prompt_type"] = "A"

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if base_model_type != HUM_ARCHITECTURE:
            migrate_source(ui_defaults, settings_version)
        # Retired style-only variants now share the contextual style choice.
        mode = ui_defaults.get("prompt_enhancer", "")
        if isinstance(mode, str):
            aliases = {"L2O": "B2O", "L4O": "B4O"}
            ui_defaults["prompt_enhancer"] = ",".join(
                aliases.get(step.replace("K", ""), step.replace("K", "")) + ("K" if "K" in step else "")
                for step in mode.split(",")
            )

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if not one_prompt.strip() or not inputs["alt_prompt"].strip():
            return "YuE2 requires lyrics and a music style."
        if base_model_type == HUM_ARCHITECTURE:
            if inputs["audio_prompt_type"].replace("N", "").replace("V", "") != "A" or inputs["audio_guide"] is None:
                return "Upload a hummed melody for Hum-to-Song."
            if inputs["model_mode"] not in (0, 1):
                return "Choose Continue Hum or Hum Only."
        else:
            try:
                source = migrate_source(inputs)
                validate_source(source, inputs["model_mode"], inputs.get("custom_guide"), inputs.get("audio_guide"))
                if source in ("1", "12"):
                    read_abc(inputs["custom_guide"], inputs["model_mode"] == 1)
            except ValueError as exc:
                return str(exc)
        if inputs["guidance_scale"] < 1:
            return "YuE2 guidance must be at least 1 (1 disables CFG)."
        if inputs["model_mode"] not in (0, 1, 2, 3):
            return "Choose melody and chords, melody only, direct generation, or instrumental."
        if inputs["model_mode"] == 3:
            from .instrumental import validate_instrumental_prompt
            error = validate_instrumental_prompt(one_prompt)
            if error:
                return error
        return None
