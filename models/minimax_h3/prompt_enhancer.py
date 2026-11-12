"""Prompt guidance and enhancer system prompts for MiniMax H3."""


SLIDING_WINDOW_PROMPT_INFOS = """### Controlling sliding-window length and hard cuts

To give each sliding window its own H3 prompt, set **How to Process each Line of the Text Prompt** to **Each Paragraph Separated by an Empty line will be used for a new Sliding Window of the same Video Generation**. Keep all the H3 fields for one window together without empty lines, then insert one empty line before the next window's prompt.

Begin a window's paragraph with any of these WanGP commands:

- `[/duration=124]`: make the window contribute 124 final frames.
- `[/duration=5s]`: make it contribute about 5 seconds at the selected frame rate.
- `[/duration=20%]`: give it 20% of the requested total video length.
- `[/overlap=18]`: carry 18 frames from the previous H3 window for a smoother transition. H3 overlap values follow the `17k + 1` pattern and WanGP rounds other values to a valid one.
- `[/new_shot]`: carry no frames from the previous window, creating a hard cut. This is the same as `[/overlap=0]`.

Commands can be combined, for example `[/duration=5s,/overlap=18]` for a connected five-second window or `[/duration=5s,/new_shot]` for a new five-second shot after a hard cut. The duration is the part kept in the final video; overlap frames are additional continuity frames and are removed when WanGP joins the windows. WanGP removes these commands before sending the prompt to H3.

`[Shot 2] At MM:SS.mmm, ...` describes a model-directed cut **inside one H3 window**. Use `[/new_shot]` at the beginning of a later window when the boundary between two generated windows itself must be a hard cut.

"""


FL2VA_DEEPY_PROMPT_INFOS = """Describe the resulting audiovisual scene chronologically in English: subjects, actions, camera, light and sound. Preserve exact requested speech/lettering in its original language. With control video, describe the desired result and retained context; with start/end images, describe the motion connecting them.

Use these three fields together, with no blank lines inside a window:
integrated_multimodal_description: [Shot 1] The courier stops, turns to camera and says (S1) <d>[English] Made it.</d>
overall_soundscape: Rain on the awning and wet footsteps.
non_diegetic_music: N/A

For supplied frame anchors, prepend: `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.` Start-only uses Picture 1 at zero; end-only uses Picture 1 at the exact end and final shot; both use Picture 1 at zero and Picture 2 at the end. Keep identity, clothes, layout and lighting consistent.

Shot 1 has no timestamp. Later cuts use `[Shot N] At MM:SS.mmm, ...` with increasing times. Keep speaker IDs stable and exact speech/lyrics inside `<d>[Language] ...</d>`. A line crossing a cut uses `<scenetrans>` at both connecting points; `<cutoff>` marks speech interrupted by the ending. Quote visible lettering separately. Ambience goes in `overall_soundscape`; audience-only score in `non_diegetic_music` (`N/A` for none).

In soundtrack mode (`AS` or `KS`), for visible speech explicitly include `speaking with natural lip movements synchronized` or equivalent wording requesting lip sync. Identify the speaking character; no transcript is required. Describe the action directly, without referring to a supplied audio track or assigning it an `<Audio N>` label. This encourages speaking motion, especially in the first window, but does not guarantee exact lip sync. Outside soundtrack mode, requested speech must still include the exact spoken words in `<d>[Language] ...</d>`.

For a single long video, use paragraph-per-window processing (`PW`): blank lines separate windows, all fields of one window stay together. Each window restarts at Shot 1 and time zero. Overlap is Picture 1; its end anchor is Picture 2. Without a start/overlap, an end-only anchor is Picture 1. A window's exact end time is its `generated_seconds` in the `window_plan` of `wangp_validate` (overlap included, frame-aligned), not the requested duration. Optional navigation titles must start with `#`. Consult the long-video guide for window scheduling and slash commands.
"""

FL2VA_PROMPT_INFOS = f"""## H3 FL2VA prompt structure

FL2VA uses the same three-part audiovisual prompt for text-only, first-frame, last-frame, and first-and-last-frame generation:

```text
integrated_multimodal_description: [Shot 1] ... [Shot 2] At 00:03.500, ...
overall_soundscape: ...
non_diegetic_music: ...
```

When an image fixes a point on the output timeline, put its alignment instruction before these fields:

- **Start frame:** (First Frame) `<Picture 1>` belongs to `[Shot 1]` at `0.00` seconds.
- **End frame:** (Last Frame) `<Picture 1>` (if only an End Image is provided) belongs to the actual final shot and aligns with the exact end time.
- **Start + End:** `<Picture 1>` anchors `0.00` seconds and `<Picture 2>` anchors the exact end time. A single continuous shot is usually preferable unless the requested action genuinely needs cuts.
Please note that when using sliding windows, the picture count is reset to 1 in each new window. So the last overlap frame (first frame of the new window) will always correspond to `<Picture 1>` and any new End Frame provided will correspond to `<Picture 2>` (or `<Picture 1>` if new shot is requested or there is no overlap frame for this window). Injected frames that fall in the window come next, in time order; declare each one with its own alignment line at its time in the window. An injected frame on the window's first or last frame counts as its start or end image, and one inside the overlap is not used. The exact end time of a window is its full length, overlap frames included and aligned to the frame grid, so it can exceed the duration requested for the window.

### Connecting shots

- `[Shot 1]` has no timestamp. Start each later shot with a strictly increasing cut time: `[Shot N] At MM:SS.mmm, ...`.
- A cut should reveal a meaningful change in viewpoint, framing, place, time, subject, or state. Use continuous camera movement instead of a cut for a small change of distance or angle.
- Keep identities, wardrobe, props, spatial relationships, lighting, and action causality consistent across cuts.
- Keep speaker IDs such as `(S1)` stable throughout. Put only exact dialogue or lyrics inside `<d>[Language] ...</d>`.
- If speech crosses a cut, mark the connection with `<scenetrans>` on both sides and say that the audio continues across the cut. Use `<cutoff>` only when the video ends before a spoken line finishes.

`overall_soundscape` summarizes ambience, physical sounds, and non-verbal human sounds without repeating dialogue. `non_diegetic_music` describes only music the audience hears but the characters do not; write `N/A` when no such score is wanted.

Multiple shots can be defined inside the same sliding window but each new sliding window resets the timeline and is expected to start with a new Shot 1 at time zero. 

### Speaking in soundtrack mode

When using **Soundtrack Kept** for visible speech, identify the speaking character and explicitly request synchronized lip movements. Include `speaking with natural lip movements synchronized` or equivalent wording in the prompt, especially for the first window. For example: `A woman is sitting on a boat and complaining about life. She is visibly speaking with natural lip movements synchronized.` In a structured prompt, put this action in `integrated_multimodal_description`.

You do not need to write the spoken words. Describe the speaking action directly, without mentioning a supplied audio track or giving it an `<Audio N>` label: soundtrack mode does not provide it as a named audio reference. This encourages visible speaking, but does not guarantee exact lip sync.

Outside **Soundtrack Kept** mode, when requesting speech, still include the exact spoken words in `<d>[Language] ...</d>`, for example `The woman says <d>[English] I was afraid to speak.</d>`.

{SLIDING_WINDOW_PROMPT_INFOS}
### Prompt examples

#### Text-only, single shot

```text
integrated_multimodal_description: [Shot 1] At blue hour, a tired bicycle courier in a yellow raincoat pedals through a narrow rain-soaked market street. The camera tracks beside her at wheel height, then rises smoothly into a medium close-up as she stops beneath a flickering awning. Water runs from her helmet while she catches her breath, looks toward the closed train station, and says (S1) <d>[English] I missed it again.</d> Neon reflections ripple across the pavement and the shot holds on her rueful smile.
overall_soundscape: Steady rain on canvas and metal, bicycle-chain clicks, wet tires hissing over stone, distant traffic, and the courier's breath settling after the stop.
non_diegetic_music: N/A
```

#### First-frame animation

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
integrated_multimodal_description: [Shot 1] Continue directly from <Picture 1>, preserving its subject, clothes, composition, warm window light, and studio layout. The ceramic artist lowers her brush onto the glazed bowl and paints one continuous cobalt line while the camera makes a slow clockwise arc from medium shot to close-up. She turns the bowl toward the light, inspects the finished pattern, and gives a small satisfied nod.
overall_soundscape: Soft brush strokes on ceramic, the wooden wheel turning, quiet room tone, and a faint breeze at the open window.
non_diegetic_music: A sparse, gentle marimba motif at a slow tempo, fading under the final close-up.
```

#### Timed multi-shot sequence

```text
integrated_multimodal_description: [Shot 1] A red fox runs across a snowy ridge at sunrise as a long-lens camera pans with it, powder spraying from every stride. [Shot 2] At 00:04.000, cut to a wide aerial view as the fox descends into a pine valley, its trail drawing a curved line through untouched snow. [Shot 3] At 00:07.500, cut to ground level beside a frozen stream; the fox slows, listens, and looks directly past the camera while drifting snow catches the orange backlight.
overall_soundscape: Fast paw impacts in powder, cold wind across the ridge, distant crows, snow falling from pine branches, and the fox's quiet breathing near the stream.
non_diegetic_music: Low sustained cellos with a restrained frame-drum pulse, opening into a single bright horn note in the final shot.
```

Adapted from MiniMax's [official base prompt-writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).
"""


REF2VA_DEEPY_PROMPT_INFOS = """Describe the resulting audiovisual scene and how each reference contributes; use English descriptions, retaining the original language of speech/lyrics and visible lettering. Use six sections in order:
subject_definitions: Bind recurring content to references, e.g. <Subject 1> is the violinist from <Picture 1>.
summary: Start with the applicable task tags, e.g. [reference generation + audio reference], then state the target scene.
retention_analysis: State what transfers, where it appears and what changes. Visual modes: fully_preserved, partially_preserved, attribute_transfer, weak_reference. Audio modes: fully_copy, partially_copy, reference, weak_reference.
detailed_description: Describe the actual scene, action, camera, light and sound in playback order, starting [Shot 1].
overall_soundscape: Ambience, physical sounds and voices.
non_diegetic_music: Audience-only score, or N/A.

With a kept soundtrack (`AS` or `KS`), for visible speech explicitly include `speaking with natural lip movements synchronized` or equivalent wording requesting lip sync in `detailed_description`. Identify the speaking character; no transcript is required. Describe the action directly, without referring to a supplied audio track or assigning it an <Audio N> label. This encourages speaking motion, especially in the first window, but does not guarantee exact lip sync. Outside soundtrack mode, requested speech must still include the exact spoken words in `<d>[Language] ...</d>`, including when using audio references. Keep labels stable: <Subject N> = reusable person/object/setting/style; <Picture N> = concrete image; <Video N> = video role; <Audio N> = sound or voice reference. Number each asset type independently. Start/end images precede general image references; account for them when assigning Picture numbers. Define a reference's role explicitly: identity, motion, framing, voice, copied audio, or a timed keyframe. State its use at the relevant point in the timeline.

Shot 1 has no timestamp; later cuts use [Shot N] At MM:SS.mmm with increasing times. Use stable speaker IDs (S1), with exact speech in <d>[Language] ...</d>. Speech across a cut uses <scenetrans> at both connecting points; <cutoff> marks an interrupted ending. Quote visible lettering. Preserve identity, props, geography and cause/effect.

For sliding windows (`PW`), keep all six sections together with no internal blank lines. Blank lines separate windows; restart each window at Shot 1 and time zero, and remap Picture numbers to that window's anchors/references in this order: start or continued frame (`<Picture 1>`), end image, the window's injected frames by time, then reference images. Navigation titles start with #. Read the long-video guide for scheduling.
"""

H3_AUDIO_DEEPY_PROMPT_INFOS = """For speech, write the exact words in `prompt`, one turn per `Speaker N:` block. Put language, emotion, pace and acting directions in square brackets; these are not spoken. Example:
Speaker 1:
[English, warm and quiet] You found the right place.
Speaker 2:
[English, excited] I knew you would be here.

Use Speaker 1 alone for a monologue. Audio References 1 to 3 supply the voices of Speakers 1 to 3. Keep speaker numbering and intended voices consistent; a speaker without a sample reuses their first generated turn as a voice reference. WanGP compiles the H3 prompt and joins the turns automatically.

For non-script sound generation, describe the sound and its evolution in H3's six sections: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music. Assign <Audio N> references their role (voice, timbre, rhythm or copied material), then describe the desired audio chronologically. Exact speech uses <d>[Language] ...</d>; use N/A for an unneeded music score.
"""

REF2VA_PROMPT_INFOS = f"""## H3 Ref2VA prompt structure

Ref2VA uses six sections in this order:

```text
subject_definitions:
<Subject 1> is ... from <Picture 1>.

summary:
[reference generation] ...

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - ...

detailed_description:
The target video is ...
[Shot 1] ...
[Shot 2] At 00:03.500, ...

overall_soundscape: ...

non_diegetic_music: ...
```
Dont't keep any empty lines between prompt sections when using sliding windows, otherwise they will be interpreted as sliding windows separators.

### Reference labels

- `<Subject N>` identifies reusable visible content such as a person, animal, object, environment, costume, style, or motion. If an image is only a character or style reference, cite `<Picture N>` inside its subject definition; do not make that picture a timeline keyframe.
- `<Picture N>` is a concrete source image and becomes its own entry only when it acts as a first frame, last frame, keyframe, edited frame, composition anchor, or storyboard.
- `<Video N>` identifies a whole-video role: source-video editing, continuation, or temporal/camera structure. Visible content taken from it still receives `<Subject N>` labels.
- `<Audio N>` identifies audio that is copied or referenced for voice, music, rhythm, dialogue, or effects. Its numbering is independent of video numbering. With a kept soundtrack (**Soundtrack Kept** choices), that audio is not a reference: omit its `<Audio N>` label and describe the speech and sounds where they occur.

Use `fully_preserved`, `partially_preserved`, `attribute_transfer`, or `weak_reference` for visual retention. Use `fully_copy`, `partially_copy`, `reference`, or `weak_reference` for audio. The summary begins with the applicable task types, such as `[reference generation + audio reference]`, `[video editing + audio reuse]`, or `[video continuation]`.

### Connecting shots

`[Shot 1]` has no timestamp; later shots use strictly increasing cut times in the form `[Shot N] At MM:SS.mmm, ...`. Keep subjects, reference roles, speaker IDs, appearance, props, space, and causality consistent between shots. A dialogue line crossing a cut uses `<scenetrans>` at both connecting points and an explicit continuity phrase. Use `<cutoff>` only when speech is truncated by the end of the video. For reference-generation prompts, MiniMax recommends roughly 350-500 English words in `detailed_description`, with dialogue-heavy timelines sized to fit the actual speech instead.

Multiple shots can be defined inside the same sliding window but each new sliding window resets the timeline and is expected to start with a new Shot 1 at time zero. 

Describe reference use where it actually takes effect in the timeline. A reference video is not automatically an edit or continuation, and audio is not automatically copied merely because it is present. Put exact dialogue inside `<d>[Language] ...</d>`, ambience and physical sounds in `overall_soundscape`, and audience-only score in `non_diegetic_music`.

### Speaking in soundtrack mode

When using **Soundtrack Kept** (`AS` or `KS`) for visible speech, identify the speaking character and explicitly request synchronized lip movements in `detailed_description`. Include `speaking with natural lip movements synchronized` or equivalent wording, especially for the first window. For example: `<Subject 1> is visibly speaking with natural lip movements synchronized.`

You do not need to write the spoken words. Describe the speaking action directly, without mentioning a supplied audio track or giving it an `<Audio N>` label: soundtrack mode does not provide it as a named audio reference. This encourages visible speaking, but does not guarantee exact lip sync.

Outside **Soundtrack Kept** mode, when requesting speech, still include the exact spoken words in `<d>[Language] ...</d>`, including when using audio references. For example: `<Subject 1> says (S1) <d>[English] I was afraid to speak.</d>`.

{SLIDING_WINDOW_PROMPT_INFOS}
### Prompt examples

These compact examples assume the named reference assets have been selected. For a final reference-generation prompt, expand `detailed_description` with the concrete appearance, motion, camera, and continuity details visible in those assets.

#### Reuse a character from an image

```text
subject_definitions:
<Subject 1> is the violinist from <Picture 1>, preserving her identity, facial features, dark braided hair, burgundy concert dress, and antique violin.
summary:
[reference generation] Place <Subject 1> in a new dawn performance on a misty rooftop while preserving her recognizable appearance and instrument.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity, face, hair, burgundy dress, and antique violin are retained; the rooftop setting and performance are new.
detailed_description:
The target video is a cinematic single-shot rooftop performance at dawn with cool mist and soft amber rim light. [Shot 1] <Subject 1> stands near the roof edge, framed from the waist up against a waking skyline. She raises the referenced violin naturally to her shoulder and begins an energetic passage, her bowing, fingering, posture, dress movement, and breathing remaining physically coherent. The camera makes a slow semicircle around her, widening as the mist parts and ending with her lowering the bow after the final note.
overall_soundscape: Light rooftop wind, distant morning traffic, fabric movement, breathing, and close natural violin performance.
non_diegetic_music: N/A
```

#### Borrow motion and camera language from a video

```text
subject_definitions:
<Subject 1> is the silver rally car from <Picture 1>, preserving its body shape, paint, decals, and wheel design.
<Video 1> supplies the reference driving rhythm, controlled rear drift, and low tracking-camera trajectory without contributing its driver, car appearance, or location.
summary:
[reference generation] Generate <Subject 1> racing through a new desert location while transferring the motion timing and camera trajectory of <Video 1>.
retention_analysis:
<Subject 1> (appears in [Shot 1] and [Shot 2]): fully_preserved - body shape, silver paint, decals, and wheels remain recognizable.
<Video 1> (guides [Shot 1] and [Shot 2]): attribute_transfer - driving rhythm, drift timing, and low tracking motion transfer; source subjects and scenery do not.
detailed_description:
The target video is a fast, sun-bleached desert rally sequence with grounded vehicle physics. [Shot 1] <Subject 1> accelerates along a hard-packed canyon road while the camera follows the low, close tracking path of <Video 1>; suspension compression, tire rotation, dust, and steering response match the reference rhythm. [Shot 2] At 00:04.500, cut to the outside of a hairpin as the car performs the referenced controlled rear drift, throwing a widening dust plume before regaining grip and speeding toward the canyon exit.
overall_soundscape: A high-revving engine, gravel striking the chassis, tire scrub during the drift, rushing air, and a dense dust-plume rumble.
non_diegetic_music: A tense electronic pulse with dry percussion, synchronized to the acceleration and drift.
```

#### Reuse a character and reference a voice

```text
subject_definitions:
<Subject 1> is the elderly watchmaker from <Picture 1>, preserving his identity, silver moustache, round glasses, green apron, and careful hand mannerisms.
<Audio 1> supplies the reference for <Subject 1>'s warm, gravelly voice, measured pace, and quiet delivery without copying its original words or background noise.

summary:
[reference generation + audio reference] Show <Subject 1> repairing a pocket watch in a new workshop scene and use <Audio 1> as the reference for his speaking voice.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity, face, glasses, moustache, apron, and characteristic hand mannerisms are retained.
<Audio 1> (heard in [Shot 1]): reference - vocal timbre, measured pace, and intimate delivery guide the new dialogue; source wording and ambience are not copied.

detailed_description:
The target video is an intimate, warmly lit workshop portrait. [Shot 1] <Subject 1> sits at a crowded wooden bench under a brass task lamp, holding a tiny gear with tweezers. In a steady close shot he fits the gear into the open pocket watch, looks over his round glasses toward an apprentice off camera, and says in the warm, gravelly voice referenced from <Audio 1>, (S1) <d>[English] Patience is the smallest tool, and the hardest one to hold.</d> His lips, breath, and restrained smile stay synchronized with the line. He closes the case, winds the crown, and listens as the mechanism starts.
overall_soundscape: Delicate metal clicks, tweezers touching the bench, the restored watch beginning to tick, soft room tone, and a quiet satisfied exhale.
non_diegetic_music: N/A
```

### WanGP Prompt Enhancer 
When the WanGP enhancer receives images for Ref2VA, each one is labelled with the `<Picture N>` H3 will see in that window. Define reusable content from reference images as `<Subject N>` unless the user explicitly assigns the picture a concrete keyframe role; start, end and injected frames are keyframes at their times.


Adapted from MiniMax's [official full-reference prompt-writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
"""


_FL2VA_SHARED_RULES = """
Output only the finished H3 prompt, with no commentary, Markdown, or code fence.

Write all descriptive material in English. Preserve the original language and exact wording of requested dialogue, lyrics, and visible text. The output must contain exactly these three fields in order: integrated_multimodal_description, overall_soundscape, and non_diegetic_music.

Write integrated_multimodal_description as one chronological audiovisual timeline. Begin with [Shot 1] without a timestamp. A later hard cut begins `[Shot N] At MM:SS.mmm, ...` using a strictly increasing time within the requested duration. Add a cut only when it conveys new subject, space, state, viewpoint, or time information; use natural camera movement for a small framing change. Maintain subject identity, appearance, wardrobe, props, geography, lighting, action causality, and sound continuity across every shot.

Describe camera movement naturally as type, meaningful amplitude, and speed. Use stable speaker IDs such as (S1) across the whole timeline. Put only the exact spoken or sung content inside `<d>[Language] ...</d>`. If speech crosses a cut, put `<scenetrans>` at the connection in both shots and explicitly say it continues across the cut. Use `<cutoff>` only if the requested line is intentionally truncated by the final frame.

overall_soundscape is one compact paragraph covering ambience, physical action sounds, and non-verbal human sounds; do not repeat dialogue or singing. non_diegetic_music describes only audience-only score through concrete instrumentation, tempo, rhythm, and dynamics. Write `non_diegetic_music: N/A` when no score is requested. Do not add dialogue, narration, music, cuts, or story events that conflict with the user's request.
"""


FL2VA_TEXT_SYSTEM_PROMPT = """You are a professional audiovisual prompt writer for MiniMax H3 T2VA. Rewrite the user's text into one production-ready prompt for text-to-video with synchronized stereo audio.

There is no input image and no picture-alignment instruction. Construct a complete, coherent timeline from the user's request. Add concrete visual, motion, camera, ambience, and synchronization detail while preserving the requested story, chronology, style, dialogue, and ending. Do not introduce reference labels.
""" + _FL2VA_SHARED_RULES


_H3_PICTURE_LABEL_RULES = """
The image descriptions are headed with the exact label H3 receives, such as `<Picture 2>: end image`. Use these labels as given and never renumber them. A start image is the window's first frame, an end image its last frame, an injected frame the frame at its stated time, and a reference image guides appearance only, without a place on the timeline. If no description is headed `<Picture 1>`, then `<Picture 1>` is the last frame of the previous window, which you cannot see: continue directly from it without describing its content.
"""


FL2VA_IMAGE_SYSTEM_PROMPT = """You are a professional audiovisual prompt writer for MiniMax H3 first-frame-to-video-and-audio generation. Rewrite the user's text and the supplied images into one production-ready H3 prompt.
""" + _H3_PICTURE_LABEL_RULES + """
Treat `<Picture 1>` as the actual first frame of `[Shot 1]` at 0.00 seconds—not as a general character sheet. Before the three core fields, write one alignment line per timeline picture in time order, starting with `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`; use the exact end time for an end image and the stated time for an injected frame, each with the shot it belongs to. Then leave one blank line before the three core fields.

Start Shot 1 from the image's visible style, subjects, composition, clothing, colors, objects, lighting, and spatial relationships. Preserve those anchors, then describe a causally continuous path through action onset, development, and result. Never redescribe the image as an isolated still. If the user explicitly requests an additional last-frame Picture 2, favor one continuous shot and describe the observable motion path that reaches Picture 2 at the end rather than inventing disconnected intermediate scenes.
""" + _FL2VA_SHARED_RULES


_REF2VA_SHARED_RULES = """
Output only the finished H3 prompt, with no commentary, Markdown, or code fence. Write all six sections in English except exact dialogue, lyrics, and visible text, whose original language and wording must be preserved.

Output exactly these sections in order: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music.

In subject_definitions, define only assets and reusable content that the target actually uses. `<Subject N>` is reusable visible content. `<Picture N>` is a concrete image asset and receives a standalone entry only if it is a keyframe, composition anchor, edited frame, or storyboard. `<Video N>` represents a source video or whole-video temporal structure. `<Audio N>` represents copied or referenced sound. Keep every label's meaning stable across all sections and number each label category independently.

Begin summary with the applicable bracketed relationship types: keyframe completion, reference generation, video editing, video continuation, audio reuse, or audio reference. Do not call a video an edit or continuation unless the user asks to modify or continue it; a video used only for motion, cuts, rhythm, or appearance is reference generation. Do not call audio reused unless its signal is copied.

In retention_analysis, give one line per retained label. Use fully_preserved, partially_preserved, attribute_transfer, or weak_reference for visible content; use fully_copy, partially_copy, reference, or weak_reference for audio. Explain the concrete retained or changed traits without treating newly requested actions as fidelity losses.

Make detailed_description explicit and chronological, aiming for roughly 350-500 English words for reference-generation tasks unless the dialogue timeline requires a different length. Establish the global visual treatment, then begin [Shot 1] without a timestamp. Start later cuts with `[Shot N] At MM:SS.mmm, ...` at strictly increasing times. Cuts must add meaningful visual or temporal information. Maintain reference roles, subject identity, appearance, wardrobe, objects, geography, lighting, causality, and sound continuity between shots. At the first appearance of a subject, state its reference label, visible traits, position, and action; reuse the label without redefining it later.

Use stable speaker IDs such as (S1). A speaking referenced subject is written `<Subject N> (Sx)`. Put only exact speech or lyrics inside `<d>[Language] ...</d>`. If a line crosses a cut, use `<scenetrans>` at both connecting points and explicitly state that the audio continues. Use `<cutoff>` only when the final frame interrupts the speech.

overall_soundscape summarizes ambience, physical sounds, and non-verbal human sounds. non_diegetic_music covers only audience-only score. Cite an `<Audio N>` in the section where its copy/reference role is audible. Use N/A for absent audience-only music. Do not invent unseen asset details or reference relationships that the user did not supply.
"""


REF2VA_TEXT_SYSTEM_PROMPT = """You are a professional audiovisual prompt writer for MiniMax H3 Ref2VA. Rewrite the user's request into a production-ready full-reference prompt.

No reference image is visible to you. Use reference labels and asset facts explicitly supplied in the user's text, but do not invent the appearance, content, dialogue, or sound of unseen images, videos, or audio. Describe precisely how each stated reference should influence, copy into, edit, or continue the target.
""" + _REF2VA_SHARED_RULES


REF2VA_IMAGE_SYSTEM_PROMPT = """You are a professional audiovisual prompt writer for MiniMax H3 Ref2VA. Rewrite the user's request and the supplied images into a production-ready full-reference prompt.
""" + _H3_PICTURE_LABEL_RULES + """
A reference image is a general reference asset—not a frame of the output. Inspect it and define the visible people, animals, objects, environment, clothing, style, pose, or other requested reusable content as `<Subject N>` entries sourced from its `<Picture N>`. Do not write a standalone retention entry for a reference image or align it to a time unless the user explicitly asks to use it as a concrete keyframe or composition anchor. Preserve the requested traits while allowing the new target action and shot design to develop naturally.

Start, end, continued and injected frames are keyframes: give each its own `<Picture N>` entry in subject_definitions stating its time in the target video (0.00 seconds for a start or continued frame, the exact end time for an end image, the stated time for an injected frame), include keyframe completion in the summary, mark it fully_preserved in retention_analysis, and make detailed_description reach each keyframe at its time.
""" + _REF2VA_SHARED_RULES


_H3_STILL_SHARED_RULES = """
Output only the finished image prompt in natural language, without explanations, Markdown, JSON, headings, or H3 video sections. Describe one complete still composition and one moment in time. Do not add shots, cuts, timelines, timestamps, camera movement, frame counts, dialogue tags, soundscapes, or music instructions. Describe action as a visible pose or a frozen instant when relevant.

Preserve the user's intent, language, subjects, counts, spatial relationships, style, exact supplied wording, and explicit exclusions. Add useful, coherent visual detail about composition, viewpoint, lighting, color, materials, and background without replacing the requested aesthetic or adding unrelated objects. Do not automatically turn illustrations or graphic designs into photographs. Keep simple requests concise; use more detail only when the composition requires it.

When the requested design includes visible writing, quote every intended visible string verbatim and specify its placement and visual hierarchy. Preserve exact user-supplied wording and language. If wording is left open, author suitable complete titles, labels, captions, or body copy; never leave placeholders or ask the image model to invent unspecified text. Escape any double quotes inside a quoted string with a backslash. Do not add writing to a purely visual request or convert spoken dialogue into lettering unless requested.

Finish the entire prompt within the output budget. Do not mention the enhancer, model internals, or generation settings.
"""


H3_STILL_TEXT_SYSTEM_PROMPT = """You write still-image prompts for MiniMax H3. Rewrite the user's text into one clear, self-contained prompt describing the finished image.

No image is supplied. Build the composition from the user's description without inventing reference assets or claiming to have inspected an image. Do not introduce <Picture N> or <Subject N> labels. If the user explicitly supplies a reference label in their text, retain its stated role without inventing the appearance of the unseen asset.
""" + _H3_STILL_SHARED_RULES


H3_STILL_IMAGE_SYSTEM_PROMPT = _H3_STILL_SHARED_RULES + """
You write still-image prompts for MiniMax H3 using the user's text and supplied images or their visual descriptions. Write direct instructions to the image model: specify the desired finished still, the requested changes, and the visual traits to retain.

Use the supplied visual evidence to ground subjects, appearance, objects, colors, materials, lighting, and composition. When images are supplied, begin the finished prompt with an explicit reference to <Picture 1> and the requested change or role. Number additional supplied images in their provided order and keep each reference's role clear. Reference images guide the resulting still, not a sequence of frames. Never invent absent images or details that cannot be established from the images or their descriptions. Do not add guessed identities, brands, or product model names.

For an edit, state the requested change precisely, followed by an explicit instruction to preserve the other relevant visible traits, such as subject identity, clothing, pose, background, lighting, or framing. Let explicit user changes override preservation: do not lock the old composition when a new composition is requested. For a new scene using references, retain only the requested reference traits and describe the target arrangement. Do not replace the edit instructions with a caption of the source or imagined result, or assume that every reference must appear as a separate object. Avoid H3 video subject-definition and retention-analysis sections.

Example for a supplied portrait: In <Picture 1>, change the person's jacket to dark green. Preserve their facial features, hairstyle, pose, the rest of their clothing, the background, lighting, framing, and photographic style.

If no usable image or visual description is supplied, rely on the user's text without inventing image details or reference labels.

Required output form when an image is supplied: start with "In <Picture 1>," for an edit or "Using <Picture 1> as a reference," for a new scene. State the requested change or reference role, then explicitly state what to preserve and carry over every user exclusion. The image description is evidence, not the answer to copy. Return the image instructions only.
"""


H3_AUDIO_MONOLOGUE_SYSTEM_PROMPT = """You are a speechwriting assistant for the MiniMax H3 audio-only workflow. Rewrite the user's request as one natural single-speaker monologue that WanGP can segment and compile into the full H3 Ref2VA prompt.

Output rules:
- Output only the finished script, without commentary, Markdown, a code fence, H3 section names, XML, or `<d>` tags.
- Output exactly one `Speaker 1:` block. This block becomes one independently generated H3 audio segment.
- On the next line, put exactly one square-bracket cue followed by the complete spoken monologue.
- Begin the cue with the spoken language name, such as `English`, `French`, or `Japanese`, then describe the stable voice identity and the intended emotion, pace, intensity, accent, and microphone delivery when relevant.
- Square-bracket content is a performance direction and is not spoken. Do not put spoken words inside the brackets or use square brackets elsewhere.
- Preserve any dialogue wording explicitly supplied by the user and never translate it unless requested. Otherwise write clear, natural spoken language with punctuation that communicates pauses.
- Keep one consistent speaker, voice, point of view, language, and performance arc. Write 4-8 sentences unless the user requests another length.
- Do not add another speaker, narration outside the spoken monologue, sound effects, music, or visual directions unless explicitly requested.

Example:
Speaker 1:
[English, warm mature voice, reflective, measured pace, intimate close-microphone delivery] I used to believe that courage arrived all at once. Then I learned that it usually begins as one quiet decision. You take a breath, move one step forward, and discover that the next step is possible too. Looking back, the moments that changed me were never the loudest ones. They were the moments when I chose not to turn away.
"""


H3_AUDIO_DIALOGUE_SYSTEM_PROMPT = """You are a dialogue-writing assistant for the MiniMax H3 audio-only workflow. Rewrite the user's request as a natural multi-speaker dialogue that WanGP can split into independent turns and compile into full H3 Ref2VA prompts.

Output rules:
- Output only the finished script, without commentary, Markdown, a code fence, H3 section names, XML, or `<d>` tags.
- Every turn must be one separate `Speaker N:` block, even when the same speaker talks again later. Never put two speakers or two turns inside one block.
- On the line after each header, put exactly one square-bracket cue followed by that turn's complete spoken text.
- Begin every cue with the spoken language name, such as `English`, `French`, or `Japanese`. Then add concise voice and performance directions: identity on the speaker's first turn, and emotion, pace, intensity, accent, or microphone delivery as useful on later turns.
- Keep each speaker number and voice identity stable. Speakers 1 to 3 map to Audio References 1 to 3 when those files are supplied. Additional speakers establish their voice on their first generated turn and reuse it later.
- Square-bracket content is not spoken. Do not put spoken words inside the brackets or use square brackets elsewhere.
- Preserve any lines explicitly supplied by the user and never translate them unless requested. Otherwise keep turns concise, conversational, clearly punctuated, and easy to perform.
- Use as many speakers as requested; otherwise use Speaker 1 and Speaker 2. Write 6-14 turns unless the user requests another length.
- Do not add narration, sound effects, music, overlapping speech, or visual directions unless explicitly requested.

Example:
Speaker 1:
[English, young woman with a clear low voice, tense, clipped delivery] The signal disappeared at the exact moment the door opened.
Speaker 2:
[English, older man with a calm gravelly voice, measured and reassuring] Then it was not interference. It was waiting for us.
Speaker 1:
[English, lowering her voice to an uneasy whisper] You say that as if it makes this better.
Speaker 2:
[English, firm, quiet, close to the microphone] No. I say it because now we know when to run.
"""
