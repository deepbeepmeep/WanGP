---
name: edited-film
description: Assemble an edited film from several shots - people speaking on camera, inserted stills, turntables or cutaways, narration, music, other language tracks and subtitles - so that every insert shows what is being talked about and every track and subtitle follows the picture.
---

# Edited Film

The timeline comes first: every shot, sound and subtitle is placed on it, and the checks compare them at the same moments.

## 1. Timeline

- Write the shot list in the plan document as a timeline: for each shot its source (generated clip, still, turntable, card), its length, what it shows and the words heard over it. An insert sits where the speech names its object.
- Make one master image per recurring person and reuse it as the start image or reference of each of their shots. The shot template's
  `wangp_templates` result says which one its model takes: `model.media_inputs.image` lists `reference` when it takes reference images,
  `multiple_references` when it takes several (its `infos` give how many), and `start` for a start image.

## 2. Recurring objects

- Anything designed elsewhere that a shot must show (an artifact in a display case, a product, a prop, a second character) comes from its own image. With a model that takes several references (`multiple_references`), give that image beside the person's master in `image_refs` and bind each in the prompt: it keeps the object without fixing the frame. Otherwise it goes into that shot's start image: edit it in from the person's master image and the object's image, then animate it. A video model told only the object's name invents another one, and the insert beside it then contradicts the shot.
- Compare the object in each shot with its own image before assembling, and the 3D model behind a turntable with the image it was built from.

## 3. Speech and sound

- One voice per speaker for the whole film: make each speaker's voice sample once (`gen_speech_from_description` of a neutral sentence,
  or the user's recording) and speak every line with `gen_speech_from_sample` from it. Describing the voice again for each line gives
  another voice each time.
- Drive each speaking shot by its own speech clip (`gen_video_with_speech`) with `video_length` set to the speech's duration in seconds (such as `"6.4s"`).
- Two people talking in one shot: a model that takes audio references (`model.media_inputs.audio` lists `reference`; check the
  `gen_video_with_speech` and the `gen_video_with_refs` defaults) can voice both. Give each speaker's voice sample (at least 2 s) as an audio reference (`audio_prompt_type` `AB`: `audio_guide`, `audio_guide2`),
  bind it to that speaker's picture in the prompt (`<Subject 1> is the woman from <Picture 1>, speaking with the voice of <Audio 1>`),
  and write the words the way its `prompt_infos` say. The model then says the words itself, less exactly than a spoken clip: keep one
  speech clip per shot for a single speaker.
- A line shorter than its shot: pad the clip with silence to the shot's length, and cut the shot about 0.35 s after the line ends. The
  model fills the silence with invented words, so the part after the line is not usable.
- Describe a speaking shot as the person speaking to camera with natural lip movements synchronized, and nothing that competes with the mouth: no written line of dialogue, no language or audio mention, no smiling, waving or nodding. Written dialogue in the prompt leaves the lips still in about half of the shots.
- A lip-synced shot carries its speech in its own soundtrack, aligned with the lips: mix with `include_video_audio: true`, add only music and other sounds, and duck the music under it with `ducking: {"tracks": [<music index>], "under": ["video"]}`. Placing the speech files again at computed offsets drifts them away from the lips.
- Narration over inserts: one clip per insert, placed where the insert starts (`remux_media` mix `offsets_s`).
- `merge_videos` with cuts keeps every clip's speech aligned; a crossfade fades the last `transition_s` of the outgoing clip's sound, so cut around speech.
- A change of actor is a change of look and of voice: make the new master and voice sample once, rewrite every prompt that describes them, shoot their shots again and speak their lines again from the new sample.
- Generate the music at least as long as the whole cut, again when shots are added, and mix once: `remux_media` mix with the music ducked under the voices and `normalize_lufs`. Its `quiet_ranges_seconds` must be empty unless a silence is intended.

## 4. Other languages

- A translated track follows the same timeline: translate segment by segment, generate one clip per segment and place each where that segment's original speech starts, over the same music; then add it as a second track (`remux_media` multitrack, `audio_languages`). Never lay one continuous translated read from 0 s.
- Lip-synced shots match the language they were generated with; regenerate them per language when the lips must match.

## 5. Subtitles

- Time subtitles from the finished audio, never by estimate: `transcribe_media` on the final cut with `subtitles_path` (and `audio_track_no` for each language track) writes an SRT timed to the film. Correct misheard names to the script's spelling without moving the times, then add the files with `remux_media` `subtitle_tracks`. Never burn subtitles into the picture (`draw_svg` over a video is for titles and labels): a subtitle track can be switched off and replaced.

## 6. Check before reporting

- A full-frame card (a chapter, an end card) hides the picture: compare its times with the subtitle times, and start it after the last line it would cover.
- One `inspect_video` over each inserted shot, asking which object it shows; compare the answer with what the speech names at that moment and with the same object in the neighbouring shots.
- For each language track, compare its transcription's segment times with the timeline: each segment must start where its shot starts.
- `black_ranges_seconds` and `cut_times_seconds` (generated videos, Inspect Video, View Media) and `quiet_ranges_seconds` (mix) are measurements: fix them unless intended, and report what was measured, not what was hoped. Inside a generated shot, and above all an interview or lip-synced shot, a cut is the model cutting away: use the part before it or regenerate the shot shorter.
