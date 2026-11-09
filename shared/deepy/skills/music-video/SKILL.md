---
name: music-video
description: Make a music video - an original song, shots of a singer lip-synced to exact slices of it, edited so the song plays without skips, lyric subtitles and a loudness-safe final mix.
---

# Music Video

The song comes first: every shot is driven by an exact slice of it, so that the edited shots play the song back without skips and the lips match the words.

## 1. Song

- Read the `gen_song` default template and keep its settings. Lyrics go in `prompt` with section markers (`[Verse]`, `[Chorus]`), the style in `alt_prompt`. Ask for about 4 seconds more than the video: songs often end with a fade.
- Name the lead voice first and plainly in `alt_prompt`, matching the singer the video shows: `female vocals, female lead singer, ...` or `male vocals, ...`; adjectives such as warm or mezzo can pull a voice lower.
- Measure it: `inspect_audio` with `window_s=1` and `beats: true` gives where the music plays, its loudness and its beats.
- Isolate the voice for timing: `wangp_postprocess(media="<song id>", action="remove_background", arguments={})` keeps the vocals. The envelope of that stem (`inspect_audio`, `window_s=0.5`, `silence_db=-40`) shows the sung blocks: they are the lyric timings. A stem that is nearly silent means an instrumental: regenerate the song with a new seed. Check the voice on the stem with `inspect_audio` `pitch: true`: a female lead sits mostly above about 165 Hz (`voice_pitch_hz`), a male one below; regenerate with a new seed when it does not match the singer.
- Read the words from the song itself, not from the stem: taking the music away damages the voice and `transcribe_media` reads it worse. The stem is for timing. Sung words transcribe unevenly, but a clearly sung line reads nearly word for word: when the transcript reads the chorus and cannot read a verse, that verse is not sung as written, even with a block of voice where it should be. One `transcribe_media` of the song shows it; do not transcribe phrase by phrase. A line that stays unreadable after a new take is worth telling the user, who can listen to it.
- The user's lyrics must all be sung, and a new song keeps them unless the user says otherwise. Compare the sung blocks and the transcript with the lines. A song that leaves lines out, or turns a chorus instrumental, is a failed take: compose again, at most three times, with the lines fewer or the length greater. Do this before anything depends on the song: no shot is re-shot on it. If lines are still missing, stop and tell the user which ones, and let them choose; do not deliver it first and mention it after.

## 2. Singer

The singer sings: lip-sync most of the shots that show them, and all of them in the choruses, with the face toward the camera for the whole shot (an orbit that passes behind them loses the words). Lips follow the voice when the face is large and steady: a medium shot or closer, without spins or fast moves. Wide and fast shots go on parts without voice, or show the singer with the mouth closed. Every shot that shows the singer singing is driven by a slice with sung words, or its lips will not match. A part of the song without voice shows something else: hands, places, the story, the singer with the mouth closed. After a new song, every singing shot is shot again on its new slice.

Make one master image of the singer (`gen_image` default template): face visible, mouth area clear, the look you want in every shot. Reuse it in every shot as a reference image (`image_refs`), so the singer stays the same person. Use it as the start image only when no template of section 3 takes references: a start image becomes the first frame, so its framing and background repeat in every shot.

## 3. Shots on exact slices

- Pick the shot template that keeps the singer's face: the `gen_video_with_speech` default when it takes reference images (`video_prompt_type` contains `I`); otherwise the `gen_video_with_refs` default when its model can keep a given audio as the shot's soundtrack (its model definition's `audio_prompt_type_sources` list `AS`), with `audio_prompt_type: "AS"`, the slice in `audio_guide` and the master in `image_refs`. Use the speech template with a start image only when neither fits.
- Name the template in each shot's settings (`{"template": "gen_video_with_speech", "prompt": ..., ...}` with only the changed keys) rather than copying it: its audio mode keeps the slice as the shot's soundtrack and drives the lips, and its LoRAs come with it. Follow the model's `prompt_infos`, and say that the singer is singing with natural lip movements synchronized.
- Validate one shot's settings before cutting any slice (slices cut to the requested length must be cut again): `wangp_validate` `window_plan.total_seconds` is the length the shot will have, which differs from the requested `video_length` (it is fitted to the model's frame steps). Use that length for every shot.
- Cut consecutive slices of exactly that length with `extract_audio` (`start_s` = previous end, `end_s` = start + length), starting where the song starts. A gap between slices becomes a skip at the cut; an overlap repeats words.
- Vary framing between shots (close-up, medium, wide, another angle) and give each shot its own short prompt; submit all shots in one batch.
- A shot shows what is sung while it is on screen: put the timing of each line beside the shots before writing their prompts. Pictures one line behind the words is the usual mistake.
- Vary the pace with the song and cut on its beat: one slice length for the whole song gives a monotonous cut. End each slice on a
  bar start of the song (`beats.bar_starts`, otherwise a beat of `beats.times`), one to three bars long, shorter in choruses. A
  slice runs from the previous slice's end to its cut, with that span as the shot's `video_length` in frames: `wangp_validate`
  gives the length the shot will really have, and the next slice starts at that real end, so the rounding never adds up and every
  cut stays within a frame step of its beat. Keep the slices consecutive whatever their lengths.

## 4. Edit and master

- `merge_videos` with cuts (`transitions: ["cut"]`): the shots' soundtracks are consecutive slices, so the song plays straight through.
- Master the soundtrack in one call: `remux_media` with `video_id` = the merged cut, `audio_ids` = [the song], `mode="mix"`, `trim_start_s` = [start of the first slice], `trim_duration_s` = [merged length], `normalize_lufs=-14`. One-file mixes go through the true-peak limiter; song models often deliver near 0 dBTP, which clips on some players.
- Subtitles: write SRT files for the lyrics with the timings of the sung blocks (shifted by the first slice's start), then add them in the same `remux_media` call (`subtitle_tracks`, one per language). Check the final file with `wangp_io info` (streams) and `inspect_audio` (loudness, true peak below -1 dBTP).

Open on the title, and end on an end card or a hold after the last words: never stop on them.

## 5. Check

Look at a contact sheet of each shot (mouth open on sung words, same singer), measure the final audio, and check that the merged length matches the sum of the shots. Fix a bad shot by regenerating that shot only, with the same slice.
