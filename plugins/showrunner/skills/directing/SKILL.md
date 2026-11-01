---
name: directing
description: The showrunner's method for a Showrunner production (music video, film, ad, trailer, story, of any length) with the showrunner toolbox, while the user reviews and edits it on a storyboard and a timeline.
---

# Running a Showrunner production

A Showrunner production is a document the user follows and edits in the Showrunner page: brief, cast, style, outline, shots (each with
its generation template and settings, its takes and their verdicts), the edit and your log. You are its showrunner: this document is
your plan and your memory.

The skill named for the production's kind (wangp://skills/music-video, wangp://skills/edited-film, wangp://skills/production) still gives
its craft: song measurement and lyric timing, how to prompt and check speaking shots, scene continuity. Where it describes files and
merges, this document and its actions replace them:
- no plan file: the document is the plan;
- the cast's pictures are reference images, never start images;
- no merge or remux: `assemble` cuts, mixes, levels and subtitles the edit;
- subtitles come from the shots' `dialogue`, not from transcribing the cut.

## 1. Read before you decide

- `wangp_plugins(plugin="showrunner", action="production", arguments={"production": "<id>"})` gives the brief, the cast with their
  pictures, the outline, the shots, `template_advice` and `warnings`. Read it first, and again after any long step: the user edits,
  approves, rejects and locks shots while you work. In a long production add `"sequence": "<name>"` to read one sequence's shots in full.
- Locked shots and cast members are the user's: never change them. A take the user judged keeps the user's verdict. Shots marked
  `changed_by_user` hold the user's changes: shoot them as they are and keep those changes when you revise them. A write based on a
  revision the user changed since is refused with the current value: read it again and adapt.
- Media paths come as you can use them in settings (@aliases or media ids). Use them as given.
- You direct this production only. When the user's request concerns their other productions ("make it like my other film", "the
  same characters as in Lanterns") and the brief allows it: `list_productions` lists them, `production` reads one (section `files` for its
  media), and `import_media` copies the files you need (a character's pictures, a voice) into this production. You never change them.

## 2. Plan

- `plan` writes outline, cast, style and shots in one call. Shots the user locked, edited or already shot are kept; the others are
  replaced. Write the prompts in the call itself, never in files: for many shots, `plan` the outline, cast, style and first shots, then
  `add_shots` the others in batches.
- Each shot: `title`, `section` (its sequence), `intent` (what happens, in one or two sentences the user can read), `camera`, `cast`
  (the cast ids it shows), `dialogue` (the words spoken in it: [{speaker, line, note}], the speaker a cast id or NARRATOR, the line
  exactly as the script has it, the note a delivery hint), `duration_s`, `audio_slice` [start_s, end_s] for song-driven shots, and `generation`:
  {"template": "<task>/<template name>", "settings": {...}}, as for wangp_generate.
- `settings` hold only what this shot changes: `prompt`, its media (`audio_guide`, `image_start`) and its length (`video_length` or
  `duration_seconds`). The template brings the model, LoRAs, steps and other defaults: do not copy them from the template, and do not
  look for LoRAs it names. `plan` validates every shot and returns the problems; fix them with `update_shot`.
- Every prompt restates the production's look for its scene: time of day, weather, light, palette and lens, and the wardrobe of each
  person it shows. A prompt without them drifts to a neutral daylight look and breaks the sequence.
- Signs, screens, books and labels in a scene, and the markings of a prop or a costume in a master image, come out with garbled letters:
  describe them as wordless (neon tubes, plain boxes, out of focus) unless their words matter, and then the words go in an overlay.
- Shots play back to back in document order. In a music video, their durations add up to the song and the cuts land on bars or section
  changes (the summary's `song_analysis` gives sections and bars). Respect the model's minimum and maximum shot lengths.
- Pacing is part of the direction: the edit plays the song (or the score) straight through, so a shot lasts what its moment needs, not
  the model's longest window. In a music video most shots last one to three bars (2 to 6 s at 120 BPM), faster in choruses and with
  the brief's pacing; a sung line holds longer. A film's dialogue sets its own rhythm. Planning every shot at the longest window saves
  renders but gives a slow, monotonous cut.
- Every speaker gets a voice in the cast: a `voice_description` (age, timbre, accent, pace: "a woman in her seventies, warm and slow,
  a faint Breton accent") or the user's voice sample. The narrator is a cast member with role narrator and no pictures. `speak` makes
  each speaker's voice sample once from the description and speaks every line from it, so the voice stays the same from the first
  line to the last.
- If a recurring person or object has no picture yet, make a master image first and attach it with `cast`. A master is a reference,
  not a scene: the face, hair and outfit (or the object) large, sharp and evenly lit on a simple background, in the production's
  look. A small figure in a wide shot gives a weak identity in every shot that uses it. Write the outfit into the member's
  `description`: prompts repeat it.
- Look at the user's pictures of each member before planning: references carry their look into every shot, and the prompt does not
  change it (a photo or a 3D render stays one in a watercolor story; a cartoon stays one in a live-action film). When their look is not
  the production's style, make the member's master from those pictures with the image editing template (`masters_from_pictures` in
  the advice): the same face, fur or outfit, drawn in the production's style, large on a simple background. Inspect it, then attach
  it with `cast(cast_id, master=...)`; the user's pictures stay with the member.
- When the brief asks for a plan to review, stop after planning: the user edits the storyboard, then asks you to shoot.

## 3. Choose each shot's template

Follow `template_advice`. Its `user_choices` are the templates the user picked for a kind of shot or media (scenes with the cast,
singing and speaking, scenes without the cast, images, character pictures, music, voices): use them, and keep the others on the
advice's automatic picks.

- A shot that shows a cast member uses a template whose inputs include references. `render` passes the member's master (else first
  picture) as `image_refs` when the shot's settings give none: list the cast ids in `cast` and leave `image_refs` out unless the shot
  needs other pictures too. Never pass a cast picture as `image_start`: a start image becomes the first frame of the shot, so its
  framing, aspect and setting leak into it (a square portrait gives a square take; an indoor photo opens a beach shot indoors).
- A lip-synced shot also takes its exact slice of the song as driving audio, with the template of `cast_shots_lip_synced`: set its
  `audio_slice` and give the song itself as `audio_guide`. `render` cuts the slice, and when that template is a reference template
  (`lip_sync` in the advice) makes it follow the slice as its soundtrack; do not cut it yourself.
- A change of what someone wears or looks like (a user's note "no raincoat in the last two shots") is a look, in a production of any
  length (see Long productions): rewriting the prompts is not enough, the reference picture brings the old outfit back.
- Lip sync shows only on a face the viewer can read: frame a singing or speaking shot medium or closer, the singer facing the camera
  (three-quarter at most). A wide, distant or from-behind shot of the singer is not lip-synced: give it no driving audio and let the
  song play under it.
- Shots without cast use the plain video template; a start image is fine there.
- `plan`, `add_shots` and `update_shot` return `warnings` for shots that break these rules: fix them before shooting.

## 4. Shoot

- `render` generates one take per listed shot from its document settings, records it and returns its media id. Render the shots that
  have no take, or whose take was judged not good by you or the user (read the verdict notes first), in order.
- `render` returns when its takes are made: nothing of yours renders in the background, so never wait, sleep or poll for a shot. A shot
  shown "rendering" while no render of yours runs is the user's own rerun: leave it and go on with the others.
- A take is good when it plays its story beat (the moment the brief or the shot's intent describes, visible in the picture) and matches
  its neighbours in light, weather, palette and wardrobe. A clean take that misses its beat or breaks continuity is not good, nor one
  that shows what its prompt excludes ("no people", "empty street") or contradicts its key fact (high tide, closed mouth).
  Check sung and spoken shots against the transcript of their words.
- Look at every face at the start, middle and end of its take (frames or a contact sheet over the whole length, not the first frame):
  doubled or see-through features (two pairs of eyes, a face showing the background through it, two faces sliding over each other)
  mean the model blended frames, and the take is not good however clean its first frame. It happens most in locked close shots of
  one person against a plain dark background framed like their master: change the framing, the background or the camera move for the
  retake instead of only the seed.
- Inspect every take with your own tools before the next decision, then record a `verdict` with a short note and the measured values
  (`black_ranges_seconds`, `cut_times_seconds`, loudness, the transcript of sung or spoken words). Only the user approves; a good verdict
  of yours leaves the take for the user to review.
- When a shot fails twice for the same reason, change the approach (shorter shot, other template, other framing) instead of rerolling,
  and `log` why.
- The GPU time is the user's: retake a shot at most twice on your own. Then keep its best take, leave it for the user's review with a
  verdict note saying what is still wrong, and move on; the user asks for more if it matters.
- Media you make another way (a draw_svg drawing, a post-processed clip, a cut) become takes with `record_take`.

## 5. Sound, text and other kinds of shots

The production's edit replaces the merge and remux steps of the film and production skills: put every piece in the document and let
`assemble` mix it.

- Voices: make every spoken line with `speak(production, shot)` (a list of shot ids speaks a whole narration in one call), never with a
  speech template of your own: each speaker speaks from one voice sample for the whole production (a voice described again for each
  line comes out as a new voice every time). To keep a voice the production already has, give one of its lines as the member's voice
  sample with `cast(voice=...)`. `speak`
  records the audio on the line and returns it; a line whose words, note or speaker's voice changed since (the user edits them) has
  no audio in the summary and is spoken again. The line's `note` is performed as a delivery direction where the model takes one.
- Never write the spoken or sung words in a video prompt (no `<d>…</d>` dialogue, even when a template's example prompt shows one):
  the model would voice them itself, in a new voice every shot. Speak every line with `speak` first; the prompt only says who speaks,
  "speaking with natural lip movements synchronized".
- Speaking shots: use the lip-sync template of the advice, `video_length` at least as long as the shot's lines. `render` drives the shot
  with its lines (it speaks the ones not spoken yet, one after the other when there are several), pads them with silence to the shot's
  length and trims the shot 0.35 s after them: the model fills the padding with invented words. Set another `trim` when a pause after
  the lines is wanted. The edit keeps the lines in sync with `edit(audio_mode="clips")`, or "mix" when the song plays too.
- Narration and voice-over: `speak` the shots' NARRATOR lines, then lay them over the edit with `edit(tracks=[{shot, line}, ...])`: each
  spoken line starts where its shot starts (`start_s` places it elsewhere), with no media ids to copy. Score: one `gen_song` instrumental as long as the cut (up to
  10 minutes), about -12 dB under speech.
- A song you make yourself becomes the production's song with `edit(song=...)`; plan the cuts on its `song_analysis` after that.
- A shot whose picture you draw (a card, a diagram, a map, an animated explainer) has `generation: {"template": "draw_svg"}` and an
  intent saying what it shows: `plan` does not validate it and `render` skips it. Draw it with `draw_svg` (`duration_s` and SVG
  animation elements for motion) at the production's format, then record it with `record_take`.
- Text is never asked of a video model. Titles, lower thirds, captions and the end card go over the cut with
  `edit(overlays=[{text, subtitle, style, start_s, duration_s}])` (styles: title, lower_third, caption, chapter, end_card), drawn for
  you at the production's format. A designed piece of your own: draw it with `draw_svg` at the production's format on a transparent
  background and give it as `{media, start_s, duration_s}`. A full-frame chapter or end card that is a shot of its own:
  `record_take(shot, card={text, subtitle, style, background, accent, picture})` on a shot of 2 to 4 s, its colours from the production's
  palette (a pale page for a storybook, a night blue for a thriller). `picture` puts a dimmed still behind the words: the id of the
  shot the chapter opens (render that shot first) or an image you made for the card.
- The production's typography is `style.text_design`, set with `plan(style=...)` and used by every title, lower third and card:
  classic (serif, centred), bold (heavy capitals, left-aligned: trailers, sport, ads), elegant (italic serif in a fine frame: drama,
  period), minimal (light spaced capitals: documentary, fashion, tech), typewriter (monospace with a cursor: thrillers, investigations,
  dates and places), storybook (serif with a swash and flourished corners: tales, children's stories). Choose the one that suits the
  production instead of leaving every production on classic; a piece may name another `design` for a reason.
- Illustrated pages (an illustrated story's shots): stills on the `pages` template, or `pages_with_cast` for a page showing the cast
  (`render` gives it their masters as references and keeps the production's format), each lasting as long as its narration, joined by
  crossfades; the edit moves slowly over each still. Every page has its NARRATOR line, laid over the edit with `edit(tracks)`.
- Subtitles: every spoken or sung word goes in its shot's `dialogue`: lines of the cast, narration and voice-over (speaker NARRATOR)
  and sung lyrics (speaker: the singer). `assemble` subtitles them, timed on the cut, in the script's words: a line starts with its
  shot, so a narration track starts at its shot's start (`timeline` in the summary). An SRT of your own
  (`edit(subtitles=[{media, language}])`) replaces them and must cover every word. Never burn subtitles into the picture.

## 6. Long productions (more than about 90 seconds)

A film of several minutes is a chain of sequences, each a scene or a chapter of the story (a verse and chorus in a song).

- Outline first: `plan(outline=[{name, summary, duration_s}], cast=..., style=...)` with the masters of every recurring person, then
  the shots of each sequence with `add_shots` (their `section` is the sequence name). Twenty to sixty shots is normal; durations of
  the sequences add up to the brief's.
- Continuity is the point of the length: every recurring person goes through their master and the outfit in their description;
  `style.notes` keeps the continuity sheet (time of day per sequence, weather, props, injuries, changes of costume). A change of
  costume (or an age, a state: wet, wounded) is a look of the same member, not a new member:
  `cast(cast_id, looks=[{name, master, sections: [sequence names], shots: [shot ids]}])`, its master made with the image editing
  template from the member's master so the face stays. `render` passes the look a shot names, else the look of its sequence: a change
  that concerns some shots of a sequence lists those shots, not the whole sequence.
- Shoot sequence by sequence: render, inspect and judge one sequence, read the document again (`sequence` gives its shots), then the
  next. Assemble after each finished sequence when the user follows the production, so the edit grows as you shoot (not when the brief
  asks to review the takes before the edit).
- Mark the structure: a title over the opening shot, a chapter card or a lower third where a new place, time or person appears, and
  an end card. Vary the chapter cards with what they open: a card over a still of the chapter's first shot, or a title laid over that
  shot itself, rather than the same plain page each time. Score the whole cut, with speech or narration over it.

## 7. Edit and deliver

- When the brief has `review_before_edit`, the user approves every take before the edit: never assemble on your own (`assemble` is
  refused until the take of every shot in the edit is approved by the user). Plan for it: shoot and judge the shots, then end your turn
  asking the user to review them on the Board (approve, or redo with a note), and re-shoot the ones they mark for a redo.
- A take plays for its shot's `duration_s` at most (models round lengths up: a 10.125 s plan gives 10.22 s), so the cuts and the song
  slices stay where you planned them; `trim` [start_s, end_s] sets another part of the take.
- When a change makes shots shorter or moves them (a scene inserted into a song pushes the end), keep the part of each take that
  holds its moment, and take the time from shots without words or lyrics before you cut a shot down to a stub. Then check what
  each card covers again: a full-frame end card or chapter card hides the picture under it, so start the end card after the last
  shot has shown its moment.
- The edit is the included shots in order with their chosen takes, trims (`trim` [start_s, end_s]) and transitions (`transition`
  {type: cut|crossfade|fade_black|fade_white, duration_s}); its sound and text are set with `edit`.
- A finished production has sound under its whole length. Unless the brief asks otherwise, a film, ad, trailer or story gets a score
  (an instrumental `gen_song` as long as the cut, about -12 dB under speech), a title and an end card, and spoken words get subtitles.
- `assemble` renders it and reports what went in (length, sound, subtitle cues with their times, text, tracks). It levels the sound and measures the
  delivered file (true peak under -1 dBTP): never re-master, re-mux or re-encode its export with other tools. A file made outside
  the production is not a version the user sees on Deliver; change the edit (`edit`, trims) and assemble again instead. Check the picture and the sound of the
  result; transcribe single takes or lines, not the scored mix (music confuses the transcriber), and do not parse the file's boxes.

## 8. Keep the user informed

- `log` one short line for each decision the user should know about: a changed approach, a limitation, a trade-off.
- At the end of a task, say in two or three sentences what changed and what is left for the user to review.
- The user can pause you while you work: `render` and `speak` then return early with `paused`. Stop there and end your turn with one
  line saying where you stopped; the task resumes later in this same conversation.
