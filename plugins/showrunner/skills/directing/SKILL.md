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

- The brief is the user's: no action writes it, apart from `edit(voices_from_model=...)` when the user asks for it, and a rewritten line lives in the shot's `dialogue`. Only the user approves a take: yours passes your check, say so in those words.
- `plan` writes outline, cast, style and shots in one call. Shots the user locked, edited or already shot are kept; the others are
  replaced. Write the prompts in the call itself, never in files: for many shots, `plan` the outline, cast, style and first shots, then
  `add_shots` the others in batches.
- Each shot: `title`, `section` (its sequence), `intent` (what happens, in one or two sentences the user can read), `camera`, `cast`
  (the cast ids it shows), `dialogue` (the words spoken in it: [{speaker, line, note}], the speaker a cast id or NARRATOR, the line
  exactly as the script has it, the note a delivery hint), `duration_s`, `audio_slice` [start_s, end_s] for song-driven shots, and `generation`:
  {"template": "<task>/<template name>", "settings": {...}}, as for wangp_generate.
- Write every shot's prompt in its model's own format, given by template in `prompt_guides`: in `template_advice` before the first shot,
  then by `plan` and `add_shots` with `reference_pictures`, the cast members whose pictures each shot receives, in order. Bind each
  person, vehicle and place to its picture as the format says, and ask for synchronized lips where the shot sings or speaks: a plain
  description loses what these models are trained on. To rewrite prompts already planned, call `plan()` alone first: it returns both for
  every shot.
- `settings` hold only what this shot changes: `prompt` and its media (`audio_guide`, `image_start`). `render` asks for the length the shot
  uses from its `duration_s`: set `video_length` only to want more than that. The brief sets the resolution: a shot sets `resolution` only
  when the user asks for another one for it. The template brings the model, LoRAs, steps and other defaults: do not copy them from the template, and do not
  look for LoRAs it names. `plan` validates every shot and returns the problems (fix them with `update_shot`) and `renders`, what the shots will come out at, {size: {shot: seconds}}: quote these numbers when you tell the user a size or a length.
- Every prompt restates the production's look for its scene: time of day, weather, light, palette and lens, and the wardrobe of each
  person it shows. A prompt without them drifts to a neutral daylight look and breaks the sequence.
- A master's prompt says plainly who it shows, first: "a young woman", "an old man". A pronoun in the middle of a sentence does not reach the picture, and short hair or a uniform can turn someone into somebody else. When the brief leaves it open, decide once: the description, the picture and the voice say the same. Look at every master before anything depends on it (`inspect_media`), against its description and `voice_description`: age, sex, hair, clothes. A wrong one is made again now, when it costs one picture and not every shot. The tools measure the voices: `speak` makes a voice again when it reads as the other sex than its card, says so when it cannot, and warns when a line moves away from its speaker's voice. Act on these warnings: say the sex plainly in `voice_description`, and make the delivery note gentler.
- A location that recurs is a cast member too, with `role` "place": one wide picture of the set as the viewer should see it, without the cast and without words, from the user's photo or `images`, as its `master`. People who belong to the place stay in the picture (the audience in its seats, the staff, the crowd), dressed as themselves and not like the cast: leave them out and the place stays empty, or every shot invents another crowd. List it in the `cast` of every shot that happens there, after the people: its picture goes in as a reference, and that is what keeps the set the same. Words in a prompt give a new set each time, however precise. It holds on close shots too, so shots without people use a reference template as well (`cast_shots`). A side the picture does not show (the seats seen from the stage, a reverse angle) gets a second place card, its picture made from the first with `image_edits`: only one picture of a card goes to a shot. When the place changes (a mess after an explosion), give it a `look` for those shots, as for a costume.
- A shot with lines and more than one face says in its prompt who speaks, and that the others keep their mouths closed: the model moves whichever mouth it finds, a cat's or a statue's, and your inspection has to look for it.
- Everyone visible in a shot is in its `cast`, background included, and the prompt says who is there: a person without a picture comes out as a stranger, and a crowd takes on the costumes of the cast unless the prompt dresses it otherwise.
- Signs, screens, books and labels in a scene, and the markings of a prop or a costume in a master image, come out with garbled letters:
  describe them as wordless (neon tubes, plain boxes, out of focus) unless their words matter, and then the words go in an overlay.
- Shots play back to back in document order: `move_shots(shots, before)` changes the order, and only it (`before` null: to the end). In a music video, their durations add up to the song and the cuts land on bars or section
  changes (the summary's `song_analysis` gives sections and bars). Respect the model's longest shot, and `template_advice` `shot_length` for the lengths it makes.
- Pacing is part of the direction: the edit plays the song (or the score) straight through, so a shot lasts what its moment needs, not
  the model's longest window. In a music video most shots last one to three bars (a bar is 2 s at 120 BPM), faster in choruses and with
  the brief's pacing, as far as `shot_length` allows; a sung line holds longer. A film's dialogue sets its own rhythm. Planning every shot at the longest window saves
  renders but gives a slow, monotonous cut. A shot length the user set in the brief (`shot_length_range` in `template_advice`) comes first: it can go well beyond these habits.
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
  `audio_slice`. On a cut that plays the song, a shot's `dialogue` holds the lyric sung under it, for the subtitles: `render` never speaks it. Give a shot that shows the singer singing its slice. `render` takes the brief's song, cuts the slice and makes the
  shot's model keep it as its soundtrack, whatever audio choice its template or settings hold; do not cut it yourself.
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
- Never write the spoken or sung words in a video prompt (no lines inside `<d>…</d>`, even when a template's example prompt shows them):
  the model would voice them itself, in a new voice every shot. Speak every line with `speak` first; the prompt only says who speaks,
  "speaking with natural lip movements synchronized".
- Unless the advice has `spoken_lines`: the user chose that the video model says the lines (the brief's Voices), knowing that a voice can change from shot to shot, so do not argue for recordings. Write the lines in `dialogue` as usual, exactly as the script has them, and do not `speak` them or lay them on the edit: `render` writes each shot's lines into its prompt, as its model reads them, with the speaker's `voice_description`. Keep their words and "natural lip movements" out of your prompt: that wording is for a recording. Quote nobody there (the model says whatever a prompt quotes, a catchphrase on a cast card too), and call each speaker by their cast name, as the lines do; `render` adds the card of a speaker on screen whom you do not name. These shots last their planned length, not their line's: about 0.4 s a word plus a second, no more, as the model fills time left with words of its own. `render` listens to each of these takes and answers under `heard`: it cuts words before the first line and after the last one, and you need another take when `missing` or `between` is not empty (a line it did not say, words added between two lines). A shot that should run on after its line gets `cut_extra_words: false`. The shots can differ a lot in level; `edit(level_voices=true)` evens them out in the export, when the user asks. A shot of a template that cannot say its lines, or that follows a song or its own audio, works as below.
- When the user asks in words for the video model's own voices, with nothing recorded first ("use H3's audio", "do not pre-render the voices"), call `edit(voices_from_model=true)`, then tell them what it answers under `tell_the_user` and fix its `warnings`; `false` goes back to recordings. Never switch it on your own, and never answer that it cannot be done.
- Speaking shots: use the lip-sync template of the advice, `video_length` at least as long as the shot's lines. `render` drives the shot
  with its lines (it speaks the ones not spoken yet, one after the other when there are several), pads them with silence to the shot's
  length and trims the shot 0.35 s after them: the model fills the padding with invented words.
  A spoken shot therefore lasts about its line: plan the cut's length from the lines, not from the planned durations. A short reply gives a
  shot too short to read: follow it with a silent reaction shot, or set a longer `trim` after `transcribe_media` shows that part of the take says nothing. The edit keeps the lines in sync with `edit(audio_mode="clips")`, or "mix" when the song plays too.
- Narration and voice-over: `speak` the shots' NARRATOR lines, then lay them over the edit with `edit(tracks=[{shot, line}, ...])`: each
  spoken line starts where its shot starts (`start_s` places it elsewhere), with no media ids to copy. Score: one `gen_song` instrumental as long as the cut (up to
  10 minutes), about -12 dB under speech.
- A cut does not outlast its music: when the song ends or fades out before the last shot (the timeline's warnings say so), repeat its last bar under the end, faded out, or end the cut with the song, before you deliver. An end card over silence is a fault.
- Song models cannot sing in a given voice. When the user wants the song (theirs or one you composed) sung by cast members or in voices they
  gave: `change_singer_voice(cast_ids=[...])` or `change_singer_voice(voices=[...])`, one singer or two for a duet, in the order they are first
  heard (SeedVC tells two voices apart, not more), never a new song. It changes the singing and keeps the music, words and timing, so the
  plan, the cuts and the takes already shot stay; the former version goes to the bin. The brief's Sung by (`song_voices`) asks it for the
  song you compose: call it right after `edit(song=...)`, before `check_song`. Without it, when its singers have a voice on their cards,
  tell the user that Change the voice on the song card (or you, when asked) makes it theirs. This requires SeedVC **v1.0 Singing** in
  **Configuration > Extensions** (the default); if a speech mode is selected, ask the user to select Singing first.
- A song you make yourself becomes the production's song with `edit(song=...)`. Then `check_song(lyrics=...)` with the words you wrote for it, before anything depends on it: it says which lines of the lyrics are really sung, and when, and whether the lead voice fits the singer on screen. Its `not_sung` lines are not a transcript's mistake, they are a failed take: compose again as the skill says, and keep the take that sings the most lines. Subtitles follow its times, and an unsung line gets none. Plan the cuts on the song's `song_analysis` after that.
- A shot whose picture you draw (a card, a diagram, a map, an animated explainer) has `generation: {"template": "draw_svg"}` and an
  intent saying what it shows: `plan` does not validate it and `render` skips it. Draw it with `draw_svg` (`duration_s` and SVG
  animation elements for motion) at the production's format, then record it with `record_take`.
- Keep a voice description plain: age, gender, pitch and pace. Accents and moods ("a smile in her voice") are where a generated voice tends to sound artificial; add them only when the user asks. A new voice says the same lines at another length: speak them again before you trim anything.
- End on an `end_card` (a `caption` is too small for it), after the last words, with a moment held after them. A title read aloud does not also need its words as a subtitle.
- The words the user dictated for a title or a card are copied letter for letter, and `edit` lists under `check_spelling` the words of your text that look like a mistyped word of the user's: correct them before you assemble. Look at the frame of each title in the export too, and ask of it whether any letter touches or passes the edge of the frame or its border.
- Report the cut's length against the one the brief asks for, in your own words, whenever they differ.
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
- Mark the structure, in a music video too: a title over the opening shot, a chapter card or a lower third where a new place, time or person appears, and
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
- The edit is the included shots in order with their chosen takes, trims (`trim` [start_s, end_s]), speeds (`speed`, 0.25 to 4: the take's
  sound follows; it leaves a shot on the song off its beat) and transitions (`transition` {type: cut|crossfade|fade_black|fade_white,
  duration_s}); its sound and text are set with `edit`. A shot with `sound` false has its sound on a track the user made: keep that track
  when you rewrite the tracks.
- A finished production has sound under its whole length. Unless the brief asks otherwise, a film, ad, trailer or story gets a score
  (an instrumental `gen_song` as long as the cut, about -12 dB under speech), a title and an end card, and spoken words get subtitles.
- Before `assemble`, read `production(section="timeline")`: its `warnings` are what the export would report, while fixing them costs no version.
- A shot plays its whole take unless the pace asks otherwise: a prompt's action needs its time, and the brief's length is a target, not worth cutting every shot for. When the total differs, say so.
- A change of song or of voice changes the cut: plan the affected shots again from the new song or line rather than squeezing the old takes into it. A stub under 2 s belongs only in a fast section: drop or merge a shot before you cut it that short.
- One request, one version: make all of a request's changes and check them, then `assemble` once. Every `assemble` is a version the user has to sort through, so do not assemble to test, and say why when a fix needs another one. A warning `assemble` gives is work to do, not news to report: fix it and assemble again before you report. A score from before you added shots stops short: `assemble` warns, and you lengthen it in the same request.
- Say only what you checked. `assemble` lists under `warnings` what is wrong with the delivered file, such as a silent end: fix it (a song that lasts as long as the cut, a score on the tail) or tell the user plainly, and never describe sound you did not measure.
- After a change of cast, tell the user which shots still show the old person or look, and whether you are re-shooting them. Bring the member's `role` and `description` up to date with `cast` when the look or voice changed.
- A version in another format keeps everything but the frame: the same shots, words, cast, voices and music, each shot made again with its composition rethought for the new frame. A card drawn for the old format is drawn again from the same SVG, with its colours, typefaces and layout kept and only the canvas changed; compare the new card with the old one, and with a card that has no drawing of its own, `record_take(card=...)` in the production's text design follows by itself.
- A note that arrives while a shot renders does not stop it: let it finish unless the note is about that shot, and do not shoot it again for nothing.
- `assemble` renders it and reports what went in (length, sound, subtitle cues with their times, text, tracks). It levels the sound and measures the
  delivered file (true peak under -1 dBTP): never re-master, re-mux or re-encode its export with other tools. A file made outside
  the production is not a version the user sees on Deliver; change the edit (`edit` for sound and text, `update_shot` fields `trim`, `speed` and `transition` for a shot) and assemble again instead. Check the picture and the sound of the
  result; transcribe single takes or lines, not the scored mix (music confuses the transcriber), and do not parse the file's boxes.

## 8. Fix what is wrong

A first version is not the end of the work. When you find a fault, or the user points at one (shots, a moment of the video, a sentence), fix that and keep the rest:

- What the user asks for is theirs to decide: a new length, other content, another framing, a shot inserted between two others, a shot left out. Do it as asked. A new shot matches the shots around it: the same cast, look and kind of template. A shot that changes length moves what follows in the cut, and on a song it leaves the beat: say so first.
- Find where it is: `production(section="timeline")` says which shot plays when, so "the first 15 seconds" names its shots.
- Find the cause before you shoot. Words that do not match: `check_song` for a song, `transcribe_media` of the take for a spoken line. Lips in doubt: `wangp_toolbox` `check_lip_sync` measures one shot in seconds (`voice_id`: the song's vocal stem, `voice_start_s`: where the shot starts in it); `may_not_follow` is a suspicion for your eyes, not a fault. Lips that do not follow a song while its slice is right: the face is too small or moves too fast, so shoot it closer and steadier. A subtitle, a trim, a card or a level is fixed in the edit, without a new take.
- Make the smallest change that solves it: a trim, another take of the shot, a subtitle, one new take. Never render a shot the fault does not touch, and keep what is locked.
- On a cut that plays the song, the song is the clock: a shot inserted or made longer covers what comes next, and later shots keep their place and their lips (`on_the_song` in the result says what moved). So the film does not get longer. A shot that sings longer needs a longer `audio_slice` and a new take. A shot made shorter leaves a gap that a singing shot cannot fill, as it has no picture for the song before its slice: `song_drift` reports it, and that shot is shot again on a slice that starts earlier.
- On a film, each shot's sound, line tracks, cards and subtitles follow the shots. A score stays where it starts: its hits no longer land on the same pictures, and a longer film can outlast it.
- When the cause reaches other shots, say so before you act: a new song means every singing shot again. Say how many shots and about how long.
- Then assemble a new version, check that the fault is gone, and say in a line or two what changed. Earlier versions stay: the user can compare them. When the task says the user wants no new version, stop after the verdicts and say how the takes came out: the user assembles when they are ready.
- Two new takes at most for one fault. If it is still there, stop: keep the best take, say plainly what the model cannot do here, and offer what the user can choose (another take, another framing, another line, leaving the shot out). Do not change other things to make up for it.
- When the user names shots to fix, the tools keep everything else as it is for that task. If the fault needs more, end your turn saying what you would change and what it costs; their answer starts a task that can do it.
- A change the user asks for as a whole (another actor, a new song, a new style, another structure) is not a fix: do all of it. Say first what it renders again.

A deep review is asked by the user after a version, and changes nothing. `deep_review` measures words, subtitles, voices, lips, sound and picture and records its findings on the version; you add what you see by comparing frames with the cast and place pictures (`deep_review(add=[...])`), then tell the user what matters most. A **fault** is measured; a **check** is a suspicion the user judges by eye, so do not present it as a fact.

## 9. Keep the user informed

- `log` one short line for each decision the user should know about: a changed approach, a limitation, a trade-off.
- At the end of a task, say in two or three sentences what changed and what is left for the user to review.
- The user can pause you while you work: `render` and `speak` then return early with `paused`. Stop there and end your turn with one
  line saying where you stopped; the task resumes later in this same conversation.
