# Showrunner

Showrunner turns a brief into a finished production: a music video, a short film, an ad, a trailer or an illustrated story. You say what
you want and drop your material; Deepy is its showrunner. It plans every shot, casts, shoots, inspects its own takes and edits, while
you follow, review and change anything on a storyboard and a timeline.

Showrunner comes enabled with WanGP. Open **Showrunner** in the app bar; on first use, choose **Keep It** in its introduction. If you removed it, restore it from **Available Locally** in **Apps & PlugIn Manager**. **Full Screen** at the top of the tab expands the editor in place; **WanGP** returns to the Showrunner tab with your production and view preserved. Open `/showrunner/` on the WanGP address to go directly to the fullscreen editor. Full screen keeps this folder URL, which you can bookmark or share. In the tab as in full screen, the address also names the production and the page you have open: a reload or a copied link opens them again.

Showrunner needs **Deepy Prime**, with a local model or a remote agent (Configuration > Prompt Enhancer / Deepy): running a production takes a planner that
can follow a long plan. On first launch, WanGP sets it up on the Qwen3.8 version that suits your NVIDIA GPU, with nothing to configure. Everything you do yourself (writing shots, rendering them, reviewing, editing, assembling) works without Deepy: see Working Without the Showrunner.

With a local model, Showrunner tells you when Deepy's settings hold a production back:

- **Context Window Tokens** below 64K, when your GPU has room for more: Deepy keeps condensing the conversation and loses details.
- A **KV Cache Quantization** of INT8 or BF16, when Q4 or lower is available: Q4 fits that context in less VRAM.
- A Qwen3.8 version too heavy for your GPU: part of the model may not fit in VRAM, and Deepy can become very slow.
- A **Speculative Decoding** choice that does not suit your GPU: one that pushes the model out of VRAM, or **Disabled** when Deepy could run about twice as fast with it. **Auto** chooses for you.

Deepy's own chat warns about the last two as well.

**Guide** at the top of every page opens this guide at the part about the page you are on (or press **?**).

## Quick Start

1. On the home page, pick a format (**Music video**, **Short film**, **Ad**…) and name the production.
2. In the **Brief**, write your idea in a few words, drop the song (music videos) and a photo of each person who should appear.
3. Press **Plan with the showrunner**. It writes the shots and stops so that you can look at the plan on the **Board**.
4. Change what you like (words, duration, camera, who is in a shot), then press **Shoot N shots**.
5. The showrunner shoots every shot, checks each take, redoes the failed ones and assembles a first cut: watch it in **Deliver**. (With
   **Review the takes before the edit** on in the brief, it waits for you to approve the takes before it assembles.)
6. Review the takes on the Board (**Approve**, or **Redo** with a note saying what should change), ask for changes in the showrunner's
   panel, and press **Assemble** for each new version.

You can write to the showrunner at any time in its panel ("faster cuts in the chorus", "make shot 3 at night"), and undo any change, yours
or its, with Ctrl+Z.

## Starting a Production

On the home page, pick a format and name the production (you can rename it later by clicking its title at the top of the page). Each
format changes how the showrunner plans and cuts:

- **Music video**: shots follow the song's sections and bars, with the singer lip-synced to the part of the song each shot covers.
- **Short film** and **Trailer**: scenes with recurring characters, dialogue and narration; a script you supply keeps its exact words. The plan is checked against the spoken lines of the script, written as `MIA: the words` or with the name in capitals on the line above.
- **Ad**: a short product spot with a clear message and a closing card.
- **Illustrated story**: picture-book pages, each a still picture held while the narrator reads it, with the same characters throughout;
  the edit moves slowly over each page and dissolves to the next.
- **Viggle Remake**: change an existing video instead of making a new one. See [Remaking a Video](#remaking-a-video).
- **Anything**: start from an idea.

The production opens on its **Brief**, which saves itself as you type:

- **Your idea**: a few words are enough (the showrunner develops them); a detailed concept or a script is kept as you wrote it. For a
  music video, **Lyrics** takes the song's words (optional): the subtitles and the lip-synced shots follow them, instead of words the
  showrunner has to make out from the recording.
- **The song** (music videos) or a **soundtrack**: drop an audio file, or a video to use its sound. No song yet? Leave it empty: the
  showrunner composes one from your idea (and your lyrics, when you write them). Showrunner listens to the song and shows its
  waveform, tempo and parts (intro, verses, choruses, bridge). The showrunner cuts on these parts and bars.
  **Change the voice** on the song card has the song sung in another voice, your own song or one the showrunner composed (song
  models cannot sing in a voice you give them): a cast member's (the voice sample on their card, or one made from their voice
  description) or a voice sample from your computer. **Two singers (a duet)** gives each of the song's two voices its own, in the
  order they are first heard: pick two cast members, or voice samples you dropped in the media bin. The music, the words and their
  timing stay, so the shots already made keep their lip sync; the former version waits in the bin. Asking the showrunner in the chat
  does the same. With no song, **Sung by** under the song field chooses whose voice sings the song the showrunner composes (a cast
  member with a voice, or two for a duet): it composes the song, then has it sung in that voice using SeedVC. Both actions require
  **SeedVC Voice Replacement** set to **v1.0 Singing** in **Configuration > Extensions** (the default). Speech modes cannot be used
  for changing a song's singer.
- **References**: pictures or videos of looks, places, moods or movements.
- **Cast**: drop a photo of each person or recurring object so that they look the same in every shot, and describe how each speaker
  sounds (or drop a recording of their voice) so that they sound the same in every line (see **The Cast**).
- **Look and format**: aspect ratio, resolution, total duration from 5 seconds to ten minutes (drag the slider, or type the seconds in
  the field next to it; or **Match the song**), pacing from calm to frenetic, **Shot length** and style tags. Leave **Shot length** empty and the showrunner
  chooses, with most shots lasting about 4 to 7 seconds. Give a range to set it yourself, such as 8 to 15 seconds for long takes (or only one end of it): long shots take longer to render, and a shot lasts at most what its model makes in one go (about 15 seconds with MiniMax H3).
- **Voices** (not for music videos and illustrated stories): how the lines get their sound.
  - **Recorded first** (the default) records each line in its speaker's voice, and the video follows the recording. A speaker keeps one voice for the whole production, which is what long dialogue needs.
  - **From the video model** lets the model say the lines itself, with its own sound: nothing is recorded, and no speech model is needed. Choose the language of the lines beside it, or simply ask the showrunner in the chat ("use the video model's own voices"): it switches this choice and tells you. It suits short pieces and single speakers: a speaker's voice and level can change from shot to shot (**Even voices** on the Timeline evens out the level in the export), and a line's subtitle stays up for its whole shot. Give a shot about 0.4 seconds a word plus a second, no more: with time left, the model tends to add words of its own. Each take is then listened to. Words before the first line and after the last one are cut, with the sound fading out, while a shot that adds nothing plays to its end; turn off **Cut words that are not in these lines**, under the shot's dialogue, for a shot that should run on, and a trim you set yourself always stays. A line the take got wrong, or words added between two lines, are shown on the shot: **Reshoot** tries another take. This only reads words, so a mumble or a laugh stays, and a cut can leave a sliver of a word. The MiniMax H3 video models can do this. When none of your templates can, lines are recorded first, and the Brief says so.
- **Review the plan before shooting** (on by default): the showrunner stops after planning so that you can edit the storyboard before any
  video is generated. Turn it off to let it plan and shoot in one go.
- **Review the takes before the edit** (off by default): the showrunner shoots and checks the shots, then waits for you instead of
  assembling a cut on its own. It assembles only once you have approved the take of every shot in the edit; the Board shows **Waiting
  for your review** with the number of shots left to approve. **Assemble** on the Board, the Timeline or the showrunner's panel still
  works at any time.
- **Let the showrunner look at my other productions** (on by default): each production has its own showrunner. When you ask for something
  like "make it like my other film" or "the same characters as in Lanterns", it can look at your other productions (their style,
  outline, shots and files) and copy the files it needs, such as a character's pictures or voice, into this production. It never
  changes another production. Turn it off to keep the showrunner to this production alone.
- **GPU budget**: an approximate limit on generation time; Auto lets the showrunner judge.
- **Models**: which of Deepy's templates makes each kind of shot or media in this production.

Press **Plan with the showrunner**.

## Try at low resolution first

Set a **Draft resolution** beside the **Target resolution** in the Brief to try shots quickly, then render the ones you like at full size.

Beside **Shoot** on the Board and in a shot's panel, a **Draft / Final** switch says which size it makes:

- **Draft** renders at the draft size, and the showrunner looks at the take. Use it for looks, framing and timing: lip sync can be poor at a small size, so judge mouths at Final.
- **Final** renders at the target size with the same prompt, seed and pictures as the shot's take. No language model runs and nobody looks at the take, so it is quick to start. It costs at least twice the time of a draft at 480p, and the frames are not identical to the draft's.
- Shoot acts on the shot in its panel, on the shots you selected, or, with none selected, on all that need it.
- **Render now** is for a shot with no take yet: it renders at once from its prompt, with no language model, and nobody looks at the take.
- Pictures (the cast, looks, places, painted frames) are always made at the target size, so what you approve stays the same.
- Both takes stay on the shot, labelled with their size: click one to use it, or compare them.
- The edit is made at the size of the takes it uses. An edit of drafts is a quick draft; if the sizes are mixed it takes the largest, and **Deliver** says how many shots are still drafts.

## Remaking a Video

A **Viggle Remake** recasts the people of an existing video, or changes its objects, setting, weather or light. The motion, camera, timing and soundtrack stay as they were. Its shots are made with the Viggle Remake models: Swap where people change, Edit for the others.

1. Drop the video in the Brief and say what should change. Change only footage you have the rights to, and never put a real person into a video without their agreement.
2. Plan. The showrunner finds the cuts and makes one shot per camera shot of your video, looks at each one, and decides from your request whether it changes and how many painted frames it takes. Before painting a frame it looks at who is in it and what each person is doing, and afterwards it checks the picture against that, painting it again when someone's size, place or expression was lost. A shot with nothing to change keeps your video's own frames, with no render. A shot too long for one render continues by itself. Splitting loses no quality and takes almost no space: each shot reads its part of your file, which is encoded again only if its frame rate differs from the model's.
3. Review the plan. Each card on the Board shows its moment of your video, or its new look once a frame is painted, and plays that part of your video under the mouse, after a moment the first time. Depending on the model, each shot has one or more **painted frames**, frames of the source showing the new look, and a prompt describing the result. They decide how the new look comes out, so this is the moment to look at them: open a shot to watch its clip and compare each painted frame with the source frame it came from. To paint one yourself, pick a frame number under **Painted frames**, download it with **Source frame**, change it in any image editor and bring it back with **Use my picture…**: it is fitted to the shot's size and takes the place of a painted frame less than 72 frames from it, and the showrunner keeps it as it is. Frames next to a cut cannot be painted: they take the look of the other shot. Once a shot has takes, the **Source** tile beside them plays your video in the same place; right-click it to compare it with the take.
4. Shoot. The showrunner measures each take against the source, for cuts that moved or disappeared and black bands at the edges, and looks at it.
5. Assemble as usual, on the Deliver page. Your video's own soundtrack plays over the shots, at its own level. Remove a shot and its sound goes with it.

**Shots rendered together**, beside your video in the Brief, joins short shots with their neighbours in one render (at least 8 seconds by default): rendered alone, a short shot's lips barely follow the speech or the singing. Each shot still gets its own take. When you render one shot again, its neighbours render with it and each gets a new take beside the one it uses, so nothing you chose changes; the shot's takes say what they were rendered with. Select shots that follow each other to render them together, up to the model's longest render (about 14 seconds): the panel says which shots render together, and **Reshoot N shots** on the toolbar, like **Reshoot them** in the panel, gives each a new take, at the size **Draft** / **Final** says. While shots render, the Showrunner button at the top says which ones and how far they are. Set it to 0, or turn off **Group rendering** on the Board's toolbar, to render every shot alone.

Under **Models**, the **Frame Painter** paints the frames where people are replaced. Its Auto is H3-Image: it keeps each person's place, pose and expression, and handles several people in one frame, including someone who stays as they are. When a person becomes a character of another kind, such as a teddy bear, it sometimes paints only the head on the original body; the showrunner checks each painted frame and paints it again. Viggle Image Swap, also in the list, gives each new person the body of their character sheet rather than the original's, but loses moving poses and expressions, and dresses everyone in their sheet's outfit. The **Image Editor** paints the frames of other changes and makes the character sheets. Choose a **Draft resolution** of 480p in the Brief for a fast first version: a two-minute video with many cuts takes hours at 720p on one graphics card.

A shot with nothing to change is not rendered at any size: the edit reads it directly from your video.

A video from 24 to 30 frames per second, such as 23.976, 25 or 29.97, keeps all its frames and its rate, and each shot delivers exactly its own frames. A faster one or one with a variable rate is converted to the model's 24 frames per second, which drops frames and can show on fast pans.

A remake works best on videos with cuts and faces large in the picture. Small faces, many fast cuts in a few seconds and non-human characters in long takes come out weaker.

## The Cast

The **Cast** page holds the people, recurring things and places of the production (a singer, a villain, the red car, the studio), each on a card:

- **Pictures**: drop photos on the card. The highlighted one is the **reference picture** of every shot the member appears in; click
  another to use it instead. **Ask the showrunner for master images** makes a clean front-facing picture for members without one, and
  **New picture** asks for a new one from the description, after a recast or when the picture no longer matches it.
  When your photos do not match the production's style (a photo in a watercolor story), the showrunner draws the reference picture
  in that style from them, so the person keeps their face and clothes and every shot keeps the style; your photos stay on the card.
- **Name, role and description**: what must stay the same in every shot (age, look, clothes).
- **Looks**: a change of costume (or of age, or a wet or wounded state) is a look of the same person: its own picture, made from the
  person's reference so that the face stays, worn in the sequences the card lists.
- **Voice**: describe the voice (age, timbre, accent, pace), or drop a recording. Each speaker, the narrator included, has one voice
  sample, made once from the description (or your recording), and every line they say is spoken from it. Remove the sample to have the
  showrunner make a new voice from the description; every line is then spoken again.
- **Lock** keeps the member exactly as it is: the showrunner never changes a locked member.
- **LoRA** (only with **LoRAs on cast cards** on, in **Settings** at the top of every page): a LoRA you trained for this person or thing,
  its strength and its trigger word. A LoRA only works with the family of models it was trained on, so the card lists those of the
  models this production uses and says how many of the member's shots can use it; the other shots are made without it. Shots already
  made do not change: **Reshoot** or **Fix…** uses it. Turned off in Settings, no shot uses a LoRA from its card.

Every shot that shows a cast member is generated from that member's reference picture, so faces, hair and clothes stay the same over
dozens of shots. Dropping a photo anywhere on the Cast page adds a new member.

**Voices** are measured against their card. A voice that sounds like the other sex than the card says is made again, a spoken line that moves far from its speaker's voice is spoken again without its delivery note, and a song whose lead voice does not fit the singer on screen is reported before shooting. The showrunner tells you when it cannot fix one.

**Places** keep a location the same from shot to shot, which words in a prompt cannot do: each shot would invent its own set. Press **Add a place**, and give its card one picture of the set as the viewer should see it, without the cast and without writing, but with the people who belong there, such as an audience in its seats: your own photo, or one from **Ask the showrunner for master images**. Every shot that lists the place gets its picture as a reference, close shots and shots without people included. A side the picture does not show, such as the seats seen from the stage, gets its own place card: only one picture of a card goes to a shot. The showrunner adds a place by itself when it plans a location that recurs, and you can ask for one at any time: "the kitchen looks different in every shot".

## Long Productions

A production can last several minutes: a whole song, a short film in chapters, a documentary. The showrunner then works in sequences
(the chapters of the story or the parts of the song): it writes the outline first, makes a reference picture of every recurring
person, plans each sequence's shots, and shoots sequence by sequence, assembling the edit after each one so that you can watch it grow.

- **The same people and voices from start to finish**: every shot of a cast member comes from their reference picture (or the picture
  of the look they wear), and every line from their voice sample (see **The Cast**).
- **Structure you can see**: the storyboard groups the shots by sequence, with each sequence's length and progress.
- **Titles and cards**: the showrunner titles the film, introduces people and places with lower thirds, separates chapters with cards
  and ends on an end card (see the timeline's text lane). It gives the production a text design that suits it, and a chapter card can
  show a dimmed still of the scene it opens. A score runs under the whole cut, with narration or dialogue over it.

## Reviewing and Changing the Plan

The steps at the top of the page show how far the production has got, and each one opens what it is about: **Brief**, **Edit** (the Timeline) and **Deliver** their pages, and on the Board, **Plan** all the shots, **Review** the takes to review and **Shoot** the shots still to shoot. The step you are looking at has a colored outline.

The **Board** (storyboard) shows every shot as a card, grouped by the parts of the song or the film. Cards show the shot's picture once it
is shot (hover to play it), its status and the showrunner's opinion of the take.

- **Select** a shot to open it on the right: watch its takes, read the showrunner's note, and edit what happens, its dialogue, the prompt,
  the duration, the camera, its model, who is in it, the transition into it, its trim and your notes for the showrunner. Edits save
  themselves as you type.
- **Dialogue**: the words spoken or sung in a shot, line by line, with who says them and an optional delivery note (whispered,
  angry…). The showrunner fills them from your script and speaks each line in its speaker's voice. A line keeps its recording until
  you change its words, its note or its speaker's voice; the showrunner then speaks it again before re-shooting. The card shows the
  first line.
- **Reorder** shots by dragging the cards; a shot dropped among another sequence's shots joins that sequence. Several shots can be
  selected with Ctrl or Shift and moved together; Shift selects only the shots the current filter shows.
- **Lock** a shot (or a cast member) to keep it exactly as it is: the showrunner never changes it.
- **Exclude** a shot to keep it out of the edit without deleting it: its card stays on the Board with a crossed-out eye, and the **Excluded** filter lists these shots.
- **Drop media** from your computer onto a shot to use it as a take, or onto the cast to add reference pictures and voices. A sound dropped
  on a shot never replaces its picture: it plays over the edit from that shot on, as a track on the Timeline's Sound lane, where you
  shorten, move or fade it. Media you drop elsewhere, and media the showrunner made outside the shots (a song it composed, for one), wait in the
  **Media bin** at the bottom: click one to play or view it (in a picture, scroll or use + and − to zoom, hold and drag to move inside it),
  drag it where you need it. Without a mouse, select a shot and right-click (or long-press) the item instead: a picture or a video becomes a
  take of that shot, a sound plays from it on. The **×** on an item takes it out of the bin: its file stays in the production's folder, and
  undo brings it back. A deleted shot leaves its takes in the bin, so a clip you removed can come back later.
- **Your own clips as shots**: drop a video or a picture on **Add a shot** at the end of the Board, or right-click a bin item → **New shot
  from it** (after the selected shot): it becomes a shot of its own, named after the file, with the file as its take. It plays in the
  Timeline and the edit like any other shot, and the showrunner works with it too. A shot without a take also offers **Use my video…** in its
  panel, beside **Render now**.
- **Start and end pictures**: a shot whose video starts (or ends) on a picture shows it on its card until it has a take, and in its panel
  under **Start picture**; click it to look at it closely.
- **Undo and redo** (Ctrl+Z, Ctrl+Shift+Z) cover every change, yours and the showrunner's, including a new take; a message says what was
  undone.

When the plan suits you, press **Shoot N shots**: the showrunner shoots the shots without a take and the ones marked for a redo, never
the others. Once every shot has a take, the same button becomes **Assemble**. **Shoot this shot** in the panel of a shot without a take shoots
that shot alone and stops there, to see it before the rest. **Render now**, beside it, renders the shot without the showrunner (see Working Without the Showrunner).

The line under **Storyboard** gives the picture size of the shots: the real size of their takes, and for a shot not shot yet what its model makes of the
brief's resolution (a model works on a grid of pixels, so 1280×720 can come out at 1280×704). Change it in the brief. Point at it for the details.

Next to the button, the Board says roughly how long generating those shots takes ("about 12 min"). The estimate comes from the time the
same models really took for earlier takes at the production's picture size: this production's takes first, then your other
productions'. Before any take of a model was timed, it says that the estimate comes after the first take. The showrunner's checks, voices
and retakes add to it. **Shoot** in the showrunner's panel shows the same estimate when you point at it.

## Takes and Verdicts

The showrunner renders each shot, inspects the result with its own tools (picture, motion, cuts inside a shot, black frames, sound) and
records its verdict: a good take counts as shot, with the showrunner's badge on its card (**To review** when takes wait for your review before the edit), a failed one is marked **Redo** and shot again, at most twice:
then the best take waits for you with a note on what is still wrong. A card's badge says whose verdict it is (**Showrunner** or
**You**).

Only you approve a shot, and you need to only when takes wait for your review before the edit. In the shot panel:

- **Approve** marks the take as final; the showrunner leaves approved shots alone. Its own opinion of the take stays in the panel under
  yours, so a note on what it found wrong is not lost when you approve or redo the take. Click it again, or press **A** again, to take
  your approval back: the take waits for your review once more, with the showrunner's opinion of it.
- **Fix…** is where you say what is wrong or what should change: the showrunner fixes this shot and keeps the rest (see Fixing What Is Wrong).
- **Redo…**, when takes wait for your review before the edit, marks a take for a new one with your note, so that you can go on reviewing: **Shoot** then takes every shot marked for a redo.
- Click another take to use it in the edit. Right-click a take to **compare** it with the one in use: a split view with a handle to slide,
  played in sync; **Use B** keeps the other take, **Done** closes the comparison.
- **Reshoot** renders another take at once from the shot's settings and prompt, without the showrunner and without any note: for a take that only needs another try, or after you edited the prompt yourself.
- **+ Shot** adds a shot after the selected one, in its part (at the end when none is selected); drag its card to move it.
- After changing someone on the **Cast** page, **Re-shoot N** on their card shoots again the shots showing them, with their new look and voice.

## Working Without the Showrunner

You can make a production by hand, with no language model at all: to keep every decision yourself, or when your computer cannot run Deepy Prime.

1. In the **Brief**, choose the format and the **Models**. On the **Cast** page, drop a photo of each person and describe their voice.
2. On the **Board**, press **+ Shot** for each shot, then write **What happens**, its camera and duration, who is in it and its dialogue.
3. Press **Render now** in the shot's panel, or right-click it, or select several shots and press it in the panel. Without Deepy Prime, the Board's main button renders every shot still to shoot.
4. Check each take yourself, use **Reshoot** for another try, then press **Assemble**.

A shot without a prompt gets one made from what you wrote (what happens, the camera, who is in it, the brief's look), word for word: open **Prompt** in the shot's panel to change it. A shot without a chosen model takes the brief's for what it shows. Lines of dialogue are recorded first in their speakers' voices, as when the showrunner shoots, unless **Voices** in the Brief is on **From the video model**: then they join the prompt when the shot renders, and a shot without a chosen model takes one that can say them. Nobody checks these takes, and nothing is retried or assembled by itself.

## The Timeline

The **Timeline** plays the edit in the browser, at once, with no rendering: the chosen takes back to back with their trims and transitions,
over the song. Shots that are not shot yet appear as cards in their place, so the pacing of the plan can be judged before any generation.

- The lanes show the parts, the shots and the song's waveform with its beats and bars. The Parts lane names the parts as the Board
  does (each shot's section); before the plan, it shows the parts heard in the song. The Sound lane also shows the tracks laid
  over the edit (narration, score, your own audio): drag one to move it, drag its edges to play less or more of its file, click it to
  hear it alone and set its start, the part of the file it plays, its fades and its level beside the monitor (**Delete** removes it from
  the edit, **Escape** closes it). Tracks play in the preview as they will in the export, whichever soundtrack is chosen. Texts and tracks
  follow the shots they start on when shots are added, removed, moved or retimed.
- The soundtrack of the preview and of the export is chosen above the lanes: **Song**, **Shot sound** (dialogue, effects) or **Mix**.
  In a mix, the song plays at half its level. When **Voices** in the Brief is on **From the video model**, **Even voices** brings every shot with lines to one level in the export, as some come out much quieter than others: turn it on, then **Assemble**. A very quiet shot gets louder along with its background noise, and the preview still plays the shots as they were made. **+ Audio**, next to **+ Text**, adds your own audio track (it starts at 0 s: drag it into place). The **Edit** card beside the
  monitor sets where the song starts; in a narrow window it is hidden, and the card beside the monitor only appears while a track is
  selected.
- Click a shot to see it in the monitor; double-click it to open it in the shot panel. Drag a shot to move it; drag its edges to trim it
  (or to change the planned duration of a shot not shot yet). With **Snap**, edges stick to the beats and bars and to the other shots'
  edges. **Fit** shows the whole edit, however long.
- **Speed**, in the shot panel, plays a take faster or slower (0.25× to 4×) and changes its length in the edit; what follows it moves
  with it. Its sound follows at the same pitch. Slow motion repeats frames, so strong slow motion can look jerky. A shot made on its part
  of the song leaves the song at another speed: its lips and its cuts no longer match, and the panel says so.
- **Detach Sound**, in the shot panel or by right-clicking a shot, makes the shot's sound a track of its own on the Sound lane: shorten it,
  extend it over the next shot or start it before its own (up to what the take holds), move it or fade it, apart from the picture.
  **Reattach** removes that track, and the shot plays its take's sound again. A new take keeps the old sound on the track until you
  reattach it; the panel says when the track comes from another take.
- Click the marker above the cut between two shots to choose the transition (cut, dissolve, fade through black or white) and its length.
  When the song drives the edit, every shot stays on its part of the song: a dissolve plays over the extra end of the outgoing take,
  so it lasts at most as long as that take runs past its shot (and becomes a cut when it does not).
- When the picture ends before the song, the edit panel says by how much.
- The **Words** lane shows each spoken line where its subtitle sits, with its speaker in the tooltip, so that you can check the lines without playing the sound.
- The **Text** lane holds titles, lower thirds (a name and a role), captions and full-frame chapter or end cards. **+ Text** adds one at
  the playhead: choose its style and its design (the production's, or another), write its words and its time on screen. Drag it to move
  it, drag its edges to change how long it stays, double-click it (or select it and press Enter) to edit it, Delete to remove it. Texts
  that overlap sit on two rows. It plays over the preview as it will in the export, fading in and out; an end card that reaches the end of the cut only fades in, and stays to the last frame. To change how long one of the showrunner's texts takes to fade, ask it ("let the end card fade in over 8 s"). Yellow texts are yours, purple
  ones the showrunner's: the showrunner never changes yours. Text is drawn, never asked of a video model, so it is always sharp, and it reads
  exactly as it was typed (read what the showrunner typed: double-click a text to correct it). A full-frame card hides the picture under it: after moving shots, check that a card does not cover a moment you want
  to see.
- Choose the soundtrack: the **Song**, the **Shot sound** of each take, or a **Mix** of both.

## Models

Showrunner generates with Deepy's templates, in the same categories as Deepy's settings:

| Deepy category | What the showrunner makes with it |
| --- | --- |
| **Image Generator** | Stills, illustrated pages and backgrounds. |
| **Image Editor** | The cast's reference pictures and looks, made from their photos. |
| **Frame Painter** (remakes, an Image Editor template) | A remake's frames where people are replaced by the cast, painted before the video. |
| **Video Generator** | Shots without the cast: landscapes, objects, crowds, inserts. |
| **Video With Speech** | Shots where someone sings or speaks, lip-synced to the song or to their line. |
| **Video Generator with Ref.** | Shots with the cast, keeping each person's face, hair and outfit from their pictures. |
| **Song Generator** | The songs and scores the showrunner composes. |
| **Speech From Description** | Each speaker's voice, made once from the voice described on their cast card. |
| **Speech From Sample** | Every spoken line (dialogue, narration), in its speaker's voice. |

Each category lists the same templates as Deepy's settings: the built-in ones and the one you added there. When only some templates of
a category can do it, **(keeps faces)** marks those that take the cast's pictures and **(lip sync)** those that follow the song or a
spoken line.

- **Auto** uses the template selected in Deepy's settings for the category; its name is shown next to Auto. The Frame Painter's Auto is
  H3-Image, which replaced people more reliably than the other image editors in our tests, keeping their poses and expressions.
- **When a template cannot do what a shot needs**, the showrunner takes the one of another category. For example, when the Video With
  Speech template cannot keep the cast's faces, the singers and speakers of the cast are shot with the Video Generator with Ref.
  template, which follows the song or the line as its soundtrack (when its model can, as MiniMax H3 Ref2VA can).
- **Your own templates**: save your settings in WanGP's generator as a preset (the list at its top, beside **Save**), keep it selected, then press **+** next to
  the category in **Ask Deepy > Settings**. The template then appears in Showrunner's lists. A template you picked in Showrunner stays
  available there even after Deepy's settings move to another one.
- **The brief's Models** sets the template of each category for this production. Picking a template applies it to the shots of that
  category that still use the previous one, and the showrunner uses it for every new shot of that category.
- **Model** in the shot panel switches one shot to another template, for example a template that lip-syncs. It lists the templates
  as Default models does, under their categories, and marks the brief's choice.
- **Default models**, at the top of every page, sets what new productions start with. A production keeps its own choices once created;
  change them in its brief.

### Your prompt rules for a model

The notes button beside each list (in the brief's Models, in Default models and next to **Model** in the shot panel) holds your rules for
the prompts the showrunner writes with that template; for **Auto**, the template Auto stands for. The showrunner follows them in every prompt
it writes with the template, in all your productions: shots, pictures, songs or voices, depending on the category. The button is lit when a
template has rules.

- **What to write**: what the model needs that the showrunner may not know, the words to use or avoid, a style to keep, or the trigger word
  of a LoRA in your template. For example: "One action per shot, in the present tense. No text or logos on screen."
- **They are instructions**: the showrunner follows them as it writes, which is not a guarantee. What you ask it in the conversation comes
  first.
- **Per template**: two templates of the same model have their own rules, so a template with a style LoRA can carry its trigger word. The
  rules stay on this computer with your Showrunner settings: an exported production does not take them along, and a template you rename
  in WanGP's presets needs its rules again.
- **Leave the box empty** to remove a template's rules. The showrunner uses changed rules from its next task.

## Talking to the Showrunner

When the showrunner finishes a task and something is up to you, the **Showrunner** button at the top of every page becomes **Your turn** and
says what ("Review the plan", "Review 5 takes", "Watch the video"); clicking it opens or folds the panel. A bar above the media bin says it in full: its button
does it and closes the bar (**Shoot**, **Assemble**, **Open Board**), and when there is nothing to press, its × closes it. "Stopped" means that its task ended early: the
Showrunner tab says why, and **Continue** picks it up. The note only shows while the page is open: if you were
elsewhere, it is there when you come back. To hear about it sooner, turn on **Tell me when a task ends** in **Settings**: when the
showrunner finishes a task or stops on an error, WanGP plays its notification sound on the computer it runs on and sends what comes next to
your notification destinations (**Configuration > Notifications**), if you have set any.

The panel on the right has two tabs:

- **Shot**: the selected shot and its takes.
- **Showrunner**: your conversation with it, the newest at the bottom, with each step of its task as it goes and what it is doing now above the input.

To follow its work more closely, turn on **Show the Deepy tab** in **Settings**: a third tab shows Deepy's own window on this production, with its thoughts and every step as they are written. It stays at the latest words unless you scroll up. You read it there, and write in the Showrunner tab.

The panel starts open on a large screen and folded on a small one, and then stays as you left it: its arrow or the D key fold it to a strip of
icons, which also shows when the showrunner is working or waiting for you. While it is folded, double-click a shot to open it, or click an icon.

In the Showrunner tab, write in plain language ("faster cuts in the chorus", "make shot 3 at night"): while it works, your message reaches it at
its next step; otherwise it starts working on it. When shots are selected, your message is about them. The row under the input holds what you can
do at that moment: **Shoot**, **Assemble** and **Re-plan**, or **Pause** and **Stop** while it works. **Stop** interrupts it; it continues where
it stopped when you ask again.
When a task ends, the panel says what the showrunner changed (shots added, rewritten or removed, takes shot and judged, text, sound,
new versions) and the shots it touched glow for a moment on the board. Shots appear as "shot 4" in its messages, in the order of the cut.
During a long wait, the line above the input tells a slow step from a stuck one: what Deepy is doing now and how many tokens it is reading before it
can answer. Rest the pointer on it for Deepy's speed in tokens per second and how full its context is; the Deepy tab shows them under its window.

Deepy keeps your latest conversations ready in RAM, so that it answers quickly when you come back to a production (**KV Caches Kept in RAM** in Configuration > Prompt Enhancer / Deepy, 2 by default). When this production's is no longer there, Deepy reads the conversation again, which can take up to a minute, and a bar in the Showrunner tab shows how far it has got.

**Pause** stops the showrunner after its current step (the shot or voice it is generating is finished and kept) and frees the GPU for
other work: WanGP's models and Deepy's leave the video memory. **Pause now** also stops what it is generating; that shot is generated
again later. The production shows **Paused**, also after you close the page or restart WanGP. **Resume** continues the same task where
it stopped, in the same conversation with the showrunner, so it remembers what it was doing; a message you write while it is paused
resumes it with your note. Meanwhile you can review, edit and assemble as usual: the showrunner reads your changes when it resumes.
**Stop** gives up the paused task.

The showrunner sometimes gets a call wrong and corrects it by itself; nothing is lost, and these calls stay out of the panel. What really failed is always shown. **Show every error** in **Settings** lists every one, for finding out what went wrong.

While the showrunner works, its conversation takes Deepy's place: your own Deepy chat comes back when it finishes or stops. Everything
else in Showrunner and in WanGP stays available.

## Delivering

**Assemble** renders the edit as one video: the included shots in order with their chosen takes, trims and transitions, the text over
it and the chosen soundtrack. When the takes can simply be joined, they are, with no loss of quality; otherwise the edit is encoded once with your output settings (Configuration > Outputs). Shots without a take are left
out: Assemble says which ones before it starts. The length you set in the Brief is a target: a shot with a spoken line ends a moment after its line, so the cut can come out shorter than planned. The showrunner tells you, and you can ask it to give those shots more room.
Other shots last a length their model really makes (with MiniMax H3: 4.46, 5.17, 5.88 s and so on), so that every action plays to its end; a music video cuts on its song instead, following **Shot length** as far as the song allows. For faster cuts, turn on **Let the showrunner pick any shot length** in **Settings**: a shot then plays the start of its video, and the end of its action can be lost. A take plays for its shot's planned length (generations often come out slightly
longer), so the cuts land where the showrunner planned them and lip-synced shots stay on their part of the song; trim a shot to use
another part of its take.

**Deliver** keeps every version, named by its number and the time it was made ("Version 3 · 10:42"), with its length, picture size
and a download button (the file keeps its name), and a checklist of what is left before the final cut. The showrunner names versions
the same way. When the version you watch carries subtitles, **Subtitles** at the top of the player shows or hides them (the choice is
remembered). When you change the edit after assembling, Deliver says that the last version is out of date, and the
home page marks the production **Changed since delivery**. **Showrunner review**, beside **Assemble** on that page, has the showrunner assemble the edit, watch it and report what it finds; for a closer look, see Deep Review.

**Sound and subtitles**, on the Deliver page, sums up what the export plays and shows besides the picture:

- the sound in one line: the soundtrack (song, shots' sound or both) and the number of tracks laid over the edit. They are placed and
  levelled on the Timeline (**Edit on the Timeline**), where they play against the picture;
- the subtitles the export carries as a track the viewer turns on: the shots' dialogue (spoken lines, narration and sung lyrics)
  becomes subtitles automatically, timed on the cut and in the script's words; or add your own SRT file (it replaces the automatic
  ones). Subtitles are never burnt into the picture.
- the level: the whole soundtrack gets one fixed gain to a delivery loudness, with a limiter on the peaks; quiet moments stay quiet.

To make the video in another format, such as vertical for social networks, use **Make it in another format** on **Deliver**, or choose the ratio in the Brief and press **Make a … version**. The showrunner makes the shots again in the new format, with the same words, cast and music. Choosing a ratio alone changes nothing that is already shot. The versions you have stay as they are.

## Fixing What Is Wrong

You do not have to start over when part of a version is wrong. Point at it, and the showrunner fixes that and keeps the rest.

- **While watching:** in **Deliver**, pause on the problem and press **Something's wrong here**. The shot playing is already picked.
- **On the Board or the Timeline:** select the shots (Ctrl+click, Shift+click) and press **Fix**, or right-click them.
- **In the showrunner's chat:** say it in your own words, such as "the first 15 seconds of subtitles do not match the song".

The window lets you add or remove shots (Shift+click for a run of them) and say what should change:

- **Redo a shot differently:** "make it 6 s and show her tasting the sauce". The length, the content and the framing are yours to change.
- **Insert a shot:** press **+** between two shots, or right-click a shot and choose **Insert a shot after it**, and say what the new shot shows and how long it lasts.
- **Just say what is wrong:** "her coat changes colour". The showrunner looks for the cause and changes as little as it can: a trim, a subtitle, another take, a new take.

With **Keep everything else as it is** on, the showrunner cannot touch anything but those shots. It then makes a new version and tells you what changed. Locked shots are never changed.

**Make a new version when done** is on by default. Turn it off when you fix shot after shot before watching: the showrunner stops once the new takes are judged and makes no version, and you press **Assemble** when you are ready to watch the edit. The window keeps your choice for the next fixes until the page is reloaded. With **Review the takes before the edit** in the brief, a fix always ends on your review instead, so the window does not offer this choice.

To try another take of a shot with the same prompt and a new seed, **Reshoot** on the Board or in the shot's panel renders it at once, without the showrunner and without a new version.

**On a film,** the rest moves along when a shot is inserted or changes length: each shot's sound, the voice-over lines, the cards and the subtitles (also those of an SRT file you added) stay on their pictures. A score keeps playing from where it starts, so its hits may no longer land on the same pictures; you are warned if the film outlasts it.

**On a music video,** the song keeps its timing, so the film does not get longer: a new or longer shot takes its time first from the shots before it where nobody sings, then covers what comes next, like a cutaway while the song carries on, and every later shot stays in sync. A shot that sings for longer needs a new take on a longer part of the song, and a shot made shorter can leave a gap that only a new take of the next one can fill; the showrunner tells you.

**If the fix is worse**, you lose nothing. The next step offers to compare the new version with the previous one, side by side, with the list of what changed, and **Go back to version N** restores the previous one as one change you can undo. On the Board, you can also click a shot's earlier take to use it again.

**When the showrunner reaches a limit**, it says so instead of trying forever: after two new takes it keeps the best one and tells you what you can choose.

**Bigger changes stay open.** Some faults reach further than the shots you see: when a song does not sing a verse as written, only a new song brings the words back, and every singing shot has to be shot again. The showrunner tells you what it would change and what it costs, and your answer lets it go ahead. Turn off **Keep everything else as it is** to let it decide from the start, and ask in the chat for changes to the whole film, such as another actor, a new song or a new style.

The showrunner also checks a song against its lyrics by itself before shooting on it, and composes it again when lines are missing. Subtitles of a music video follow the singing, and a line that is not sung gets no subtitle.

## Deep Review

**Deep Review** in **Deliver** goes further than the checks made while shooting. It takes some minutes, changes nothing, and measures the latest version against its script: what is really said or sung and when, subtitles against the words, voices against their cards, lips against the voice, the sound's level and silences, black or frozen picture. The showrunner then looks at every shot against the pictures of its cast and place.

Each finding gives the moment and the shot. A **Fault** is measured; a **Check** is a suspicion for your eyes, as lips on a small face are hard to measure. Click the time to go there, **Fix…** to open the fix window with that shot and that note, or **Fix these…** for all of them at once: remove what you want to keep before sending. With a Nemotron model chosen in **Configuration > Prompt Enhancer / Deepy > Transcription**, the review also compares who is heard with who the script says, and tells you when two characters sound alike.

## Versions

When two versions are compared, the bar under the picture moves both to any moment, and clicking a line in the list of changes goes straight to that shot. The comparison of two takes has the same bar.

Changing a delivered production, yourself or through the showrunner, keeps working on the same production: each Assemble adds a
numbered version, and every version keeps the state it was made from (shots and their takes, cast, text and sound). A re-shoot does not
make one by itself: when the cut has changed, the **Your turn** button says so and Assemble makes the new version. The showrunner makes one
version per request, and another when a fix needs it, saying why.

- **Compare** on a version shows it beside the one you are watching, in one frame with a handle to slide between them, played together
  (switch the sound between A and B). Below the player, a short list says what changed between them: shots added, removed or reordered,
  a new take, words or timing changed, a recast, a new voice or picture, text moved or changed, and sound tracks.
- **Open** shows what a version used, beside today's production. Each piece the production no longer uses gets a **Bring back** button:
  a shot's take (the other takes stay on the shot), a shot removed since (it returns at its place in the cut), a cast member's pictures,
  looks and voice, a text, a sound track or a subtitle file. Each one is a single change that undo reverses.
- **Restore this version**, in the same view, takes the whole production back to that version. Nothing is deleted: takes made since stay
  on their shots, unused, shots added since stay but leave the edit, and undo brings today's state back.
- **Duplicate from this version** starts a new production from that version, with its media, and leaves this one as it is: useful to
  try another direction without losing the current one.

Versions assembled before this feature only open while the undo history still holds their state.

## Keyboard Shortcuts

| Keys | Where | What it does |
| --- | --- | --- |
| **?** | Everywhere | Opens this guide at the part about the page you are on. |
| **Ctrl+Z**, **Ctrl+Shift+Z** (or **Ctrl+Y**) | In a production | Undo and redo, yours and the showrunner's changes alike. |
| **1** to **5** | In a production | Brief, Board, Timeline, Cast, Deliver. |
| **D** | In a production | Shows or hides the showrunner's panel. |
| **Arrows** (with **Shift** to extend) | Board | Selects the next or previous shot. |
| **A**, **X** | Board | Approves the selected shots (or takes their approval back), or marks them for a redo with a note. |
| **L**, **E**, **Delete** | Board | Locks, excludes or deletes the selected shots. |
| **N** | Board | Adds a shot after the selection (or at the end). |
| **Space**, **Home**, **End** | Timeline | Plays and pauses, goes to the start or the end. |
| **Left**, **Right** (with **Shift**: 1 s) | Timeline | Steps one frame. |
| **+**, **-**, Ctrl + mouse wheel | Timeline | Zooms. |
| **Enter**, **Delete** | Timeline, on a text | Edits or removes the selected text. |
| **Escape** | Everywhere | Closes a window, or clears the selection. |

## Where Productions Are Kept

Each production is a folder holding its document, its media, its exports and the showrunner's session. The folders are under
`settings/showrunner/productions` by default; `settings/showrunner/config.json` can point `productions_root` elsewhere (restart WanGP
after changing it). A production's media also appear in WanGP's gallery under a workspace named after it.

To tidy the home page, point at a production and click **Archive**: it leaves the grid for the folded **Archived** list at the bottom of the
page, with everything kept, and its **Bring back** button returns it. **Delete** moves the production's folder to the `.trash` folder of
the productions root, from where it can still be restored by hand.

To keep a production or move it to another WanGP, open its **…** menu on the home page and choose **Export as .zip**: one file with its
shots, takes, media, versions and exports. **Import a .zip…** above your productions opens such a file as a new production, here or in
another installation. Like a duplicate, the copy starts with a fresh showrunner: its conversation and undo history stay with the original.

---

> Applies to: The bundled Showrunner Plugin (available from the app bar and the normal lists in Plugins) with Deepy Prime as its showrunner.
