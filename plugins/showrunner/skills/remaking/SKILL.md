---
name: remaking
description: "The showrunner's method for a remake in Showrunner: an existing video re-rendered with its people, objects, setting, weather or light changed, keeping its motion, camera, timing and soundtrack."
listed: false
---

# Remaking a video

A remake changes an existing video. Every shot renders from a clip of the source, so the motion, camera, timing and sound come from there. You decide what changes, prepare what the template needs and judge the takes. The rest of `wangp://skills/showrunner/directing` holds, except that shots come from the source's cuts and not from your own plan.

Five actions exist for remakes alone and are not in the toolbox's list: `ingest_source`, `make_sheet`, `look_at_frame`, `paint_frame` and `check_take`. Call them like the others, `wangp_plugins({"plugin": "showrunner", "action": "ingest_source", "arguments": {"production": "<id>", ...}})`; `"arguments": null` reads an action's contract.

Only remake footage the user may change. Do not put a real person into a video without their agreement: if the brief asks for a named real person and does not say they agreed, ask first.

## What changes

Only what the brief names. A title is not a brief.

- **A recast** changes people and characters: faces, bodies, hair, clothes. The set, vehicles, props and light stay as the source has them, and you make no picture of a place. When the brief gives no looks, invent them, each clearly different from the original, and say what you chose in your report.
- **When the brief does not say what should change,** or reads as if it was meant for something else: do not guess and do not paint. Say so and ask, then stop.

## 1. Read the source

- Call `ingest_source` (`change`, from what the brief asks: `people` when people or characters are replaced, `scene` for objects, setting, weather or light). It writes one shot per camera shot of the source. Shots cannot be added, replaced, removed or moved: write each one's prompt and cast with `update_shot`, and never `plan` shots here.
- A shot's `source_range` is what it delivers, in source frames, at any length: WanGP fits it to what the model renders. A shot with `windows` above 1 is one long take that its template continues by itself. Its `video_guide` is that part, whose first frame is frame 0: any tool takes it as a video. Leave it as it is.
- `ingest_source` looks at every shot with `inspect_video` and returns `seen`: who is in it, how their faces show, what changes and when, in seconds of the video (frame = seconds x fps), and any cut measured inside it. It does not know the brief. To look closer, use `wangp_toolbox` with `inspect_media` (`media_inputs`: `source` from `ingest_source` or from the brief, the whole file, with several `frame_no` from the shot's range; its frame numbers are the plan's), or `view_media` when your toolbox has it. If a cut is wrong or missing, call `ingest_source` again with corrected `cuts` before anything else.
- Then go through the shots one by one against the brief and decide two things. Does anything the brief changes appear in this shot? Go by `seen`: when it does not name them, one closer look settles it, and someone you do not find there is too small to show in the remake. If not, leave it alone: no cast, no painted frame, no prompt, and `render` keeps its part of the source as its take, without the model. If so: how many painted frames does it take (see 3)?
- A shot needing the other kind of change takes the other template: `update_shot` with `generation.template` from `template_advice`. Keep its settings.

## 2. Prepare by what the template needs

`ingest_source` returns `needs`. Read the template's input and prompt guidance before the first shot (`wangp_templates` with its task and name).

- **`painted_frames`:** see below.
- **`references`:** give pictures of what the edit brings in as `image_refs` in the shot's settings.
- **`mask`:** make a `video_mask` for the shot's `video_guide` (`create_mask` names objects) and put it in its settings.
- **None of these:** the prompt alone carries the change: go to step 4.

## 3. Painted frames

The video model sees only these pictures of the new look, never the cast's sheets or photos. They decide identity. Use `paint_frame`, or attach your own finished picture with its `picture` argument.

The painter takes each person's looks from their character sheet, which is what keeps them the same from shot to shot: a large portrait beside their whole body from the front, the side and the back, so it brings their looks and no framing or expression. So first add every replaced person with `cast`, with a `description` of their looks and complete outfit, the same words for every frame and nothing about whom they replace, and call `make_sheet` for each (the user's photos in `images` are its starting point). Then give each shot its `cast` with `update_shot`. Only then paint: `paint_frame` refuses a shot without them (`references=[]` for a frame with nobody of the cast in it).

Which frames:

- **One per shot** that contains a change, where it is largest and sharp, at least 3 frames from a cut. When replacing people: every replaced person visible with the whole face. A person in no painted frame renders as the source person.
- **A second one** when `seen` shows a face turning from frontal to profile or back, or a new person entering, at least 3 seconds from the first. A shot too short for two keeps the one that shows the face: a back view alone leaves the face to the model, a new one in every take. To move a frame, paint the new one with `replaces` set to the old one. For a scene change, only for a shot over 8 seconds or a big change of look.
- **A shot that shows the people or what changes,** even dimmed or blurred under a title, is painted like any other. Never ask the painter to replace somebody you cannot see in that frame.
- A part the frame does not show (legs, shoes, a bag) is fixed after the first take, by painting a frame of that take around that part, not by painting more of the source.
- **A painted frame with `by: "user"`** is the user's own picture, brought from the page: keep it as it is. Never paint that frame again, or one that would replace it, unless the user asks.

How to paint:

- When people are replaced, three steps for each frame:
  1. `look_at_frame` lists everyone in that frame: where, how large, what shows of them, hair and clothes, expression, focus.
  2. `paint_frame` with `people`, one entry per person it listed: `who`, `where` (the inspector's words for them, unchanged) and `becomes` (a cast member's name, or null for someone who stays as they are). For someone replaced, `who` finds them by their clothes and place in this frame only, "the woman in the turquoise jumpsuit in the centre", never by their hair, face or glasses: the painter keeps what it reads there. `instruction` only adds what is special in this frame. The tool writes the painter's prompt: the new face and hair come from the cast member's photo and their card's `description`, the clothes from their sheet, and the frame keeps its expression, open mouth, pose and crop. When the brief keeps the person's costume and changes only who wears it, pass `clothes="frame"` on every paint: the frame's own clothes stay, exactly. Never write that characters take each other's place: the painter swaps them.
     The painter is the brief's `frame_painter`. Its Auto, H3-Image, keeps each person's place, pose and expression and handles several people in one frame. When a person becomes a character of another kind (a bear, a robot), it sometimes paints only the head, on the original's body and clothes: `after` shows it, and painting the frame again is the answer. A Frame Painter trained on swaps, such as Viggle Image Swap, takes the face, body and outfit from the character sheet alone, whatever `clothes` says, and no `instruction`: every replaced member then needs a sheet whose card describes the outfit they wear in the remake. Frames without people replaced, and the sheets, go through `image_edits`.
  3. Verify. `paint_frame` returns `before` and `after`, what the inspector sees in each picture apart, and `faults`, measured from them. Its `verify` says what fails. The painter varies a lot from try to try, so a failure usually means painting the same frame again with the same `people`, up to 3 tries, and not rewriting anything. Never go on from a picture that fails without saying so in your report: a wrong frame pulls its whole render toward it.
- For any other change, say what changes and what fills the space, and everything that stays: composition, framing, pose, number of people, background. For a removal, say what is visible where it stood.
- The whole person changes: face, hair, build and whole outfit. Give each character a complete outfit description, the same in every instruction, and keep faces visible (no sunglasses, masks or helmets) unless the brief asks for them.
- For any other change, `paint_frame` returns `comparison`, what the visual inspector sees in both pictures, without knowing what you asked: read it every time (when it is missing, compare `picture` with `source_frame` yourself). What must hold: composition and head count unchanged, nobody added. A wrong person in a frame comes from a wrong instruction: look at `source_frame` again before you repaint, and never choose a frame's people from memory.
- Compare across shots too: outfit colours must match.
- Keep the frame's light: a backlit or dark frame stays that dark, or the render lights it up.
- Keep `pin` on for every painted frame of a swap: unpinned, the model can render the source person again. For an edit, keep it on when the shot has cuts or its composition matters. It holds shots on time.

## 4. Prompts

Follow the template's prompt guidance, not the prompts of older productions: they may use a format since corrected (written lines of dialogue, a swap's painted frame as a `target keyframe`). For a six-section prompt, describe the finished video with `[Shot N]` times from `00:00.000` at the start of the shot's `source_range`, not at the source's start. Name every painted frame as `<Picture N>`, in frame order. A shot with `windows` above 1 needs one complete prompt per window, separated by a blank line, each with its own window times.

What each section of a shot's prompt carries, as the model's own examples have it:

- **The new character:** `<Subject 1> is Dolores in <Picture 1> and <Picture 2>: <face, hair, skin and the whole outfit down to the shoes, in the same words as her cast description>. She replaces <the source person by her own looks: hair, clothes, place in the frame> in <Video 1>.` Bind her to every painted frame of the shot. Describe the outfit as hers even when it is the source's: never "only the face" or "keeps the original clothes", which the model reads as keep the original person.
- **`[Shot 1]` opens with its framing,** then who is where and what they do, and the time each `<Picture N>` matches: `[Shot 1] A medium shot of <Subject 1> skating down the dusty street. At 00:01.750 she is exactly as <Picture 1> shows her.`
- **End a swap with what must not come back:** `In every frame <Subject 1> is this same woman exactly as in <Picture 1>: no short silver hair, no sunglasses at any moment.`, naming the source person's own hair, glasses and clothes items that differ.
- `render` writes the lines that tie the window together: the painted frames' placing and `reference` lines, each character bound to all their pictures, the cuts and the opening of each later shot. Write the rest for the shot alone.

When people speak, call `transcribe_media` (`wangp_toolbox`) once on the whole source, the brief's `source`, only to know who speaks in each shot and when. Never write their words in the prompt: the sound comes from the source, and a written line can make the model voice it instead of following that sound. Write `<Subject 2> (S2) is speaking with natural lip movements synchronized, <d>[unclear]</d>` with the time it starts, leave a shot's `dialogue` empty, and keep the action to speaking: a smile or gesture written at that moment competes with the mouth.

A mouth or a head that does not follow the source is not fixed by a new seed or more words in the prompt. Never judge lip sync on a draft: the same shot, frame and seed followed the speech at 720p and not at 480p, so tell the user to look at the mouths at Final. Otherwise render it again once at most, keep the better take and say so in your report: a loop of renders costs minutes each and changes nothing.

## 5. Shoot and judge

- `render` all the shots in order, the ones you left alone too: they come back with `kept_source`, already judged. For long remakes paint and write every shot first, then render them together, so that models swap as little as possible.
- Short shots render with their neighbours in one request (the brief's Shots rendered together): alone, the lips barely follow the speech. Keep writing each shot's prompt for that shot alone, its times from its own first frame: `render` joins them. A render lists in `rendered_with` the other shots it gave a take (chosen when they had none, else beside the one they use): judge those too, and do not render them again. A shot renders with its neighbours only once they all have their painted frames and prompts.
- `render` returns `check`, each take measured against its source (`check_take` measures an older take). Look at each take beside the source, with a frame of every camera setup in it (`media_inputs`, one `{media_id, frame_no}` item per frame: without one you see only its first frame, before its cuts), and record `verdict` with the measures as `checks`. Ask about each person with the `description` of their cast card, read in the production, not from memory.
- **Wrong:** wrong identity, an original face or costume item left, wrong head count, a missed or added cut, melted faces, light lost.
- **`check` faults:** a cut moved late or dropped: when the setup after it has no painted frame, paint and pin one first; otherwise `render` with `new_seed`. Never pass a take with a dropped cut as a detail. A cut moved early: paint and pin a frame of the outgoing shot 6 frames before that cut, not another seed. An edge band: a painted frame carries it; repaint that frame, another seed will not help.
- At most one fix per shot on your own. Then report what is still weak: the user decides.

## 6. Deliver

`assemble` joins the takes on their trims and plays the source's own sound over them, at its own level, and keeps the source's subtitles: do not add a song or subtitles unless asked. Report what is weak by time range.
