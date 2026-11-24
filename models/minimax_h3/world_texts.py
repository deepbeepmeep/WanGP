# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Help texts of pi-H3 worlds (UI, Deepy)."""

H3_WORLD_INFOS = """### What it makes
A scene you walk through, about 4 seconds at a time, with its sound: each move continues the last one without a cut, so the walk stays
one continuous shot. It runs on the 2-step pi-H3 model, close to real time at 832x480 on a fast GPU. The world is saved in the **Worlds**
gallery, where **Explore** continues it full screen with the keyboard: **W A S D** walk and step sideways, the **arrows** turn and tilt
the view, **Space** hops, **Shift** runs, **C** crouches, and **Enter** types an event. **Esc** leaves; **Esc** twice quickly drops the
queued moves and pauses (a move key goes on). Keys pressed while a move is being generated are queued: the next moves take them in order. Holding a key keeps walking:
the next move is prepared while the current one plays, so the walk does not stop (it goes on for one more move after you let go).

### Inputs
- **Prompt**: a description of the scene (a plain sentence or a structured H3 prompt with its soundscape), then optional instruction
  lines that move the camera, the same moves as the explorer's keys: `[Forward]`, `[Backward]`, `[Left]`, `[Right]`, `[Look_Left]`,
  `[Look_Right]`, `[Look_Up]`, `[Look_Down]`, `[Jump]`, `[Sprint]`, `[Crouch]`, `[Wait]`, joined with `+`, and
  `[Event] text`. Each instruction is one move of about 4 s (`[Forward 8s]` makes two); without instructions the description alone plays
  for the selected length.
- **Start Image** (optional): the first view of a new world.
- **Continue a World**: a world of the Worlds gallery to extend with the prompt's instructions; the description can be left out.

### Continuity and memory
Each move starts from the last frame and the sound of the previous one, so moves are generated faster than they play on a fast GPU and
the walk does not pause. The explorer's Memory can also show each move the last 0.7 or 1.4 s of the previous one, for steadier motion,
but moves then get shorter and slower to make than to watch (short pauses between them). It remembers what was just shown, not places
seen long ago.

### Limits
- The moves follow the text: the model chooses how far and how fast the camera goes, and big moves can change the scene.
- The explorer's clips are silent; the sound is made for worlds generated from the prompt (World and its Video).
- Replay and Record keep the last 3 minutes of an exploration."""

H3_WORLD_PROMPT_INFOS = """### Writing the prompt
Describe the place, its look and its sound, as for a continuous single take:
*"integrated_multimodal_description: [Shot 1] A narrow cobblestone street in an old town at sunset, cafe terraces on both sides,
first-person view at eye level.*
*overall_soundscape: Footsteps on stones, distant chatter and a church bell."*
A plain sentence works too: it becomes the description field. Avoid cuts, timestamps and pictures (`<Picture 1>`): each move is one
continuous take of the same scene.

### Moves
One instruction per line after the description:
- `[Forward]`, `[Backward]`, `[Left]`, `[Right]` walk and step sideways, `[Look_Left]`, `[Look_Right]`, `[Look_Up]`, `[Look_Down]` turn
  and tilt the view, `[Jump]` hops, `[Sprint]` runs (with `[Forward]` or alone), `[Crouch]` lowers the view, `[Wait]` holds the camera still; `[Forward+Look_Left]` combines them. Each move adds its camera sentence
  to the description, in the H3 format (the soundscape field keeps yours, or gets one for a walk).
- `[Forward 8s]` repeats a move (about 4 s each).
- `[Event] text` adds what happens during the move: *`[Forward] [Event] the street lamps light up`*.

Example:
```
A misty pine forest at dawn, a dirt trail between tall trunks, soft light rays. Birds and a light wind.
[Forward]
[Right]
[Forward] [Event] a deer crosses the trail
```"""

H3_WORLD_DEEPY_INFOS = """pi-H3 World (2-step pi-H3 DWD, 832x480 default, 24 fps): an explorable walk through a scene with sound, generated about 4 s per move
(107-frame windows: the first segment shows 103 frames, each next one continues from the previous segment's last frame and adds 102
new frames, 4.25 s, about 3.7 s to generate on a 5090; with 1 or 2 history clips of the previous segment, the explorer's Memory, 85 or 68
new frames and slower than playback). No cut: the previous segment's last frame (and history clips) encoded once, its audio latents
reused, nothing decoded. Prompt = scene description (plain text or H3 structured prompt), then optional instruction lines, the explorer's
keys: [Forward] [Backward] [Left] [Right] (walk, step sideways), [Look_Left] [Look_Right] [Look_Up] [Look_Down] (pan, tilt), [Jump] [Sprint]
[Crouch] [Wait], joined with +, optional length ([Forward 8s] = 2 segments), [Event] text. Each move appends a fixed
camera sentence (type, amplitude, speed) to the description field and the prompt keeps H3's three fields (a plain description gets a
walking soundscape and non_diegetic_music N/A). No instructions: the description alone for video_length frames. image_start optional
(image_prompt_type "S") for the first view. custom_guide = a .wgpworld to continue. Output world_output "world" (Worlds gallery),
"world_video" or "video". 2 fixed steps, no guidance. Memory: the last frame (or 0.7 / 1.4 s) of video, no long-term memory of places."""

H3_WORLD_DEEPY_PROMPT_INFOS = """Prompt: the scene as one continuous take (place, look, sound; H3 fields integrated_multimodal_description / overall_soundscape
allowed, no cuts, no timestamps, no <Picture N>, no camera moves: the instructions add them). Then one instruction per line: [Forward]
[Backward] [Left] [Right] [Look_Left] [Look_Right] [Look_Up] [Look_Down] [Jump] [Sprint] [Crouch] [Wait], + to combine,
optional length in seconds (whole 4.25 s moves), [Event] followed by what happens during that move."""
