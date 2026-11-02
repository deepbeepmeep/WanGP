---
name: web-apps
description: "Use before building a web app with wangp_python (app_new, app_check, app_publish): interactive animations, games, three.js scenes, physics toys and visualizers that the user plays inside WanGP, with pictures, 3D models and music made by WanGP."
---

# Web apps

A web app runs in the user's browser, inside WanGP (a Play card in the chat, then the Apps gallery). It is a folder `apps/<name>/` in the Python sandbox: `app.json`, `index.html`, ES modules and `assets/`. The app itself is JavaScript: `wangp_python` creates and checks it, `wangp_io` reads and edits its files, and Python runs make its data and assets. Use an app for something the user interacts with or that runs live (a game, a toy, an explorable 3D scene, a generative animation, a visualizer). A fixed clip is a video (`python-clips`), not an app.

## Loop

0. Something played through in several steps (levels, puzzles, quests, an adventure, an interactive story or mystery): read wangp://skills/progression-design and plan the progression before making assets.
1. `app_new` with a template: `three-scene` (3D scene, orbit controls), `three-game` (3D game: keys, score, timer, best score), `phaser-game` (2D game with physics and sprites), `canvas-2d` (2D animation without a library), `physics-toy` (three.js + cannon-es), `audio-visualizer` (Tone.js, starts on a click), `blank`. Name and describe it; the template runs as it is.
2. Make the assets with WanGP (below), then write the code with `wangp_io` on the app's folder (`@documents/sandbox/apps/<name>/`, as `app_new` returns it): `read_text` a file, change it with `edit` (an exact passage replaced, the usual way to change code), add modules with `write_text`. A tool call holds at most a few thousand tokens and a longer one is cut off before it runs, so keep each call under about 150 lines: split a game into modules (`main.js` importing `./notes.js`, `./render.js`, `./input.js`), and write a long file in parts with `write_text` then `append_text`. Replacing a whole template file is a `write_text` with `mode: overwrite`, not an `edit`.
3. `app_check` after each change: it returns error findings, the state probe after each input step, the frame rate, screenshots and, with `record`, a motion strip. Fix every error finding before anything else.
4. Look at it when appearance matters: `wangp_toolbox` `inspect_media` (`wangp.inspect` in a script) on a screenshot path or on the motion strip (`@documents/sandbox/tmp/app_checks/...`), asking about what the user should see.
5. The chat shows a Play card after each change to the app; when the user plays it, `app_logs` returns what their browser reported (errors, frame rate).
6. `app_publish` when it works: the app joins the Apps gallery as a new version.

To change an app the user already has: if it is in the sandbox (`apps/<name>/`), edit it and publish again; an app of the Apps gallery that is not there (made in another conversation, imported) comes back with `app_open` (its media id from `wangp_list_gallery`, or its `.wgpapp` path), which copies it into the sandbox and numbers the next publication after its latest version.

A new module:

```json
{"action": "write_text", "arguments": {"path": "@documents/sandbox/apps/orbit-garden/planets.js", "text": "import * as THREE from 'three';\n\nexport function makePlanet(texture, radius, distance, speed) {\n  const map = new THREE.TextureLoader().load(texture);\n  map.colorSpace = THREE.SRGBColorSpace;\n  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 48, 32), new THREE.MeshStandardMaterial({ map, roughness: 0.8 }));\n  return { mesh, distance, speed, angle: Math.random() * Math.PI * 2 };\n}\n"}}
```

A change: copy whole lines from `read_text` into `old_string`, indentation included, with enough lines to occur once; `new_string` replaces them. Several changes are several calls.

```json
{"action": "edit", "arguments": {"path": "@documents/sandbox/apps/orbit-garden/main.js", "old_string": "  planet.angle += dt * planet.speed;", "new_string": "  planet.angle += dt * planet.speed * 0.5;"}}
```

Data computed in Python (levels, beat maps, word lists) is written by a run as a JSON file under `assets/` (`json.dump(level, open('apps/orbit-garden/assets/level.json', 'w'))`), which the app fetches with a relative path.

## Assets made by WanGP

Generate, then copy into the app with `wangp.app.import_asset(media, app, path, max_size=None, transparent=False)` (or the `import_asset` action): it converts to the format of `path` and resizes pictures to `max_size` on the longer side.

- Textures, skies, backgrounds, sprites, characters, icons: `wangp.generate(...)` with an image template; `import_asset(media_id, 'orbit-garden', 'assets/mars.webp', max_size=1024)`. Seamless textures: ask for "seamless tileable texture" and check the edges. Sprites, characters and objects with a transparent background: generate them on a plain background, then `import_asset(media_id, app, 'assets/fox.png', max_size=256, transparent=True)` cuts the subject out and crops to it (no hand-made masks). Animate a generated character with tweens (bob, squash, lean, flip) rather than redrawing it as pixel art; for other poses, edit the same picture with the image-editing template (it as the reference), one pose per picture.
- 3D models: generate a picture, then an image-to-3D template; import as `.glb` and load with `GLTFLoader`.
- Music: song templates. Sound effects (lasers, impacts, explosions, pickups, UI clicks, ambience): no template task; find the models with `wangp_models` search `specialities: ["sound effects"]`, then generate with that `model_type` and a prompt describing the sound and its length. Import as `.ogg` (or `.mp3`); play with `new Audio('assets/theme.ogg')` or `Tone.Player`, after a click.
- Video backgrounds and screens: a video template; import as `.mp4` (`max_size=1280`). Drawn into WebGL (Phaser `add.video`, `THREE.VideoTexture`) the video needs `crossOrigin = 'anonymous'` (Phaser: `this.load.setCORS('anonymous')` before `this.load.video`), or WebGL refuses it as cross-origin data; a muted `<video autoplay loop playsinline>` behind the canvas needs a transparent canvas instead (Phaser `transparent: true` in the game config, three.js `alpha: true` and no `scene.background`).
- Levels, maps, beat maps, word lists: computed in Python, written as JSON under `assets/`.

A beat map from a generated song (notes on onsets in 4 lanes, from bass to treble, and the tempo), in one run:

```python
import json, numpy as np, soundfile as sf
from scipy.signal import find_peaks

audio, rate = sf.read(wangp.path(song_id), dtype='float32', always_2d=True)  # wangp.path: soundfile needs a local path, not an @alias
mono = audio.mean(axis=1)  # one channel: stereo arrays break 1-D signal functions
hop, size = 512, 2048
frames = np.lib.stride_tricks.sliding_window_view(mono, size)[::hop] * np.hanning(size).astype(np.float32)
flux = np.maximum(np.diff(np.log1p(np.abs(np.fft.rfft(frames, axis=1))), axis=0), 0)  # rising energy per frequency
edges = np.searchsorted(np.fft.rfftfreq(size, 1 / rate), [200, 800, 3000])
strength = np.stack([band.sum(axis=1) for band in np.split(flux, edges, axis=1)])
strength = (strength - strength.mean(axis=1, keepdims=True)) / (strength.std(axis=1, keepdims=True) + 1e-9)
onset, frame_s = strength.mean(axis=0), hop / rate
lags = np.arange(int(60 / 180 / frame_s), int(60 / 80 / frame_s))  # tempo between 80 and 180 BPM
bpm = 60 / (lags[int(np.argmax([np.dot(onset[:-lag], onset[lag:]) for lag in lags]))] * frame_s)
peaks, _ = find_peaks(onset, height=0.8, distance=int(0.15 / frame_s))  # raise height for fewer notes
notes = [{'t': round(((index + 1) * hop + size / 2) / rate, 3), 'lane': int(np.argmax(strength[:, index]))} for index in peaks]
json.dump({'bpm': round(bpm, 1), 'duration': round(len(mono) / rate, 2), 'notes': notes}, open('apps/neon-beat/assets/beatmap.json', 'w'))
print(round(bpm, 1), len(notes), np.bincount([note['lane'] for note in notes], minlength=4))
```

The app plays the song with `new Audio('assets/song.ogg')` and places the notes against `audio.currentTime`, never against its own clock, so they stay in sync.

Keep the whole app under 200 MB; 1024 px textures and OGG audio are plenty.

## Libraries (import map)

Import by these names only; nothing loads from the internet (no CDN, no web fonts, no remote fetch).

- `import * as THREE from 'three'` (r186) and plugins by file: `three/plugins/controls/OrbitControls.js`, `three/plugins/loaders/GLTFLoader.js`, `three/plugins/postprocessing/EffectComposer.js` (+ `RenderPass.js`, `UnrealBloomPass.js`, `OutputPass.js`), `three/plugins/geometries/TextGeometry.js` with `FontLoader.js` and the fonts at `/apps/lib/three/fonts/helvetiker_regular.typeface.json`. There is no `three/plugins/Plugins.js` index. Use `THREE.Timer` (`timer.update(time)` in the loop), not the deprecated `THREE.Clock`.
- `import * as CANNON from 'cannon-es'`: physics, stepped with `world.step(1 / 60, dt, 3)`.
- `import Phaser from 'phaser'` (3.90): scenes, arcade physics, input, tweens, sound. Pictures display at their own pixel size: give sprites their game size with `setDisplaySize(w, h)` (a full-window background: `this.add.image(0, 0, 'bg').setOrigin(0).setDisplaySize(width, height).setScrollFactor(0)`), and pin texts that must stay on screen (score, counters) with `setScrollFactor(0)`, or the camera scrolls them away. Keys: `this.cursors = this.input.keyboard.createCursorKeys()` and `this.input.keyboard.addKey('SPACE')`; `key.isDown` is a property (not a function), `Phaser.Input.Keyboard.JustDown(key)` is one press. Key events take Phaser's key names: `keydown-SPACE`, `keydown-A`, `keydown-ONE` (digits spelled out: `keydown-1` never fires).
- `import { Tween, Group, Easing } from '@tweenjs/tween.js'`, `import { createNoise2D, createNoise3D } from 'simplex-noise'`.
- `import * as Tone from 'tone'`: synths, sequences, players; `await Tone.start()` inside a click handler.

three.js calls that are often misremembered: `loader.load(url, onLoad, onProgress, onError)` (the third callback is download progress, not an error), `composer.addPass(new RenderPass(scene, camera))` (no `addRenderPass`), `quaternion.setFromUnitVectors(a, b)` (plural), `object.lookAt(x, y, z)`, `THREE.MathUtils.clamp(v, min, max)`; a texture needs `texture.colorSpace = THREE.SRGBColorSpace`, and tiling needs `texture.wrapS = texture.wrapT = THREE.RepeatWrapping; texture.repeat.set(8, 8)`. Arithmetic only on numbers: `object.uuid` is a string, so `Math.sin(t + mesh.uuid)` gives NaN and the object vanishes (use an index or `mesh.id`).

Prefer the built-in materials to a `ShaderMaterial`: hand-written GLSL fails on details three.js otherwise handles (undeclared attributes, matrix sizes). A textured terrain with height bands is `new THREE.MeshStandardMaterial({ map, vertexColors: true })` (the colours tint the texture; keep them light so it shows), with `map.wrapS = map.wrapT = THREE.RepeatWrapping`, `map.repeat.set(16, 16)` and `map.anisotropy = renderer.capabilities.getMaxAnisotropy()` so that it stays sharp at grazing angles.

## Rules of a good app

- Fill the window without scrollbars (the templates' `style.css`), follow `resize`, cap the pixel ratio at 2.
- Clamp the time step (`Math.min(dt, 0.05)`): the player pauses apps when they are hidden.
- List the controls in `app.json` (`"controls": {"WASD": "move", "Space": "jump"}`): the player shows them under the app. Prevent the default of arrow keys and Space so that the page does not scroll.
- Keep `window.wangpState()` returning the values that tell whether the app works (score, lives, level, positions, counts): `app_check` records it after each step, so game logic is checked without pictures. Include the positions of the key objects (player, targets, enemies): an object that `inspect_media` cannot see, or a score that never rises, usually shows there as `null` (JSON's NaN) or as a position far from where it should be.
- `localStorage` works and keeps its data per app (best scores, saves, settings), also in the published version.
- Light every three.js scene and point the camera at the subject: a black canvas usually means no light, a camera looking away or an error before the first frame.
- Sound starts only after a click or key press (browser rule): show a start button or start it from the first input.

## Checking input

`app_check` takes the app's folder name as `app` (the `app` of every result). `steps` play the app: `{"key": "ArrowRight", "hold_s": 1}`, `{"key": "Space"}`, `{"click": "#start"}`, `{"click": "text=Play"}`, `{"click": [640, 360]}`, `{"mouse": [x, y]}`, `{"drag": [[x1, y1], [x2, y2]]}`, `{"wait_s": 1}`, `{"shot": "after-jump"}`, and `{"type": "..."}` only to fill a text field. A step done several times is written once with a count: `{"key": "Space", "repeat": 20, "every_s": 0.3}`. Each step's result carries the state probe, and `state` is where the steps led: check that the score rose, the player moved, the level changed. Hits, pickups and deaths in an action game seldom happen under blind input: give the game a test key (such as `T`) that sets up a sure case (a target right in front of the player, an item under it), so a check can press it, act, and read the score or lives change. Use `"viewport": "phone"` when the app should also work on a phone (touch input).
