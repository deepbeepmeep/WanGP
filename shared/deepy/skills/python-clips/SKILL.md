---
name: python-clips
description: "Use for clips built frame by frame in wangp_python: animated titles and charts, pan and zoom over stills, slideshows, kinetic text, and subtitle files (SRT) from measured timestamps."
---

# Clips frame by frame

`wangp.publish_video(frames, fps=24, name="clip", audio=None)` is the only encoder (H.264). It takes PIL images or arrays of one size with even dimensions, PNG paths, or a glob such as `"tmp/frames/*.png"`. Without `audio` the clip is silent; `audio=<media id or path>` gives it a soundtrack in the same call: the source video's sound when you redrew its frames (titles, overlays, colour grades), or music and narration.

## Editing the frames of an existing video

Read the source with OpenCV (`cv2.VideoCapture(wangp.path(video_id))`, its `CAP_PROP_FPS` and frame size), draw on each frame with PIL (titles, lower thirds, watermarks, grading), write PNGs under `tmp/frames/`, then `wangp.publish_video("tmp/frames/*.png", fps=source_fps, name=..., audio=video_id)` so the result keeps the original sound.

## A clip from scratch

```python
# Animated title reveal, 4 s at 24 fps
from PIL import Image, ImageDraw
W, H, fps, seconds = 1280, 720, 24, 4
font = wangp.font(120, "display", bold=True)
ease = lambda t: 1 - (1 - t) ** 3
frames = []
for i in range(fps * seconds):
    t = ease(min(1, i / (fps * 1.5)))
    frame = Image.new("RGB", (W, H), (12, 14, 22))
    ImageDraw.Draw(frame).text((W // 2, int(H * 0.8 - t * H * 0.3)), "BLUE HOUR", font=font, fill=(255, 255, 255), anchor="mm")
    frames.append(frame)
wangp.publish_video(frames, fps=fps, name="title_reveal")
```

- Frames never go under `out/` (each file there is published). Keep short clips in memory; for clips longer than about 10 s write PNGs under `tmp/frames/` (create it first: `os.makedirs("tmp/frames", exist_ok=True)`) while rendering and pass the glob. Rendering hundreds of large frames (a zoom per frame over 30 s at 1024 px) takes minutes: pass `timeout_s=300` to that run.
- Match an existing video's size and frame rate (`wangp.io("info", {"path": ref})`) when the clip will be merged with it; then `merge_videos`. Sound comes afterwards with `remux_media` (music, narration).
- Pan and zoom over a still: crop a moving window of a large image per frame (ease the motion) and resize it to the output size.
- Animated chart: draw the same matplotlib figure per frame with values scaled by an easing of `t`, turn it into a PIL image (`fig.canvas.draw(); Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())`), close the figure, collect.
- Check the published clip at several times (`inspect_video`, or `inspect_media` with several `media_inputs` times): a correct last frame says nothing about the motion before it.

## A slideshow timed to a narration or a song

Generate the narration (or song) and the pictures first, then one run makes the video with the sound as its soundtrack; no merge or remux chain is needed:

```python
# Slideshow timed to the narration
from PIL import Image
narration, pictures = "audio:NARRATION", ["visual:PICTURE_1", "visual:PICTURE_2", "visual:PICTURE_3", "visual:PICTURE_4"]
seconds = wangp.io("info", {"path": narration})["duration_seconds"]
fps, W, H = 24, 1280, 720
frames = []
for ref in pictures:
    frames += [Image.open(wangp.path(ref)).convert("RGB").resize((W, H))] * round(seconds * fps / len(pictures))
wangp.publish_video(frames, fps=fps, name="slideshow", audio=narration)
```

To change pictures on each sentence, take the sentence start times from `transcribe_media` with word timestamps and give each picture the frames between two starts; a slow zoom per picture (see the title example above) makes it livelier.

## Subtitles

`transcribe_media` with word timestamps gives words and times; group 6 to 8 words (break at punctuation), end each cue at the next cue's start, write `out/captions.srt` (`HH:MM:SS,mmm --> HH:MM:SS,mmm`), then pass the returned path in `remux_media` `subtitle_tracks`. Subtitles stay a separate track: burn text into frames only for designed on-screen text.

## Timing

Shot plans, offsets and window lengths are arithmetic: compute them here, and ask `wangp_validate` for the exact lengths WanGP will produce before cutting music or narration to them.
