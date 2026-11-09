# Media Flow

Media Flow gives existing videos and images a new life: it restores them (removes blur, compression artifacts or a stylized grade, converts SDR to HDR, adds fine detail), outpaints them to a new aspect ratio, upscales them or makes their motion smoother. It works on one file or on whole folders, and on videos of any length: they are processed piece by piece, so memory use stays low, a long job can be stopped and resumed hours later, and the original soundtrack is kept without re-encoding.

Media Flow comes enabled with WanGP. Open **Media Flow** in the app bar, or open `/media-flow/` on the WanGP address to go directly to its full-screen view. **Full Screen** at the top of the tab expands it in place; **WanGP** returns to the normal view with your settings preserved.

## Quick Start

1. At the top, choose **One Item** or **Batch**, then **Video** or **Image**.
2. In **Process**, pick a **Model**, then a **Process**. The card below the list says what the process does; processes marked **Recommended** are the best starting point for their task.
3. In **Source**, type the path of your file or click the folder button to browse.
4. Leave **Output** empty to save the result next to the source, or type a file name or a folder.
5. Click **Start Process**. Follow the progress in the **Run** panel, watch the **Live Preview**, and find the finished file under **Result**.

Media Flow runs one process at a time. Deepy and external agents can run the same processes; **Stop** in this panel also stops a process an agent started.

## Choosing a Process

Each process applies one task with one model. Pick the model first: the **Process** list then shows what it can do.

| Task | What it does | When to use it |
| --- | --- | --- |
| **Outpaint** | Extends the picture to a new aspect ratio, inventing the missing borders. | Turn a 4:3 or vertical video into 16:9, or the reverse. |
| **Refocus** | Removes focus and motion blur. | Soft, out-of-focus or shaky footage. |
| **Uncompress** | Removes blocking, banding and smearing left by heavy compression. | Web videos, old streams, low-bitrate files. |
| **Ungrade** | Removes a stylized color grade and restores neutral colors. | Footage with a strong teal-orange, sepia or faded look. |
| **SDR to HDR** | Converts standard dynamic range media to HDR. | Bring older material to HDR displays. |
| **Detail** | Adds fine texture and sharper edges. | Generated or upscaled videos that look smooth or plastic. |
| **Upscale** | Increases the resolution. | Low-resolution sources, or a final pass before delivery. |
| **Interpolate** | Adds frames between frames for smoother motion and a higher frame rate. | Choppy or low frame rate videos. |
| **User Settings** | Runs your own WanGP settings on each piece of the source. | Workflows not covered by the built-in processes. |

Some processes need a large model and a recent GPU, others (such as Lanczos) are almost instant. Processes marked **Legacy** use an older model version and are kept for comparison.

**Your own settings:** in WanGP, select a model that can continue a source video, enable its **Control Video**, disable **Reference Images** and audio inputs, and save your settings. In Media Flow, choose **Add User Settings** in the **Model** list, pick that settings file in **Process** and click the **+** button. Linked settings appear under their model, marked with an asterisk; the bin button unlinks them. Image settings work the same way with an image model.

## Sources and Batches

**One Item** processes a single video or image. Type its path in **Source** or click the folder button to browse.

**Batch** processes many files in a row. Enter one entry per line in **Sources**:

- a file, such as `D:\clips\interview.mp4`;
- a folder, which adds every supported file it contains;
- a wildcard pattern, such as `D:\clips\*.mp4` or `D:\shoot\day*\*.mov`.

The **Output** field of a batch accepts:

- nothing: each result is saved next to its source;
- one folder: every result is saved there;
- one line per source line: a file name for a single source, a folder, or a pattern with the same number of `*` as its source line, such as `D:\clean\*_clean.mp4`, which receives the parts the source pattern matched.

**Batch Name** names the batch and its progress files, saved in the output folder (or next to the first source). With **Continue** enabled, starting the same batch again with the same name and settings skips the files already done and resumes the one that was interrupted. A batch with a new name, or with Continue disabled, starts from the beginning.

A file that fails is recorded and the batch moves on to the next one; the **Run** panel counts completed files and failures.

## Output and Quality

- **Output:** leave it empty for an automatic name next to the source (it includes the process, ratio, resolution and time range), type a full file name, or type a folder for an automatic name inside it. When that file already exists and **Continue** is off, a new numbered name is used, so nothing is overwritten.
- **Output Resolution:** the size of the result for processes that generate the picture again. Higher resolutions keep more detail but take longer and need more memory. 1440p and 2160p appear when 4K resolutions are enabled in **Configuration**.
- **Target Ratio:** for outpainting, the aspect ratio of the result. Upscalers and frame interpolators show their **Upsampling** multiplier here instead, such as x2.
- For **FlashVSR**, select the model and transformer quantization in **Configuration > Extensions > FlashVSR**, then choose One Pass or Two Pass in Media Flow.
- For **SeedVR2**, select **3B** or **7B** in **Configuration > Extensions > SeedVR2**. The Upsampling control shows the selected model for both image and video processes. Each run or batch keeps the model it started with.
- **RelayVSR** carries its memory of the previous keyframes from one piece to the next, so its pieces join as if the video were upscaled in one go. A resumed run rebuilds that memory from the frames just before the resume point, which keeps the join smooth but not identical to an uninterrupted run.
- **Process Strength:** how strongly a restoration is applied. Lower values stay closer to the source, higher values push the effect further.
- **Upscaler options:** some upscalers add their own settings below, such as the number of passes or tile sizes.
- **Live Preview:** shows the last frame of the latest finished piece, so you can judge the result early and stop if it is not what you want.
- **Keep in Galleries & Output Folders:** each piece of a long video is also a WanGP generation. Media Flow keeps only the most recent ones in the galleries and the outputs folder and deletes older pieces as it goes. The final result is always kept.

## Long Videos

Long videos are processed in **chunks**, short pieces processed one after the other and joined without visible seams. Only the current chunk is held in memory, so a two-hour video needs no more memory than a short clip.

- **Chunk Size:** the length of each piece, in seconds. Shorter chunks need less memory and give earlier previews; longer chunks give the model more context and fewer joins. A chunk cannot be longer than what the process generates in one pass (481 frames for the LTX processes, 16 seconds at 30 FPS); a longer value is reduced to that length when the run starts.
- **Sliding Window Overlap:** the number of frames each chunk shares with the previous one so that the joins stay seamless. The default suits most videos; increase it if you notice jumps at chunk boundaries.
- **Start** and **End:** process only part of the video. Enter seconds (`90`), minutes and seconds (`01:30`) or hours, minutes and seconds (`01:02:30.5`). Leave them empty to process the whole video.
- **Audio Track:** **Auto** keeps every audio track of the source; choose a track number to keep only that one. The audio is copied as it is, without re-encoding.

## Prompts

Some processes accept a prompt. It describes the content or the rendering you want, such as the scenery to invent around an outpainted picture or the texture and sharpness of a refined video. Processes without a prompt hide this field.

For a long video, give different prompts to different parts of the video with **timed blocks**: each block starts with a timestamp on its own line, followed by its prompt, and blocks are separated by an empty line.

```text
00:00
A calm harbour at dawn, soft mist over the water.

01:30
The harbour at noon, bright sun and crowded quays.
```

Each block applies from its timestamp until the next one; timestamps are `MM:SS` or `HH:MM:SS`, with optional hundredths.

## Stop and Resume

- **Stop** finishes the chunk in progress, then stops. What was processed is already saved.
- To resume, keep **Continue** enabled and start the same process with the same source and output: Media Flow finds the unfinished output and processes only what is missing, then joins everything into one file.
- **Continue** also resumes an interrupted batch (see **Sources and Batches**). Disable it to start again from the beginning in a new file.
- You can switch to another WanGP tab while a process runs; come back to Media Flow to follow it.
- Keep any small resume files Media Flow leaves next to an unfinished output if you move it, so that it can resume just as smoothly.

## Tips and Troubleshooting

- Try a short range first with **Start** and **End**: you see the result in minutes before committing to a long run.
- Restoration processes work best on the problem they target: combine them in successive runs (for example **Uncompress**, then **Upscale**) rather than expecting one process to fix everything.
- Outpainting to the same resolution as the source spreads the original picture over a smaller area; choose a higher **Output Resolution** to keep its detail.
- If a process runs out of memory, reduce the **Chunk Size** or the **Output Resolution**, or choose a lighter process.
- If joins between chunks are visible, increase **Sliding Window Overlap**.
- A process that is missing from the list may need a feature your installation does not provide, such as a compatible NVIDIA GPU for DLSS.

---

> Applies to: The bundled Media Flow app: restoring, outpainting, upscaling and interpolating existing videos and images, single files or resumable batches, from its WanGP tab or full-screen view.
