---
name: character-sheet
description: Build a character sheet (turnaround views at one scale, optional close-ups, title, labels and notes) from a 3D asset or a character image, check it visually, and reuse it as consistent references for image or video generation.
---

# Character Sheet

A character sheet shows one character from several angles at the same scale, labelled, on one image. Build it from a 3D asset so that every view is the same character, then lay it out.

## 1. Source

- **A 3D asset exists** (3D gallery: GLB mesh or Gaussian splats): use its media ID. The gallery listing gives `kind`, `faces` / `gaussians` and, for an animation, `frame_count`, `fps` and `duration_seconds`.
- **Only a character image exists**: first make a textured mesh with a `gen_3d` template (a shape model with a Texturer; TripoSG and PartCrafter alone give untextured shapes), keeping the automatic background removal on. Use a full-body image, front or three-quarter, with the whole character visible.
- Look at the asset before rendering: `inspect_media` accepts the 3D media ID directly (three-quarter preview).

## 2. Orientation

Named views assume the character faces the front (+Z): `left` and `right` are the character's own sides. Render `views=["front"]` once and inspect it. If the character does not face the camera, find the azimuth that shows its face (try `angles=[[90, 0]]`, `[[180, 0]]`, `[[-90, 0]]`) and use custom `angles` offset by that azimuth for the whole sheet: front `a`, character's left `a + 90`, back `a + 180`, character's right `a - 90`.

## 3. Turnaround

Render all turnaround views in **one** `render_3d_views` call so they share one scale:

```json
{"action": "render_3d_views", "arguments": {"media_id": "3d:ASSET", "views": ["front", "three_quarter", "left", "back", "right"], "width": 512, "height": 768}}
```

- Orthographic projection is the default (no perspective distortion). Portrait sizes such as 512 x 768 suit standing characters; landscape suits creatures and vehicles.
- `three_quarter` is seen from 20 degrees above, so its feet sit lower than in the level views. When the sheet needs one flat baseline, use `"views": ["front", "left", "back", "right"], "angles": [[35, 0]]` (the custom 3/4 view comes last in the results).
- The background is transparent by default, which is best for a designed layout; use `"background": "white"` for a plain grid.
- Animated asset: choose a representative pose with `frame_no` (below `frame_count`); the rest pose is the default.
- Gaussian splats render in `shaded` or `mask` mode only and have no frames.

## 4. Optional details

Close-ups use a separate call (their scale differs), with `focus` as fractions of the bounding box and `zoom`:

- face: `"views": ["front", "three_quarter"], "focus": [0.5, 0.9, 0.6], "zoom": 4`;
- props or hands: inspect a full view first to estimate their position, then set `focus` there.

`mode="depth"` or `"normal"` views make control images for depth- or normal-guided generation.

## 5. Layout

- **Quick grid**: `side_by_side` with the view media IDs in order, `layout="horizontal"` (or `COLSxROWS`) and `legends` such as `["Front", "3/4", "Left", "Back", "Right"]`.
- **Designed sheet**: `draw_svg` with the views embedded through `<image href="MEDIA_ID">`, a title, labels under each view, an optional ground line, colour swatches and short notes. Keep views the same size and aligned on one baseline. Quote every visible text exactly; do not leave placeholders.

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="2700" height="1000" viewBox="0 0 2700 1000">
  <rect width="2700" height="1000" fill="#f3efe6"/>
  <text x="1350" y="70" text-anchor="middle" font-family="sans-serif" font-size="46" font-weight="bold">Aura — Character Sheet</text>
  <line x1="40" y1="880" x2="2660" y2="880" stroke="#b8b0a0" stroke-width="2"/>
  <image href="visual:FRONT" x="40" y="110" width="512" height="768"/>
  <image href="visual:THREE_QUARTER" x="570" y="110" width="512" height="768"/>
  <image href="visual:LEFT" x="1100" y="110" width="512" height="768"/>
  <image href="visual:BACK" x="1630" y="110" width="512" height="768"/>
  <image href="visual:RIGHT" x="2160" y="110" width="512" height="768"/>
  <g font-family="sans-serif" font-size="30" text-anchor="middle" fill="#3b3631">
    <text x="296" y="930">Front</text><text x="826" y="930">3/4</text><text x="1356" y="930">Left</text>
    <text x="1886" y="930">Back</text><text x="2416" y="930">Right</text>
  </g>
</svg>
```

Replace the `visual:...` placeholders with the media IDs returned by `render_3d_views`, in view order.

## 6. Check and refine

Inspect the sheet with `inspect_media`: every view present and complete, same scale and baseline, labels readable and correctly placed, no overlap. Redraw with `draw_svg` to fix the layout (the PNG's settings keep the SVG source). Do not describe details the 3D asset does not show.

## 7. Use

- 3D textures are softer than the source image. To sharpen or restyle, edit each view with an image editing model ("the same character in the same pose and outfit, clean high-detail illustration") before the layout, then verify identity with `inspect_media`.
- Views or the sheet become references for image or video generation (`image_refs`), which keeps the character consistent across shots.
