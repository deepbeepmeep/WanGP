---
name: python-images
description: "Use for exact text over pictures (posters, titles, captions), grids and contact sheets, picture-book pages, palettes, masks and measurements, or procedural SVG drawings made in wangp_python."
---

# Images and text in Python

Models paint the picture; Python places exact text and measured layouts over it. Generate or pick the key art first, then compose.

## Text over a picture

```python
# Poster title over the key art
from PIL import Image, ImageDraw, ImageFilter
img = Image.open(wangp.path("visual:IMAGE")).convert("RGB")
W, H = img.size
title = "BLUE HOUR JAZZ"
font = wangp.font(int(H * 0.11), "display", bold=True)
box = ImageDraw.Draw(img).textbbox((0, 0), title, font=font)
x, y = (W - (box[2] - box[0])) // 2, int(H * 0.72)
shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
ImageDraw.Draw(shadow).text((x + 6, y + 6), title, font=font, fill=(0, 0, 0, 180))
img = Image.alpha_composite(img.convert("RGBA"), shadow.filter(ImageFilter.GaussianBlur(8)))
ImageDraw.Draw(img).text((x, y), title, font=font, fill=(255, 240, 200, 255))
img.convert("RGB").save("out/poster.png")
(W, H, x, y)
```

- Fonts only through `wangp.font(size, family, bold, italic)`: families `sans`, `serif`, `mono`, `condensed`, `display`. Never guess font files.
- Text inside any box (speech bubble, badge, button, tile, panel): measure it centred, `l, t, r, b = draw.textbbox((0, 0), text, font=font, anchor="mm")`, size the box as `(r - l) + 2 * pad` by `(b - t) + 2 * pad`, then draw the text at the box centre with `anchor="mm"`. Without the anchor, PIL places the top-left of the line box, so the glyphs sit low and touch the bottom edge.
- Measure where the text really is: `box = draw.textbbox((x, y), text, font=font)` at the drawing position gives the glyphs' true top and bottom (a box measured at `(0, 0)` starts below 0, so `y + height` lands inside tall letters). Place the next line, rule or panel from `box[3]` plus a gap. `anchor="mm"` centres a line on a point; `textwrap.wrap(text, width=n)` with `draw.multiline_text(..., spacing=...)` for paragraphs.
- Margins as fractions of the width (0.06 to 0.1); one display face for titles, one sans for the rest; a translucent panel (`Image.new("RGBA", (w, h), (0, 0, 0, 140))`) behind small text over busy pictures; check contrast with `ImageStat.Stat(crop).mean`.
- Then `inspect_media` on the result with pointed questions: "Does any line, shape or shadow cross or touch the letters?", "Read the text exactly", "Is any text cut off?". A general "does it look right" misses a rule drawn through a word.

## Diagrams and infographics

Several labels beside drawn shapes (clouds, arrows, a mountain) collide easily, and a visual check of the whole picture often misses light text over a shape. Lay out in code:

- Keep every placed element as a box `(x0, y0, x1, y1)`: text boxes from `draw.textbbox((x, y), ...)` at their real position, shapes from the coordinates you drew them with.
- Before drawing, test each text box against every shape box (`a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]`); move the text, shrink the shape, or put the text on a panel until nothing intersects. Print the collisions you found and fixed.
- Text inside a circle or badge: `draw.text((cx, cy), "3", font=f, anchor="mm")` centres it; without `anchor` the glyph hangs below and to the right of the point.
- Inspect crops of the dense regions (`bbox` in `inspect_media`) as well as the whole picture.

## Grids, sheets and pages

Open each image, `ImageOps.fit(img, (cell_w, cell_h))`, paste on a canvas with gutters, caption under each cell; round corners by pasting through an `L` mask drawn with `rounded_rectangle`. A picture-book page: picture in the upper two thirds, wrapped text in the lower third on a paper-coloured background.

## Palettes, masks, measurements

- Dominant colours: `small = img.convert("RGB").resize((64, 64)); pal = small.quantize(6).getpalette()[:18]`, swatches with hex labels (`"#%02x%02x%02x"`).
- To change one object (recolour a product, blur a background, cut out a subject), ask WanGP for its mask first: `m = wangp.toolbox("create_mask", media_id=image, keywords=["the sneaker"])` gives a mask image (its `media_id` in the result), far cleaner than thresholds or OpenCV contours. Then `mask = Image.open(wangp.path(m["media_id"])).convert("L")` and work only inside it: hue shift in HSV with numpy, `Image.composite(edited, original, mask)`.
- Masks from `image_channels` are images too; the centroid of a mask tells where the subject sits.
- Visual duplicates: a perceptual hash (8x8 or 16x16 grey thumbnail, bits above the mean) groups resized copies; confirm close pairs with a pixel comparison at one size.

## Procedural SVG

Build markup in Python when a drawing has many regular elements (rays, grids, stars, waves, gradients, diagrams), then `wangp.svg(markup, W, H)` gives an RGBA image to composite or save. For an animated vector card without Python, `draw_svg` with `duration_s` is the direct tool.
