---
name: python-3d
description: "Use to build 3D objects in wangp_python (parametric shapes, terrains, assemblies), save them as meshes in the 3D gallery and get snapshots, turntables, depth or normal maps with WanGP's renderer for checks or as inputs to image and video models."
---

# 3D objects in Python

Build vertices and faces with numpy (diagrams, 3D charts, props, terrains, logos); save the mesh under `out/`. After the run it joins the 3D gallery with a `media_id` (OBJ, PLY, STL and GLTF are converted to GLB with a preview; GLB is kept). Then WanGP renders it.

```python
# Parametric vase as OBJ
import numpy as np
rings, around = 40, 48
t = np.linspace(0, 1, rings)
radius = 0.35 + 0.25 * np.sin(np.pi * t) ** 2 + 0.1 * np.sin(6 * np.pi * t)
theta = np.linspace(0, 2 * np.pi, around, endpoint=False)
verts = np.array([[r * np.cos(a), 2.0 * h, r * np.sin(a)] for r, h in zip(radius, t) for a in theta])
faces = []
for i in range(rings - 1):
    for j in range(around):
        a, b = i * around + j, i * around + (j + 1) % around
        c, d = a + around, b + around
        faces += [[a, c, b], [b, c, d]]
faces += [[0, (j + 1) % around, j] for j in range(1, around - 1)]  # bottom cap (fan), so the vase holds water
with open("out/vase.obj", "w") as f:
    f.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in verts)
    f.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)
len(verts), len(faces)
```

- OBJ faces are 1-based and counter-clockwise seen from outside (normals point out). Close open ends with caps when a solid is expected. Y is up.
- Colours: per vertex as `v x y z r g b` lines (0..1 floats, for gradients such as terrain by height), or per part with a `.mtl` next to the OBJ (`mtllib` + `usemtl`, `Kd r g b`). WanGP converts the OBJ to GLB itself: do not write GLB binaries by hand (one misaligned buffer offset shifts every colour channel).
- Several parts: offset each part's vertices, append its faces with the index offset.
- Keep vertex counts in the thousands for quick renders; a sphere is a grid in (latitude, longitude), a terrain a height grid, a twisted shape a rotation applied per ring.

## Text, labels and logos

`wangp.text_mesh(text, height=None, width=None, depth=0.1, family="sans", bold=False)` returns `(vertices, faces, normals)` of extruded text: several lines with `\n`, centred on the origin, reading along +X and facing +Z, fitted inside `height` and/or `width`. It is drawn with the sandbox's fonts and extruded from the pixels, so holes (o, a, R), accents and every font are right and the normals point outward. Never triangulate glyph outlines yourself: matplotlib's `TextPath` draws one line only (a `\n` becomes a missing-glyph box and the label runs past its box) and its outlines need hole handling.

```python
# Labelled box: its name on the front face
import numpy as np
V, F, N, C = [], [], [], []
def add(v, f, n, rgb):
    start = len(V); V.extend(v); F.extend(np.asarray(f) + start); N.extend(n); C.extend([rgb] * len(v))  # f may be a list
def box(cx, cy, cz, w, h, d, rgb):
    corners = np.array([[cx + sx * w / 2, cy + sy * h / 2, cz + sz * d / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    quads = [((0, 1, 3, 2), (-1, 0, 0)), ((4, 6, 7, 5), (1, 0, 0)), ((0, 4, 5, 1), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)), ((0, 2, 6, 4), (0, 0, -1)), ((1, 5, 7, 3), (0, 0, 1))]
    add(corners, np.array([t for (a, b, c, d), _ in quads for t in ((a, b, c), (a, c, d))]), [n for _, n in quads for _ in (0, 1)], rgb)
w, h, d = 2.4, 0.85, 0.9
box(0, 0, 0, w, h, d, (0.95, 0.76, 0.50))
v, f, n = wangp.text_mesh("Multi-Head\nAttention", height=h * 0.7, width=w * 0.9, depth=0.04, bold=True)
add(v + [0, 0, d / 2 + 0.02], f, n, (0.13, 0.13, 0.16))  # on the front face, slightly proud of it
with open("out/labelled_box.obj", "w") as fh:
    fh.writelines(f"v {x:.5f} {y:.5f} {z:.5f} {r:.3f} {g:.3f} {b:.3f}\n" for (x, y, z), (r, g, b) in zip(V, C))
    fh.writelines(f"vn {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in N)
    fh.writelines(f"f {a + 1}//{k + 1} {b + 1}//{k + 1} {c + 1}//{k + 1}\n" for k, (a, b, c) in enumerate(F))
```

A title is the same call placed where it belongs (`pixels=128` for large letters); rotate the vertices to put text on another face. Any other flat shape (a downloaded logo, an icon, a silhouette, a stencil, a cookie cutter) goes through `wangp.extrude(shape, height=, width=, depth=)`; to make a silhouette of an object in a photo, cut it out first with `wangp_toolbox` `create_mask` (WanGP's segmenter, one call) rather than OpenCV thresholds or GrabCut, then extrude the mask. Curved outlines show fine pixel steps, smaller than a pixel of a 768-pixel render at the default `max_pixels=400` (800 halves them): judge the model at the size the user sees it, not in close-up renders, never blur or filter the renders to hide them, and never build your own outline triangulation instead: a PIL image, an image path or Gallery id, or a boolean mask, whose bright or opaque pixels become solid (invert a dark logo on white first with `PIL.ImageOps.invert(image.convert('L'))`). A coloured design (a logo, a flag, an icon) is one call: `parts = wangp.extrude_parts(image, colors=3, width=, depth=)` returns one `(vertices, faces, normals, rgb)` per colour, aligned so the parts fit together; add each with its `rgb` as in the labelled box above. Count every colour of the design, small details included (eyes, outlines, a white star), and use a transparent PNG when you have one. Colour other parts by giving their vertices their own colour: one flat colour per part. The extruded faces are a few large rectangles, so a per-vertex gradient comes out as bands: that is how the mesh is built, not a renderer fault. One normal per face (`vn`) keeps the edges crisp: without them WanGP smooths shading across sharp edges.

## Snapshots with WanGP's renderer

Check a build once with `render_3d_views` and `inspect_media`, then show it to the user: a good first version beats several silent rebuilds. Small floating fragments the inspector reports can be misreadings: look at the spot with `focus` and `zoom` before rebuilding, and when a flaw remains, name it and offer the fix instead of iterating. Never debug the renderer or the file format (test planes, parsing GLB bytes): change the model, or report the limit. One `inspect_media` of a render is the check: re-reading every label from crops or pixel probes is not needed, and text on a face reads normally from its front only (from behind you see the box).

A mesh under `out/` is published when the run ends: render it in the next run with the `media_id` that run returned in `outputs`.

```python
# Front and three-quarter views of the vase
views = wangp.toolbox("render_3d_views", media_id="3d:MESH", views=["front", "three_quarter"], width=768, height=768, gallery=False)
views
```

Views: `front`, `back`, `left`, `right`, `top`, `bottom`, `three_quarter`, or `angles=[[azimuth, elevation], ...]` in degrees; `zoom`, `focus`, `mode` (shaded, or depth / normal passes: read the contract once with `wangp_toolbox` `action="render_3d_views"` and no arguments), `turntable_s` for a 360-degree video. `gallery=False` keeps check renders out of the Gallery (they go to `@documents/renders/`, still inspectable); render the views the user should see without it. Then `inspect_media` on a view checks the shape ("Is the base closed? Any holes or inverted faces?"). A turntable makes a product shot.

## From a render to a picture

A render is a precise layout for an image model: `wangp.generate({"template": "edit_image/<template>", "prompt": "Photorealistic photograph of the house shown in <image1>, same layout ...", "image_refs": [render_id]})`. The `edit_image` templates (`wangp_deepy_templates` with task `edit_image` lists them) keep the reference's composition; describe the materials and setting the render lacks. A three-quarter view shows more of the shape (roof pitch, depth) than a straight front view; a depth or normal render can serve as a control input for models that take one.
