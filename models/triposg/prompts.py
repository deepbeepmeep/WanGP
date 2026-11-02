_LICENSE_NOTE = ("TripoSG and PartCrafter weights are MIT. Their model code derives from Tencent HunyuanDiT (Tencent Hunyuan Community License, "
                 "not licensed in the European Union) and TripoSG's surface extraction from Tencent FlashVDM (not licensed in the European Union, "
                 "the United Kingdom and South Korea); these agreements also forbid using the outputs to improve other AI models. See "
                 "models/triposg/Notice.txt.")

TRIPOSG_INFOS = f"""
# TripoSG

TripoSG (VAST, 1.5B parameters) turns one picture of an object into a detailed 3D mesh (GLB, geometry only). It is a rectified-flow
transformer generating 2048 shape latents, decoded into a fine occupancy volume from which the surface is extracted.

## Inputs

- **Object Image** (required): one whole, uncropped object on a plain background. The background is removed automatically; an image
  with a transparent background keeps its own cutout. The object is cropped and centred with a small margin.
- **Asset Name / Description**: names the files and is stored in the metadata. The shape comes from the image alone.

## Settings

- **Steps** (50, upstream) and **Guidance** (7.0, upstream): fewer steps (25-30) give quick drafts; lower guidance follows the picture
  less, much higher values can add artifacts. Guidance 1 disables the unconditional pass (twice as fast, weaker shapes).
- **Mesh Resolution**: 512 (upstream) keeps fine details such as thin edges and engravings; 256 is a faster draft with fewer faces.
- **Max Faces**: 0 keeps the full surface (often over a million faces at 512); a number reduces it with quadric decimation for lighter
  files and real-time use.
- **Number of Samples**: several candidate meshes, each with the seed plus its index.

## Output

A watertight GLB mesh, Y up, without UVs or materials. Its presets add PBR textures (SUPRA-Tex or Hunyuan3D 2.1 Paint) and fine
geometric detail (Hi3D) in the same generation.

## Limits

One object per image; hidden sides are inferred. Clean, well-lit product shots and renders work best.

{_LICENSE_NOTE}
"""

PARTCRAFTER_INFOS = f"""
# PartCrafter

PartCrafter (built on TripoSG) turns one picture of an object into **separate part meshes** generated together: for example the seat,
backrest and legs of a chair, or the head, body and limbs of a toy. The result is one GLB with one node per part (`part_0`, `part_1`,
...), each part in its own colour, ready to be moved, edited or textured separately.

## Inputs

- **Object Image** (required): one whole object on a plain background (removed automatically). PartCrafter was trained on rendered
  objects: clean renders and product shots work much better than busy photos.
- **Asset Name / Description**: names the files only.

## Settings

- **Number of Parts** (1-16): how many parts to generate. Choose it from the object's natural pieces (a mug: 2, a chair: 4-6, a
  robot: 5-8). Too many parts split pieces arbitrarily; too few merge them. A part that comes out empty is omitted.
- **Steps** (50) and **Guidance** (7.0): upstream defaults.
- **Mesh Resolution**: 512 (upstream) or 256 (faster). Each part is decoded on its own, so the time grows with the number of parts.
- **Max Faces**: 0 keeps everything; otherwise the total face budget, shared between the parts by size.
- **Number of Samples**: several candidates, each with the seed plus its index.

## Limits

Parts are inferred, not semantic labels: their order and colours are arbitrary. Small parts can overlap or be missing. Geometry only.

{_LICENSE_NOTE}
"""

PROMPT_INFOS = """
# Asset Name

TripoSG and PartCrafter do not read the prompt: it only names the generated files and is stored in the GLB metadata. Write a short name,
for example `red toy robot` or `oak chair`.

The Object Image decides the result: one object, whole, centred, on a plain background, seen from the front or three-quarters, with even
light. To start from text, generate such an image first (for example "a red toy robot, full body, three-quarter view, plain white
background, soft even lighting"), then use it as the Object Image.
"""

TRIPOSG_DEEPY_INFOS = ("Image to detailed 3D mesh (GLB, geometry only). Needs exactly one `image_refs` image: one whole object on a plain background "
                       "(removed automatically). `prompt` = asset name only, not read. Steps 50, guidance 7. custom_settings: `mesh_resolution` "
                       "256/512, `max_faces` (0 keeps all). batch_size = candidate meshes. " + _LICENSE_NOTE[:_LICENSE_NOTE.index(" See ")])

PARTCRAFTER_DEEPY_INFOS = ("Image to a 3D object made of separate part meshes (one GLB, nodes part_0..., one colour per part). Needs exactly one "
                           "`image_refs` image of one object on a plain background (clean renders work best). `prompt` = asset name only. Steps 50, "
                           "guidance 7. custom_settings: `num_parts` 1-16 (natural pieces of the object, default 4), `mesh_resolution` 256/512, "
                           "`max_faces` (0 keeps all, else total budget). " + _LICENSE_NOTE[:_LICENSE_NOTE.index(" See ")])

DEEPY_PROMPT_INFOS = ("`prompt` (optional): a short asset name such as `red toy robot` for the file name; nothing else is read. The reference image decides the result: one "
                      "centred, uncropped object, plain background, front or three-quarter view, even light.")

ASSET_NAME_INSTRUCTIONS = """You name 3D assets. Look at the image and name the single main object it shows in 2 to 6 words, lower case, for
example: red toy robot, ceramic teapot with blue flowers, wooden office chair. Keep the words of the user's text when it already names the
object. Answer with the name only, without punctuation, quotes or commentary."""

GEOSAM2_INFOS = """
# GeoSAM2

GeoSAM2 (VAST, CVPR 2026, Apache-2.0) splits an existing mesh into its parts: the mesh is rendered from 12 views (normal and position
maps), a fine-tuned SAM 2.1 segments one view, propagates the segments to the other views like a video, and the masks are lifted back
to the faces. The result is one GLB with one node per part (`part_0`, `part_1`, ... from the largest), each in its own colour unless
the mesh has a texture, which is kept.

## Inputs

- **Input Mesh** (required, .glb/.obj/.ply/.stl): any closed or open triangle mesh, for example a TripoSG or Hunyuan3D result. Its
  orientation should follow glTF (Y up, front towards +Z) so that the views see it upright.
- **Point Prompts** (optional): leave empty for an automatic segmentation into all its parts. To choose the parts, give points on
  one rendered view in GeoSAM2's format, a JSON list such as
  `[{"frame_idx": 3, "obj_id": 1, "point": [512, 420], "label": 1}, {"frame_idx": 3, "obj_id": 2, "point": [512, 700], "label": 1}]`:
  `frame_idx` is the view (0-11, all points on the same view), `obj_id` the part number, `point` the pixel (x, y) in the 1024 x 1024
  view, `label` 1 to include the point, 0 to exclude it. Every run writes a `_views.png` side file showing the 12 views (512-pixel tiles:
  double the tile coordinates). View 3 sees the front from slightly below, view 9 the back from above, views 0 and 6 the sides.
- **Asset Name / Description**: names the files only.

## Settings

- **Small Part Cleanup** (0.02, upstream): fragments under this fraction of the largest part (in faces and in area) are merged into
  their neighbours before every connected region becomes its own part. 0.01 keeps finer pieces, 0.035 gives fewer, larger parts.
- **Parts**: *Automatic + Prompted* segments everything (the prompted parts first); *Prompted Parts Only* segments only the prompted
  parts: the rest of the mesh is not split further, but each of its separate pieces (not touching the prompted parts or each other)
  still becomes its own part.
- **Seed**: the automatic masks start from random points of the views and the faces are sampled randomly: another seed gives a
  slightly different split.

## Limits

Parts are what the views show: hidden inner surfaces take the label of their nearest visible neighbours. Very thin parts can be
merged. Large meshes (over a million faces) take longer to lift and clean up.

**License:** GeoSAM2 is Apache-2.0 (VAST-AI Research): models/triposg/LICENSE_GeoSAM2 and NOTICE_GeoSAM2.
"""

GEOSAM2_PROMPT_INFOS = """
# GeoSAM2 Prompts

The main prompt is only the asset name (file names and metadata).

**Point Prompts** (optional) use GeoSAM2's own point format: a JSON list of points on one of the 12 rendered views,
`{"frame_idx": view, "obj_id": part, "point": [x, y], "label": 1}`, with x and y in pixels of the 1024 x 1024 view (twice the
position in the 512-pixel tiles of the `_views.png` side file of a previous run). Several points with the same `obj_id` describe one
part; `label` 0 marks a point that must not belong to that part. Example, a mug split into its body and handle on view 0:
`[{"frame_idx": 0, "obj_id": 1, "point": [480, 520], "label": 1}, {"frame_idx": 0, "obj_id": 2, "point": [700, 500], "label": 1}]`.
Leave it empty for the automatic segmentation.
"""

GEOSAM2_DEEPY_INFOS = ("3D part segmentation of an existing mesh: `custom_guide` = the mesh file (.glb/.obj/.ply/.stl, required) -> one GLB "
                       "with one node per part (part_0 = largest), parts coloured unless textured. No image input. Automatic when `alt_prompt` "
                       "is empty; otherwise `alt_prompt` = GeoSAM2 point prompts JSON list on one rendered view: "
                       "[{\"frame_idx\": 0-11, \"obj_id\": part, \"point\": [x, y] in 1024 px, \"label\": 1 or 0}] (the `_views.png` side file "
                       "of a run shows the views at 512 px). custom_settings: `postprocess_pa` 0.01/0.02/0.035 (fragment cleanup, larger = "
                       "fewer parts), `auto_segment` 1 = automatic + prompted parts, 0 = prompted parts only (the rest stays unsplit, one part per separate piece). Apache-2.0.")

GEOSAM2_DEEPY_PROMPT_INFOS = ("`prompt` (optional): asset name only. `alt_prompt`: empty for automatic parts, or a JSON list of points "
                              "{\"frame_idx\", \"obj_id\", \"point\": [x, y], \"label\"} on one 1024 px rendered view.")
