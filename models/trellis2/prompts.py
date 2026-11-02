TEXTURER_RANKING = """
## Texturers, from the most to the least faithful

1. **SUPRA-Tex**: the visible side keeps the picture's colours, print and patterns, the hidden side is completed consistently. It
   needs a picture that lines up with the mesh: seen from the front or three quarters, at most moderately from above (a steep top-down
   photo misplaces the colours). Its material stage is non-commercial.
2. **Hunyuan3D 2.1 Paint**: paints views all around the mesh; colours follow the picture, fine print comes out blurred. The safe
   choice when the picture and the mesh differ.
3. **TRELLIS.2 Texture**: several times faster than the others; plausible materials whose colours only approximate the picture and
   change with the seed, without its print or patterns.
"""

SHAPE_INFOS = """
# TRELLIS.2 Shape

TRELLIS.2 (Microsoft, 4B parameters, MIT) turns one image of an object into an untextured 3D mesh (GLB). It keeps thin and intricate
structures (handles, leaves, frames) well. The shape is generated as a coarse occupancy structure, then a detailed shape, extracted as a
clean closed surface and simplified.

## Inputs

- **Object Image** (required): one picture of a single object. A transparent image (PNG/WebP with alpha) is used as is, its alpha
  being the object mask; otherwise the background is removed automatically. The object is then cropped and centered. Whole, unoccluded
  objects on a plain background, seen from a three-quarter view, work best; cropped, cluttered or very dark images give incomplete shapes.
- **Asset Name / Description**: names the files and is stored in the metadata. TRELLIS.2 is conditioned on the image only, so this
  text does not change the result.

## Pipelines

- **512**: fastest, coarse geometry (512 voxels across the object).
- **1024**: shape generated directly at 1024.
- **1024 Cascade** (recommended): a 512 shape is generated first and refined at 1024, which keeps the 512 proportions with 1024
  details.
- **1536 Cascade**: the same at 1536 for the finest details; slower and more memory hungry. For complex objects its resolution is
  lowered by steps of 128 until the shape fits in **Max Tokens**.

## Settings

- **Number of Inference Steps** (12): steps of each stage (structure, shape).
- **Guidance** (7.5): image guidance, applied to the early steps with a rescale, as in the official pipeline. Lower values give
  simpler shapes, higher values follow the image more strictly.
- **Max Faces** (100,000): the raw mesh (millions of faces) is simplified to this face count; lower it for real-time use, raise it
  for printing or close-ups.
- **Seed / Batch Size**: each asset of a batch uses the seed plus its index.

## Textures and Presets

The mesh comes out untextured. Pick a preset, or a Texturer in **Post Processing**, to add PBR textures (base colour, metallic,
roughness) in the same generation from the same picture; a 3D Refiner (Hi3D) can add fine relief first.
""" + TEXTURER_RANKING + """
## Output

GLB files (Y up) that open in any glTF viewer, game engine or 3D editor. The asset is generated in the canonical pose learned from
the training data: it usually faces +Z, but nearly symmetric objects can come out turned by 90 or 180 degrees depending on the seed.
The surface is rebuilt as a clean shell around the generated one (slightly thicker thin parts), then simplified. Very thin parts can
merge or show small holes.

**License:** TRELLIS.2 is MIT (models/trellis2/LICENSE), as is the TRELLIS sparse-structure decoder it reuses. Its DINOv3 image encoder
is under Meta's DINOv3 License (LICENSE_DINOv3.md beside the weights).
"""

TEXTURE_INFOS = """
# TRELLIS.2 Texture

TRELLIS.2 Texture (Microsoft, MIT) generates PBR textures (base colour, metallic, roughness) for a supplied mesh from a picture of the
same object. It is the fastest Texturer, but the least faithful one: the materials are plausible, their colours only approximate the
picture, and its print or patterns are not reproduced. Prefer SUPRA-Tex or Hunyuan3D 2.1 Paint when the result must match the picture.

## Inputs

- **Input Mesh** (required, .glb/.obj/.ply/.stl): the mesh to texture, for example a TRELLIS.2 Shape, Hunyuan3D, TripoSG or Hi3D
  result. It should stand upright (Y up) with its front towards +Z, as the meshes of these models do.
- **Object Image** (required): a picture of the same object; its background is removed when it has no transparency.

## Settings

- **Texture Voxels**: 1024 (recommended) gives finer colour detail than 512.
- **Number of Inference Steps** (12) and **Guidance** (1 = off, the official setting): above 1 the materials follow the picture more
  strongly but can saturate.
- **Max Faces** (0 = keep all): the mesh is reduced to this face count first.
- **Texture Size**: size of the texture images. **Transparency**: *Alpha Blending* keeps the generated transparency (glass, thin
  materials), *Opaque* (default) ignores it.
""" + TEXTURER_RANKING + """
**License:** TRELLIS.2 is MIT (models/trellis2/LICENSE), as is the TRELLIS sparse-structure decoder it reuses. Its DINOv3 image encoder
is under Meta's DINOv3 License (LICENSE_DINOv3.md beside the weights).
"""

TRELLIS2_PROMPT_INFOS = """
# TRELLIS.2 Asset Name

TRELLIS.2 builds the 3D asset from the Object Image alone: the text is not a prompt. Write a short name or description of the asset,
such as `Bronze lion statue` or `Red armchair, low poly reference`; it becomes the output file name and is stored with the settings.

To create an asset from text, first generate the object with an image model (a single centered object, plain background, whole
object in frame, three-quarter view, soft even light), then send that image to TRELLIS.2 as its Object Image.
"""

SHAPE_DEEPY_INFOS = """Image-to-3D shape: one `image_refs` image of a single object (an RGBA image's alpha is used as the mask, else background removal) -> untextured GLB mesh (Y up) that keeps thin, intricate structures. The prompt is only the asset name (file name), not a conditioning. `model_mode`: 0 = 512 (fast, coarse), 1 = 1024, 2 = 1024 Cascade (default, best balance), 3 = 1536 Cascade (finest, slowest). Defaults: 12 steps, guidance 7.5. Custom settings: `max_faces` (100000), `max_tokens` (1536 cascade cap). Textures: set `temporal_upsampling` to a Texturer, from the most to the least faithful: `supratex` (keeps the picture's colours and print when the image lines up with the mesh, not from a steep top-down view; non-commercial material stage), `hunyuan3d_paint` (colours follow, print blurred, robust to mismatches), `trellis2_texture` (several times faster, approximate colours only); `spatial_upsampling` `hi3d` adds relief to solid objects first. Best inputs: whole object, plain background, three-quarter view; for text-to-3D generate such an image first."""

TEXTURE_DEEPY_INFOS = """Mesh texturing, fast but approximate: `custom_guide` = the mesh (.glb/.obj/.ply/.stl, Y up, front towards +Z) and one `image_refs` picture of the same object -> GLB with PBR textures whose colours only approximate the picture (no print or patterns). Prefer `supratex_texture` (most faithful) or `hunyuan3d_21_paint` unless speed matters. `model_mode` 512 or 1024 (default) texture voxels; 12 steps; guidance 1 (off, official). Custom settings: `max_faces` (0 = keep all), `texture_size` 1024/2048/4096, `alpha_mode` opaque/blend."""

TRELLIS2_DEEPY_PROMPT_INFOS = """The prompt is optional: a short asset name such as `Bronze lion statue` that names the output file; it does not change the 3D result, which depends only on the reference image."""
