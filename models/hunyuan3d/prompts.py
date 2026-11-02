_LICENSE_NOTE = ("Hunyuan3D 2.1 is distributed under the Tencent Hunyuan 3D 2.1 Community License (models/hunyuan3d/LICENSE): it is not "
                 "licensed in the European Union, the United Kingdom and South Korea, and its outputs may not be used to improve other AI models.")

SHAPE_INFOS = f"""
# Hunyuan3D 2.1 Shape

Turns one picture of an object into a 3D mesh (GLB, geometry only).

## Inputs

- **Reference Image** (required): one object, whole and uncropped, ideally a front or three-quarter view on a plain background. The
  background is removed automatically; an image with a transparent background keeps its own cutout.
- **Prompt**: the asset name, used for the file names and metadata only. The shape comes from the image alone.

## Settings

- **Steps**: 30 is the upstream default; 50 adds a little detail, 10-20 give quick drafts.
- **Guidance**: 5.0. Lower values follow the image less, much higher ones can add artifacts.
- **Mesh Resolution**: 256 is standard; 384 and 512 capture finer details with more faces and a slower surface extraction; 196 is a
  fast draft.
- **Max Faces**: 0 keeps the full surface (often several hundred thousand faces); a number reduces it with quadric decimation, for
  lighter files and real-time use.
- **Number of Samples**: several candidate meshes of the same image, one GLB each.

## Textured Assets

The mesh has no UVs or materials. Add a texture in the same generation with a **Texturer** (Post Processing tab, also chosen by the
presets): **Hunyuan3D 2.1 Paint** for the upstream PBR pipeline (the preset *Hunyuan3D 2.1 Paint PBR* reproduces the former
Hunyuan3D 2.1 PBR model), **SUPRA-Tex** for textures that keep the picture's print. A **3D Refiner** (Hi3D) can add fine geometric
detail first.

## Limits

One object per image. Hidden sides and thin parts are inferred and can be wrong.

{_LICENSE_NOTE}
"""

PAINT_INFOS = f"""
# Hunyuan3D 2.1 Paint

Paints an existing mesh with PBR textures (GLB: base colour plus metallic-roughness) that match a picture of the object. The same
model is the **Hunyuan3D 2.1 Paint** Texturer of the Post Processing tab, which paints the meshes of any 3D model or of the 3D gallery.

## Inputs

- **Input Mesh** (required, .glb, .obj, .ply or .stl): the mesh is welded, reduced to Max Faces and unwrapped: its own UVs and
  materials are replaced. It should face the same way as the picture (front towards +Z, up +Y, as glTF).
- **Reference Image** (required): the same object, whole and uncropped, on a plain background. The background is removed
  automatically.
- **Prompt**: the asset name, used for the file names and metadata only.

## Settings

- **Guidance** (3.0) drives the generated texture views. The texture model always runs 15 steps.
- **Max Faces**: 40000 (upstream), 0 meaning 40000; higher values keep more detail but unwrap and bake more slowly. Large meshes are
  unwrapped in smaller regions with progress and cancellation, then packed into one texture atlas. This preserves the face geometry
  but can add UV seams between regions.
- **Texture Views**: 6 views at 512 px are the fastest and lightest; 8 views at 768 px (upstream default) cover the object better and
  give sharper textures; 9 views add one more angle when needed.
- **Texture Size**: 2048 (upstream); 1024 for small files; 4096 is sharper, baked at full size.

## Limits

The texture follows the visible side of the image and invents the hidden ones; fine text and logos come out blurred. Lighting
baked into the picture can stay in the base colour. Among the Texturers it ranks between SUPRA-Tex (more faithful when the picture
lines up with the mesh: it keeps print and patterns) and TRELLIS.2 Texture (faster, approximate colours); it is the safe choice when
the picture and the mesh differ.

{_LICENSE_NOTE}
"""

PROMPT_INFOS = """
# Asset Name

Hunyuan3D 2.1 does not read the prompt: it only names the generated files and is stored in the GLB metadata. Write a short name, for
example `red toy robot` or `oak chair`.

The Reference Image drives everything: one object, whole, centred, on a plain background, seen from the front or three-quarters, with
even light. To start from text, generate such an image with an image model first (for example "a red toy robot, full body,
three-quarter view, plain white background, soft even lighting"), then use it as the Reference Image.
"""

SHAPE_DEEPY_INFOS = ("Image to 3D mesh (GLB, geometry only). Needs exactly one reference image (`image_refs`): one whole object on a plain "
                     "background (removed automatically). `prompt` = asset name only, not read by the model. Steps 30, guidance 5. "
                     "custom_settings: `octree_resolution` 196/256/384/512 (detail), `max_faces` (0 keeps all faces). batch_size = candidate meshes. "
                     "Textured asset in the same task: `temporal_upsampling` `hunyuan3d_paint` (upstream PBR paint; temporal_postprocessing_parameters "
                     "`texture_views`, `texture_size`, `texture_max_faces` 40000, `texture_guidance` 3) or `supratex`; `spatial_upsampling` `hi3d` refines "
                     "the geometry first. Tencent license: not for EU/UK/South Korea; outputs may not train other AI models.")

PAINT_DEEPY_INFOS = ("PBR textures (base colour, metallic-roughness) painted on a supplied mesh: `custom_guide` = the mesh (.glb/.obj/.ply/.stl, "
                     "facing +Z, up +Y), one reference image (`image_refs`) of the same object on a plain background. `prompt` = asset name only. "
                     "guidance 3.0 = texture guidance (the texture model always runs 15 steps). custom_settings: `max_faces` (40000, 0 = 40000; large "
                     "meshes unwrap in regions, preserving geometry but adding UV seams), `texture_views` 6_512/8_768/9_768, `texture_size` "
                     "1024/2048/4096. Also the `hunyuan3d_paint` Texturer (temporal slot) of any 3D model; less faithful than `supratex`, more than "
                     "`trellis2_texture`. "
                     "Tencent license: not for EU/UK/South Korea; outputs may not train other AI models.")

DEEPY_PROMPT_INFOS = ("`prompt` (optional): a short asset name such as `red toy robot` for the file name; nothing else is read. The reference image decides the result: "
                      "one centred, uncropped object, plain background, front or three-quarter view, even light.")

ASSET_NAME_INSTRUCTIONS = """You name 3D assets. Look at the image and name the single main object it shows in 2 to 6 words, for example:
red toy robot, ceramic teapot with blue flowers, wooden office chair. If the user's text already names the object, repeat it character for
character, keeping its capital letters and spelling. If it gives only part of a name, keep its words unchanged and complete them from the
image. Answer with the name only, without quotes or commentary."""
