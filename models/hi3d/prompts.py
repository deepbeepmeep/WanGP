_LICENSE_NOTE = ("Hi3D is licensed under the Hi3D Open License Agreement Version 1.0 (models/hi3d/LICENSE and NOTICE): commercial use is "
                 "allowed below 200,000 monthly active users and USD 1M monthly revenue, products built on it must display \"Powered by Hi3D "
                 "1.0\", and the Acceptable Use Policy applies. Generated meshes themselves carry no such obligation.")

HI3D_INFOS = f"""
# Hi3D v1.0

Hi3D refines a **coarse 3D mesh** into **high-resolution geometry** guided by a **picture of the same object**: sharp edges, seams,
buttons, scales, engravings and other fine relief that image-to-3D generators only hint at. The result is a geometry-only GLB.

## Inputs

- **Object Image** (required): the object the mesh shows, ideally the picture the coarse mesh was made from. The background is removed
  automatically; an image with a transparent background keeps its own cutout.
- **Input Mesh** (required, .glb/.obj/.ply/.stl): the coarse mesh, for example a Hunyuan3D 2.1 Shape, TripoSG or TRELLIS.2 result.
  Select it in the 3D gallery and use its **To Input Mesh** button. Its orientation should follow glTF (Y up, front towards +Z), as
  WanGP's 3D generators produce.
- **Asset Name / Description**: names the files and is stored in the metadata. The geometry comes from the mesh and the image.

## Settings

- **Refinement Level**: the detail of the final surface.
  - **512**: one pass, a quick upgrade (about half a minute on a fast GPU).
  - **1024** (recommended): two passes, much finer detail (one to three minutes on a fast GPU).
  - **1536**: three passes for the finest relief (around five minutes on a fast GPU). It differs little from 1024 on most
    objects, and on a large or complex object the finest pass runs at a coarser grid (between 1024 and 1536).
- **Steps** (25 per pass): fewer steps are faster but leave the surface noisier.
- **Max Faces**: 0 keeps the level's default (1 million faces, 2 million at 1536). A lower value gives lighter files.
- **Number of Samples**: several refinements of the same mesh, each with the seed plus its index.

## Output

A closed GLB mesh in the position, orientation and scale of the Input Mesh, without UVs or materials. To texture it in the same
generation, choose a **Texturer** in the Post Processing tab: SUPRA-Tex keeps every face (and the picture's print); with Hunyuan3D 2.1
Paint, raise its Textured Max Faces (40,000 faces would smooth away most of the refined detail). SUPRA-Tex Texture and Hunyuan3D 2.1
Paint also texture it later, with the same picture.

## Limits

- Hi3D keeps the overall shape of the coarse mesh: missing or wrong parts, merged limbs or a base plate stay. Fix the coarse mesh
  first.
- Sides hidden in the picture are completed plausibly, not reconstructed.
- The image and the mesh must show the same object; a different picture can add unrelated relief.
- Its detail goes to solid parts. Thin sheets such as leaves, fabric or paper are finer than its grid: where it would perforate or
  lose them, the coarse mesh's surface is kept instead (without added relief).

{_LICENSE_NOTE}
"""

HI3D_PROMPT_INFOS = """
# Hi3D Prompts

The prompt is only the asset name: it names the output files and is stored in the metadata. The refinement follows the Object Image and
the Input Mesh. Example: `Chesterfield armchair, refined`.
"""

HI3D_DEEPY_INFOS = ("Mesh refinement: `image_refs` = one picture of the object (background removed automatically, alpha cutouts kept) + "
                    "`custom_guide` = the coarse mesh (.glb/.obj/.ply/.stl, e.g. a Hunyuan3D 2.1 shape, TripoSG or TRELLIS.2 GLB) -> geometry-only "
                    "GLB with fine detail, same frame and scale as the input mesh. custom_settings: `level` 512 (1 pass, ~0.5 min on an RTX 4090) / "
                    "1024 (default, 2 passes, 1.5-2.5 min) / 1536 (3 passes, ~5-6 min, may run at 1100-1500 on complex objects), `max_faces` "
                    "(0 = 1M, 2M at 1536). `num_inference_steps` = Heun steps per pass (25). No guidance setting. Keeps the coarse structure: "
                    "fix structural errors upstream. Detail goes to solid parts: thin sheets (leaves, fabric, paper) it would perforate keep the coarse surface. Texture in the same task with temporal_upsampling 'supratex' (keeps every face) or "
                    "'hunyuan3d_paint' (raise temporal_postprocessing_parameters.texture_max_faces), or later with supratex_texture / hunyuan3d_21_paint. "
                    "Hi3D Open License 1.0.")

HI3D_DEEPY_PROMPT_INFOS = "`prompt`: asset name only (file names, metadata); the geometry comes from the image and the mesh."
