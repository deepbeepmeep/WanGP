TRIPOSPLAT_INFOS = """
# TripoSplat

TripoSplat (VAST / Tripo, MIT) turns one picture of an object into a 3D Gaussian splat: up to 262,144 small coloured, semi-transparent
3D blobs that together render the object from any angle. Splats look photographic and keep fine, fuzzy or thin details (fur, foliage,
hair, fabric), but they are not a mesh: they suit viewing, previews, AR/VR scenes and splat-capable engines, not 3D printing, UV
texturing or rigging.

## Inputs

- **Object Image** (required): one picture of a single object. Its background is removed automatically, then the object is cropped
  and centered. A whole, uncropped object on a plain background, in a front or three-quarter view with even light, works best. An
  image with a transparent background keeps its own cutout.
- **Asset Name / Description**: names the files and is stored with the settings. TripoSplat only reads the image, so this text does
  not change the result.

## Settings

- **Number of Inference Steps** (20): 10-20 is the recommended range; fewer steps are faster but less detailed.
- **Guidance** (3.0): higher values follow the image more closely and add detail, too high can oversaturate the colours; 1 turns
  guidance off (twice as fast).
- **Shift Scale** (3.0): spends more of the steps on the coarse shape; 1 is a uniform schedule.
- **Gaussian Count**: 256K keeps the most detail; 32K-128K give lighter files (about 68 bytes per Gaussian) that display faster, for
  background props. The model runs once whatever the count: only the final decoding changes.
- **Seed / Number of Samples**: each sample of a batch uses the seed plus its index.

## Output

A Gaussian splat PLY (the standard 3D Gaussian Splatting layout, colours without view dependence, Y up), viewable in the 3D gallery
and in splat viewers and editors such as SuperSplat or SparkJS. The pictured side usually faces +Z (the default front view), but the
model places the object in its own frame and some seeds turn it a quarter turn. Stylised objects, props, toys and creatures work
best; faces of people often come out blurred, and the hidden sides are inferred.

**License:** TripoSplat is MIT (models/triposplat/LICENSE). Its DINOv3 image encoder is under Meta's DINOv3 License (LICENSE_DINOv3.md beside the weights).
"""

TRIPOSPLAT_PROMPT_INFOS = """
# Asset Name

The model builds the 3D Gaussian splat from the Object Image alone. Write a short name of the asset, such
as `Ceramic teapot` or `Toy pirate ship`; it becomes the output file name and is stored with the settings.

To create an asset from text, first generate the object with an image model (one centered object, whole in frame, plain background,
three-quarter view, soft even light), then use that image as the Object Image.
"""

TRIPOSPLAT_DEEPY_INFOS = ("Image to 3D Gaussian splat (PLY, Y up), not a mesh. Needs exactly one reference image (`image_refs`): one whole object "
                          "on a plain background (removed automatically). `prompt` = asset name only, not read by the model. Defaults: 20 steps, "
                          "guidance 3 (1 = off, faster), flow_shift 3. custom_settings: `num_gaussians` 32768/65536/131072/196608/262144 "
                          "(detail vs file size, ~68 bytes per Gaussian). batch_size = candidate splats. Best for stylised objects, props, "
                          "creatures; human faces come out blurred.")

TRIPOSPLAT_DEEPY_PROMPT_INFOS = ("`prompt` (optional): a short asset name such as `Toy pirate ship`; it names the file and does not change the splat, which "
                                 "depends only on the reference image (one centred, uncropped object, plain background, three-quarter view).")

IDOL_INFOS = """
# Idol

Idol by Speridlabs turns one picture of a whole object into a Gaussian splat PLY. The visible side guides the shape and colour.
Idol infers the hidden surfaces, so details on the back may differ from the real object.

## Object Image

Choose **Render or Generated Image** for a tight crop around a synthetic object. Choose **Photograph** to leave more space
around a photographed object. An image with transparency supplies its own cutout. The optional asset name labels the output.

## Settings

- **Gaussian Count** defaults to 256K. Use 32K for lighter files or up to 1M to capture fine detail. Higher counts take longer to decode.
- **Inference Steps** defaults to 50. Lower it for a quicker preview.
- **Guidance** defaults to 4 on WanGP's scale, equivalent to 3 in the Idol SDK. Set it to 1 for unguided sampling.
- **Shift Scale** defaults to 3. Higher values spend more steps forming the coarse shape.

Open the PLY in the 3D gallery or a Gaussian splat viewer. Mesh printing and rigging require a separate mesh reconstruction.

[Idol model card and license](https://huggingface.co/speridlabs/idol)
"""

IDOL_DEEPY_INFOS = ("One object image (`image_refs`) to a Y-up Gaussian splat PLY. Set `model_mode` to `synthetic` for renders or "
                    "`real` for a photograph with a wider crop. The image supplies the shape and colour, hidden surfaces are inferred, "
                    "and `prompt` names the file. Defaults are 50 steps, guidance_scale 4, flow_shift 3. "
                    "`custom_settings.num_gaussians` accepts 32768, 65536, 131072, 196608, 262144 (default), 524288 or 1048576.")

ASSET_NAME_INSTRUCTIONS = """You name 3D assets. Look at the image and name the single main object it shows in 2 to 6 words, for example: Toy
pirate ship, Ceramic teapot with blue flowers, Stone cottage. Keep the words of the user's text when it already names the object. Answer with
the name only, without punctuation, quotes or commentary."""
