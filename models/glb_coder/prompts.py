# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

# The released CCORE workflow generates a complete Three.js page with a GLB export action.
SYSTEM_PROMPT = """Create a good-quality low-poly game asset for Godot from the user's request. Write one complete standalone HTML file using Three.js. Display the asset and include a button labelled Export GLB which downloads a binary GLB using GLTFExporter. If the user requests animation, include its AnimationClips in the exported GLB. Otherwise keep the asset static.
Use ES module imports from 'three' and 'three/addons/'. The host provides the import map. The page runs offline.
Build the asset in one THREE.Group. Use short reusable geometry helpers that add each part directly to its parent. Add the group to the scene once and pass that same group directly to GLTFExporter. Keep the preview controls and HTML concise.
For a moving part, give it a unique Object3D.name and target that exact name in its keyframe tracks. For a full turn use unit quaternions from setFromAxisAngle at quarter-turn intervals. Play the clip with AnimationMixer and export the same clip with exporter.parseAsync(asset, {binary: true, animations: [clip]}). Return only the complete HTML."""

ENHANCER = """Turn the request into a concise description for a low-poly game asset or small scene. Preserve the user's language, subject, exact wording, exclusions and requested motion. Clarify shapes, parts, placement, materials and colours where helpful. Return only the asset description."""

INFOS = """### Create a 3D Asset
Describe a game prop, character or small scene. WanGP adds the GLB and a still preview to the 3D gallery. For an animated asset, name the moving part and its motion. An exported animation clip travels with the GLB into a viewer or editor such as Godot.

### Reference Image
Choose **Text and Reference Image** under **Input** to load the vision encoder, then supply one image and describe the asset to build. The image guides its appearance.

### Context and Memory
**KV Cache Size** sets the total space for the description, reference image and generated scene code. The default is 64,000 tokens. Increase it if the model runs out of space before finishing. **KV Cache Quantization** defaults to INT8. Q6/Q4 and Q4 reduce memory use for long scenes.

### Export
Export needs Edge, Chrome or Chromium installed locally. If the generated page fails, WanGP tries simple syntax and Three.js corrections, then gives the model one chance to fix the remaining error. If motion matters, play the saved GLB to check the clip.

Turn on **Save Three.js Source** to keep the editable HTML that produced the GLB. A batch gets one ZIP containing the source for each asset. To run the HTML locally, serve it over HTTP beside a copy of WanGP’s `shared/apps/lib/three` directory."""

PROMPT_INFOS = """### Describe the Asset
Short English requests work well. Name the subject, silhouette, parts and materials. For a small scene, say where each object belongs. With a reference image, describe which features to keep.

- **Static prop.** A low-poly treasure chest with dark oak panels, two brass bands and a large brass lock.
- **With a reference.** Keep the watchtower’s square footprint and red roof from the picture. Add a wooden ladder on the left.
- **Animated prop.** A cream windmill with four blue sails. The sails turn together around the hub once every four seconds in a seamless loop. Keep the tower still."""

DEEPY_INFOS = """- Generate a low-poly GLB in the 3D gallery from `prompt` with `image_mode=1`. Plain text needs no media. For one reference image, use `config=",vision"`, `video_prompt_type="I"` and one item in `image_refs`.
- Request motion in `prompt`. Verify the exported animation with `render_3d_views` at different `frame_no` values. Failed scene exports get a syntax check and one model repair attempt. Set `custom_settings.save_source=1` for the successful HTML side file (a ZIP for batches). It defaults to 0.
- `custom_settings.kv_cache_layout` accepts `int8` (default), `q6_q4` or `q4`. `custom_settings.kv_cache_size` covers prompt, reference and output together. The default is 64000 tokens, with a range of 1024 to 262144."""
DEEPY_PROMPT_INFOS = """Use a short asset brief in `prompt`, preferably in English. Specify visible parts, colours, materials and their placement. With `image_refs`, name the features to keep.
- Static example: “A low-poly treasure chest with dark oak panels, two brass bands and a large brass lock.”
- For animation, name the moving group, pivot, timing and what stays still. Example: “Four blue windmill sails rotate around their hub once every four seconds in a seamless loop. Keep the tower still.”"""
