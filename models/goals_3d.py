# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""First level of the model selector for 3D models: what to make, not the model family. Each 3D handler places its model types in one
of these groups (model def "group") and declares the groups it uses (query_family_infos)."""

IMAGE_TO_3D, TEXTURE_MESH, REFINE_MESH, MESH_PARTS, ANIMATE_MESH, RECONSTRUCTION = "image_to_3d", "texture_mesh", "refine_mesh", "mesh_parts", "animate_mesh", "recon3d"
GOALS = {IMAGE_TO_3D: (3010, "Create"), TEXTURE_MESH: (3020, "Texture"), REFINE_MESH: (3030, "Refine"), MESH_PARTS: (3040, "Split"),
         ANIMATE_MESH: (3050, "Animate"), RECONSTRUCTION: (3060, "Reconstruct")}


def family_infos(*goals):
    return {goal: GOALS[goal] for goal in goals}
