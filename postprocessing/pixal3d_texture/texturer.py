# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from postprocessing.supratex.texturer import SupraTexTexturer
from shared.utils.mesh_postprocessing import reference_image


class Pixal3DTexturer(SupraTexTexturer):
    name = "Pixal3D Texture"
    method = model_type = "pixal3d_texture"
    position = 15
    accepts_reference_list = True
    description = "Paints base colour, metallic and roughness from one matching object image or a four-view turnaround."
    reference_label = "Object Image or Four Views"
    reference_help = "One image, four equally framed images (front, right, back, left), or a horizontal sheet of four square views."

    def object_images(self, value):
        if isinstance(value, list):
            return [reference_image(item) for item in value]
        image = reference_image(value)
        return [image] if image is not None else []
