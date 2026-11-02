"""Model modes and tuning settings of the reconstruction architectures, shared by the handler (UI declarations) and the pipeline."""

MOGE2_MODES = {"choices": [("Point Cloud (Gaussian Splats)", "points"), ("Mesh (Textured GLB)", "mesh")], "default": "points", "label": "Output"}

MAX_POINTS = {"id": "max_points", "label": "Max Points (a random subset is kept above it)", "name": "Max Points", "type": "int", "default": 1000000,
              "min": 10000, "max": 4000000, "inc": 10000}
MAX_FRAMES = {"id": "max_frames", "label": "Video Frames Used (evenly spaced over the video)", "name": "Video Frames", "type": "int", "default": 16,
              "min": 2, "max": 64, "inc": 1}

CUSTOM_SETTINGS = {
    "moge2": [
        {"id": "max_size", "label": "Working Resolution (longest side of the photo, larger photos are reduced)", "name": "Working Resolution",
         "type": "dropdown", "default": 1024, "choices": [("512", 512), ("768", 768), ("1024 (Recommended)", 1024), ("1536", 1536), ("2048", 2048)]},
        {"id": "detail", "label": "Inference Detail (more tokens: finer geometry, slower)", "name": "Inference Detail", "type": "dropdown", "default": 9,
         "choices": [("Low", 0), ("Medium", 5), ("High (Recommended)", 9), ("Ultra", 30)]},
        {"id": "edges", "label": "Depth Discontinuities", "name": "Depth Discontinuities", "type": "dropdown", "default": "remove",
         "choices": [("Removed (no stretched surfaces between objects)", "remove"), ("Kept", "keep")]},
        dict(MAX_POINTS, model_mode=["points"]),  # a mesh keeps every point
    ],
    "mapanything": [
        MAX_POINTS,
        MAX_FRAMES,
        {"id": "confidence", "label": "Low-Confidence Filter (drops this share of the least confident pixels of each view)", "name": "Confidence Filter",
         "type": "dropdown", "default": 0, "choices": [("Off", 0), ("10%", 10), ("25%", 25), ("50%", 50)]},
    ],
    "da3_points": [
        MAX_POINTS,
        dict(MAX_FRAMES, max=128),
        {"id": "confidence", "label": "Low-Confidence Filter (drops about this share of the least confident pixels)", "name": "Confidence Filter",
         "type": "dropdown", "default": 40, "choices": [("Off", 0), ("10%", 10), ("25%", 25), ("40% (Recommended)", 40), ("60%", 60)]},
        {"id": "process_res", "label": "Working Resolution (longest side)", "name": "Working Resolution", "type": "dropdown", "default": 504,
         "choices": [("504 (Recommended)", 504), ("756", 756), ("1008", 1008)]},
    ],
}

DEFAULT_CUSTOM_SETTINGS = {architecture: {one["id"]: one["default"] for one in settings} for architecture, settings in CUSTOM_SETTINGS.items()}
