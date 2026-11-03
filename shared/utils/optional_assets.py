# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Optional asset inventory for the model manager; never a generation prerequisite."""
import inspect


def _builtin_entries():
    from shared.prompt_enhancer.assets import checkpoint_inventory as enhancer_inventory
    from shared.deepy.assets import WHISPER_DOWNLOAD_DEFS
    from shared.tinyvae.decoder import REGISTRY
    from preprocessing.processors import depth_checkpoint_inventory, query_preprocessor_download_def
    from preprocessing.depth_anything_v3.assets import query_download_def as depth_v3
    from preprocessing.matanyone.utils.model_assets import MATANYONE_SETTINGS_KEY, MATANYONE_VERSION_LABELS, query_matanyone_download_def
    from preprocessing.speaker_separator.assets import query_speaker_separator_download_def
    from preprocessing.rembg import assets as rembg
    from preprocessing.roformer.assets import query_download_def as roformer
    from models.hyvideo.data_kits.assets import query_download_def as face
    from models.wan.fantasytalking.assets import query_download_def as wav2vec
    from models.wan.multitalk.assets import query_download_def as chinese_wav2vec

    for entry in enhancer_inventory():
        yield {"family": "Deepy / Prompt Enhancer", **entry}
    for name, definition in WHISPER_DOWNLOAD_DEFS.items():
        yield {"family": "Audio Preprocessors", "name": "Whisper", "variant": name, "download_defs": [definition]}
    for kind, label in [("pose", "DWPose"), ("scribble", "Scribble"), ("flow", "RAFT Optical Flow")]:
        yield {"family": "Video Preprocessors", "name": label, "download_defs": [query_preprocessor_download_def(kind)]}
    for entry in depth_checkpoint_inventory():
        yield {"family": "Video Preprocessors", **entry}
    yield {"family": "Video Preprocessors", "name": "Depth Anything", "variant": "3 Large", "download_defs": [depth_v3()]}
    yield {"family": "Video Preprocessors", "name": "Face Detection", "download_defs": [face()]}
    for version, label in MATANYONE_VERSION_LABELS.items():
        model = {"name": "MatAnyone", "variant": label.removeprefix("MatAnyone ")} if label.startswith("MatAnyone ") else {"name": label}
        yield {"family": "Video Preprocessors", **model, "download_defs": [query_matanyone_download_def({MATANYONE_SETTINGS_KEY: version})]}
    for name in rembg.MODELS:
        yield {"family": "Image Preprocessors", "name": "Background Removal", "variant": name, "files": [rembg.query_download_def(name)["filename"]]}
    for name, defs in [("Vocals Separation", [roformer()]), ("Speaker Separation", query_speaker_separator_download_def())]:
        yield {"family": "Audio Preprocessors", "name": name, "download_defs": defs}
    for variant, definition in [("Standard", wav2vec()), ("Chinese", chinese_wav2vec())]:
        yield {"family": "Audio Preprocessors", "name": "Wav2Vec", "variant": variant, "download_defs": [definition]}
    for name, decoder in REGISTRY["decoders"].items():
        yield {"family": "Preview Decoders", "name": decoder["model"], "variant": name, "files": ["preview_decoders/" + decoder["filename"]]}


def _owner_entries(owner):
    query = getattr(owner, "query_checkpoint_inventory", None)
    if query is not None:
        entries = query()
        if entries is not None:
            yield from entries
            return
    query = getattr(owner, "query_download_defs", None) or getattr(owner, "query_download_def", None)
    if query is not None:
        params = inspect.signature(query).parameters
        accepts_enabled = "enabled_only" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs = {"enabled_only": False} if accepts_enabled else {}
        defs = query(**kwargs)
        if defs:
            yield {"download_defs": defs if isinstance(defs, list) else [defs]}


def checkpoint_inventory(plugins, on_error):
    """Read declarations from registered owners without running downloads or loading checkpoints."""
    from postprocessing import spatial_upsamplers, temporal_upsamplers, audio_processors

    for entry in _builtin_entries():
        yield {"owner": "wangp", **entry}
    registries = (
        (spatial_upsamplers.upsampler_handlers(), "query_upsampler_def", ("video", "image")),
        (temporal_upsamplers.registered_temporal_upsamplers(), "query_temporal_upsampler_def", ("video",)),
        (audio_processors.processor_handlers(), "query_audio_processor_def", ("audio",)),
    )
    for handlers, query_def, default_media in registries:
        for handler in handlers:
            try:
                definition = getattr(handler, query_def)()
                media = definition.get("media", default_media)
                kind = "audio" if default_media == ("audio",) else next(kind for kind in ("video", "image", "3d") if kind in media)
                family = f"{kind.upper() if kind == '3d' else kind.title()} Postprocessors"
                for entry in _owner_entries(handler):
                    yield {"family": family, "owner": query_def + ":" + definition["name"], "name": definition["name"], **entry}
            except Exception as exc:
                on_error(f"{type(handler).__name__}: {exc}")
    for plugin_id, plugin in plugins.items():
        try:
            for entry in _owner_entries(plugin):
                yield {"family": "Plugin Extensions", "owner": plugin_id, "name": plugin.name, **entry}
        except Exception as exc:
            on_error(f"{plugin_id}: {exc}")


def query_download_defs(server_config=None):
    """Flat download-definition view retained for plugin callers."""
    def report_error(message):
        raise ValueError(message)
    return [definition for entry in checkpoint_inventory({}, report_error) for definition in entry.get("download_defs", ())]
