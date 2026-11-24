"""MiniMax Music 3 integration for WanGP."""

from shared.startup_imports import lazy_exports

__all__ = [
    "MiniMaxMusic3ConditionEncoder",
    "MiniMaxMusic3RVQDepthDecoder",
    "MiniMaxMusic3Transformer1DModel",
    "MiniMaxMusic3Vocoder",
]

# on first use: the handler's import would otherwise load the model code (and diffusers' model classes)
__getattr__ = lazy_exports(__name__, {"MiniMaxMusic3ConditionEncoder": ".condition_embedder_minimax_music3", "MiniMaxMusic3RVQDepthDecoder": ".minimax_music3_rvq_depth_decoder",
                                      "MiniMaxMusic3Vocoder": ".minimax_music3_vocoder", "MiniMaxMusic3Transformer1DModel": ".transformer_minimax_music3"})
