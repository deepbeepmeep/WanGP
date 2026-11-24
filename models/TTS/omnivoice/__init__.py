from shared.startup_imports import lazy_exports

__all__ = ["OmniVoice", "OmniVoiceConfig", "OmniVoiceGenerationConfig", "VoiceClonePrompt"]

# on first use: the handler's import of the settings module would otherwise load the model code (transformers' generation code)
__getattr__ = lazy_exports(__name__, {name: ".modeling_omnivoice" for name in __all__})
