from shared.startup_imports import lazy_exports

__all__ = [
    'WanVAE',
    'WanModel',
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
    'HuggingfaceTokenizer',
    'pay_attention',
]

# on first use: importing one submodule (posemb_layers for SCAIL 2's handler) would otherwise load the whole model code
__getattr__ = lazy_exports(__name__, {'pay_attention': 'shared.attention', 'WanModel': '.model', 'T5Decoder': '.t5', 'T5Encoder': '.t5', 'T5EncoderModel': '.t5', 'T5Model': '.t5',
                                      'HuggingfaceTokenizer': '.tokenizers', 'WanVAE': '.vae', 'Wan2_2_VAE': '.vae2_2'})
