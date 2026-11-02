# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""FLUX.2 Klein projection aliases; naming conversion stays in the shared mapper."""
from shared.utils.lora_mapping import convert_lora_keys


def convert_klein_lora(model, state_dict):
    names = set(dict(model.named_modules()))
    split_map = model.split_linear_modules_map
    for name in tuple(names):
        for fused, spec in split_map.items():
            for split in spec['mapped_modules']:
                if name.endswith('.' + split):
                    names.add(name[:-len(split)] + fused)
    root = {'img_in': 'x_embedder', 'txt_in': 'context_embedder', 'final_layer.linear': 'proj_out',
            'final_layer.adaLN_modulation.1': 'norm_out.linear',
            'time_in.in_layer': 'time_guidance_embed.timestep_embedder.linear_1',
            'time_in.out_layer': 'time_guidance_embed.timestep_embedder.linear_2',
            'double_stream_modulation_img.lin': 'double_stream_modulation_img.linear',
            'double_stream_modulation_txt.lin': 'double_stream_modulation_txt.linear',
            'single_stream_modulation.lin': 'single_stream_modulation.linear'}
    double = {'img_attn.q': 'attn.to_q', 'img_attn.k': 'attn.to_k', 'img_attn.v': 'attn.to_v',
              'txt_attn.q': 'attn.add_q_proj', 'txt_attn.k': 'attn.add_k_proj', 'txt_attn.v': 'attn.add_v_proj',
              'img_attn.proj': 'attn.to_out.0', 'txt_attn.proj': 'attn.to_add_out',
              'img_mlp.0': 'ff.linear_in', 'img_mlp.2': 'ff.linear_out',
              'txt_mlp.0': 'ff_context.linear_in', 'txt_mlp.2': 'ff_context.linear_out'}
    single = {'linear1': 'attn.to_qkv_mlp_proj', 'linear2': 'attn.to_out',
              'linear1_attn_q': 'attn.to_q', 'linear1_attn_k': 'attn.to_k', 'linear1_attn_v': 'attn.to_v',
              'linear1_mlp': 'proj_mlp'}
    aliases = {alias: name for name, alias in root.items()}
    for name in names:
        if name.startswith(('double_blocks.', 'single_blocks.')) and name.count('.') >= 2:
            tower, index, child = name.split('.', 2)
            mapping = double if tower == 'double_blocks' else single
            if child in mapping:
                target = 'transformer_blocks' if tower == 'double_blocks' else 'single_transformer_blocks'
                aliases[f'{target}.{index}.{mapping[child]}'] = name
    normalized = convert_lora_keys(state_dict, names | aliases.keys(), split_linear_modules_map=split_map)
    mapped = {}
    for key, value in normalized.items():
        if not key.startswith('diffusion_model.'):
            mapped[key] = value
            continue
        module, suffix = key[len('diffusion_model.'):].split('.lora_', 1) if '.lora_' in key else (None, None)
        if module is None:
            module, suffix = key[len('diffusion_model.'):].rsplit('.', 1)
            suffix = '.' + suffix
        else:
            suffix = '.lora_' + suffix
        if module == 'norm_out.linear' and suffix == '.lora_B.weight':
            value = value.reshape(2, -1, *value.shape[1:]).flip(0).flatten(0, 1)
        target = 'diffusion_model.' + aliases.get(module, module) + suffix
        if target in mapped:
            raise ValueError(f'LoRA keys collide at {target}')
        mapped[target] = value
    return convert_lora_keys(mapped, names, split_linear_modules_map=split_map)
