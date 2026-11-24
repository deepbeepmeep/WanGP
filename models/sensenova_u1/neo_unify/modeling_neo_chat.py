from shared.utils.phase_progress import PhaseProgress, text_encoding_progress, set_phase_status
from typing import List, Optional, Tuple, Union
import math
import os
import torch.utils.checkpoint
from torch import nn
import transformers
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_neo_chat import NEOChatConfig, NEOMoELLMConfig
from .conversation import get_conv_template
from .modeling_neo_vit import NEOVisionModel
from .modeling_qwen3 import Qwen3ForCausalLM, StaticKVCache, create_block_causal_mask
from mmgp import cuda_graphs
from shared.llm_engines.cudagraph_kit import AutoRegressiveCudaGraphKit
from .modeling_qwen3_moe import Qwen3MoeForCausalLM
from .modeling_fm_modules import PositionEmbedding, TimestepEmbedder, FlowMatchingHead, RMSNorm, NerfEmbedder, SimpleMLPAdaLN, ConvDecoder
from .utils import load_image_native, smart_resize, SYSTEM_MESSAGE_FOR_GEN

REVIEW_MAX_PIXELS = 2048 * 2048  # a drawn image is read back at most at the size of the reference images (the understanding branch's range)

logger = logging.get_logger(__name__)


def _take_tensor(tensor_list):
    tensor = tensor_list[0]
    tensor_list.clear()
    return tensor


def _contiguous_disposable(tensor_list):
    tensor = _take_tensor(tensor_list)
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

def optimized_scale(positive_flat, negative_flat):
    # Force the divisor computation to float32 regardless of the surrounding
    # autocast (the squared-norm/division is what we don't want in fp16/bf16).
    # ``device_type`` is taken from the input so this runs equally on CUDA and
    # XPU; ``mps`` is rerouted to ``cpu`` because torch.autocast rejects it.
    device_type = positive_flat.device.type
    if device_type == "mps":
        device_type = "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        positive_flat = positive_flat.float()
        negative_flat = negative_flat.float()

        # Calculate dot production
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

        # st_star = v_cond^T * v_uncond / ||v_uncond||^2
        st_star = dot_product / squared_norm

    return st_star

def build_abs_positions_from_grid_hw(grid_hw: torch.Tensor, device=None):
    """
    Compute patch coordinates (x, y)

    Args:
        grid_hw: (B, 2) tensor representing (H, W) per image
    """
    device = grid_hw.device
    B = grid_hw.shape[0]

    # Get the number of patches per image
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    # Create the batch index for each patch (B x patch count)
    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)  # (N_total,)

    # Generate intra-image patch index (row-major order)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = patch_id_within_image - torch.cumsum(
        torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0
    )[patch_to_sample]

    # Get H/W for each patch according to its image
    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


class NEOChatModel(PreTrainedModel):
    config_class = NEOChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "NEOVisionModel",
        "Qwen3DecoderLayer",
        "Qwen3MoeDecoderLayer",
    ]
    # mmgp loads it only while it runs: the head of written text (Think, self-correction), not while denoising. The token table is looked up on
    # the CPU (sensenova_u1_main: runtime.lookup_on_cpu), never in VRAM: otherwise each written token would also reload its 1.24 GB
    _offload_separate_blocks = ("language_model.lm_head",)

    # support transformers 4.51.+
    _tp_plan = ''

    def __init__(self, config: NEOChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        config.llm_config._attn_implementation = 'eager'

        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = NEOVisionModel(config.vision_config)
            vision_model_mot_gen = NEOVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            # Pick the right backbone class based on the LLM config: dense
            # Qwen3 (DANCE family) or Qwen3-MoE (A3B family). The two share
            # the same NEO-Unify two-branch attention/norm layout, so the
            # rest of this class works against either.
            if isinstance(config.llm_config, NEOMoELLMConfig):
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            else:
                self.language_model = Qwen3ForCausalLM(config.llm_config)

        merge_size = int(1 / self.downsample_ratio)
        output_dim = 3*(patch_size*merge_size)**2
        llm_hidden_size = self.config.llm_config.hidden_size
        self.use_deep_fm_head = self.config.fm_head_layers > 2
        self.use_pixel_head = self.config.use_pixel_head
        if self.use_deep_fm_head:
                fm_head = FlowMatchingHead(llm_hidden_size, output_dim, dim=self.config.fm_head_dim, layers=self.config.fm_head_layers, mlp_ratio=self.config.fm_head_mlp_ratio)
        else:
            fm_head = nn.Sequential(
                    nn.Linear(llm_hidden_size, 4096, bias=True),
                    nn.GELU(),
                    nn.Linear(4096, output_dim, bias=True),
                )

        timestep_embedder = TimestepEmbedder(llm_hidden_size)
        self.fm_modules = nn.ModuleDict(
                    {   
                        "vision_model_mot_gen": vision_model_mot_gen,
                        "timestep_embedder": timestep_embedder,
                        "fm_head": fm_head
                    }
                )

        if self.use_pixel_head:
            self.fm_modules["fm_head"] = ConvDecoder(llm_hidden_size)

        self.concat_time_token_num = config.concat_time_token_num
        self.noise_scale = config.noise_scale
        self.noise_scale_mode = config.noise_scale_mode
        self.noise_scale_base_image_seq_len = config.noise_scale_base_image_seq_len
        self.add_noise_scale_embedding = config.add_noise_scale_embedding
        self.noise_scale_max_value = config.noise_scale_max_value
        self.time_schedule = config.time_schedule
        self.time_shift_type = config.time_shift_type
        self.base_shift = config.base_shift
        self.max_shift = config.max_shift
        self.base_image_seq_len = config.base_image_seq_len
        self.max_image_seq_len = config.max_image_seq_len

        if self.add_noise_scale_embedding:
            noise_scale_embedder = TimestepEmbedder(llm_hidden_size)
            self.fm_modules['noise_scale_embedder'] = noise_scale_embedder


        self.img_context_token_id = None
        self.img_start_token_id = 151670
        self.last_think_content = ""
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

        # Transformers 5 builds composite-model metadata (including
        # ``all_tied_weights_keys``) in ``post_init``.  Without this call a
        # real checkpoint reaches the final loading pass with that metadata
        # missing, even though the nested language model initialized it.
        self.post_init()

    @property
    def device(self):
        return torch.device("cuda") if torch.cuda.is_available() else next(self.parameters()).device

    def preprocess_loras(self, model_type, state_dict):
        prefix = "diffusion_model."
        if any(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state_dict.items()}
        return state_dict

    def _notify_layer_offload_phase(self, phase: str) -> None:
        callback = getattr(self, "_layer_offload_phase_callback", None)
        if callback is not None:
            callback(phase)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        raise NotImplementedError('forward')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def extract_feature(self, pixel_values, gen_model=False, grid_hw=None):
        if gen_model:
            return self.fm_modules['vision_model_mot_gen'](pixel_values=pixel_values, 
                                 output_hidden_states=False, 
                                 return_dict=True, 
                                 grid_hw=grid_hw).last_hidden_state
        else:
            return self.vision_model(pixel_values=pixel_values, 
                                 output_hidden_states=False, 
                                 return_dict=True, 
                                 grid_hw=grid_hw).last_hidden_state

    def extract_feature_list(self, pixel_values_list, gen_model=False, grid_hw=None):
        vision_model = self.fm_modules['vision_model_mot_gen'] if gen_model else self.vision_model
        return vision_model(pixel_values_list=pixel_values_list, output_hidden_states=False, return_dict=True, grid_hw=grid_hw).last_hidden_state

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        raise NotImplementedError('batch_chat')
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses
    
    def patchify(self, images, patch_size, channel_first=False):
        """
        images: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
        x = images.reshape(shape=(images.shape[0], 3, h, patch_size, w, patch_size))

        if channel_first:
            x = torch.einsum('nchpwq->nhwcpq', x)
        else:
            x = torch.einsum('nchpwq->nhwpqc', x)
        
        x = x.reshape(shape=(images.shape[0], h * w, patch_size**2 * 3))
        return x

    def patchify_list(self, images_list, patch_size, channel_first=False):
        return self.patchify(_take_tensor(images_list), patch_size, channel_first)
    
    def unpatchify(sle, x, patch_size, h=None, w=None):
        """
        x: (N, L, patch_size**2 *3)
        images: (N, 3, H, W)
        """
        if h is None or w is None:
            h = w = int(x.shape[1]**.5)
        else:
            h = h // patch_size
            w = w // patch_size        
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        images = x.reshape(shape=(x.shape[0], 3, h * patch_size, w * patch_size))
        return images

    def unpatchify_list(self, x_list, patch_size, h=None, w=None):
        return self.unpatchify(_take_tensor(x_list), patch_size, h, w)
    
    def _euler_step(self, v_pred_list, z, t, t_next):
        v_pred = _take_tensor(v_pred_list)
        v_pred.mul_(t_next - t)
        return z.add_(v_pred)

    def _calculate_dynamic_mu(self, image_seq_len: int) -> float:
        denom = self.max_image_seq_len - self.base_image_seq_len
        if denom == 0:
            return float(self.base_shift)
        m = (self.max_shift - self.base_shift) / denom
        b = self.base_shift - m * self.base_image_seq_len
        return float(image_seq_len) * m + b

    def _apply_time_schedule(self, t: torch.Tensor, image_seq_len: int, timestep_shift: float) -> torch.Tensor:
        self.time_schedule = "standard"
        sigma = 1 - t
        if timestep_shift != 1:
            self.time_schedule = "standard"
        if self.time_schedule == "standard":
            shift = timestep_shift
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
        elif self.time_schedule == "dynamic":
            mu = self._calculate_dynamic_mu(image_seq_len)
            mu_t = t.new_tensor(mu)
            if self.time_shift_type == "exponential":
                shift = torch.exp(mu_t)
                sigma = shift * sigma / (1 + (shift - 1) * sigma)
            elif self.time_shift_type == "linear":
                sigma = mu_t / (mu_t + (1 / sigma - 1))
            else:
                raise ValueError(f"Unsupported time_shift_type: {self.time_shift_type}")
        else:
            raise ValueError(f"Unsupported time_schedule: {self.time_schedule}")
        return 1 - sigma

    def _build_t2i_query(self, prompt_text, system_message=None, append_text=None):
        template = get_conv_template(self.template)
        template.system_message = self.system_message if system_message is None else system_message
        template.append_message(template.roles[0], prompt_text)
        template.append_message(template.roles[1], None)
        if append_text is not None:
            return template.get_prompt() + append_text
        return template.get_prompt()

    def _build_t2i_text_inputs(self, tokenizer, query: str):
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)

        t_idx = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        h_idx = torch.zeros_like(t_idx)
        w_idx = torch.zeros_like(t_idx)
        indexes = torch.stack([t_idx, h_idx, w_idx], dim=0)

        attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}
        return input_ids, indexes, attention_mask
    
    def _build_t2i_image_indexes(self, token_h, token_w, text_len, device):
        t_image = torch.full((token_h * token_w,), text_len, dtype=torch.long, device=device)
        idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)
    
    def _t2i_prefix_forward(self, input_ids, indexes, attention_mask):
        with text_encoding_progress(self.language_model.model.layers):
            out = self.language_model.model(
                input_ids=input_ids,
                indexes=indexes,
                attention_mask=attention_mask,
                use_cache=True,
            )
            result = (out.past_key_values, out.last_hidden_state.device, out.last_hidden_state.dtype)
            del out
            return result

    def _it2i_prefix_forward(self, input_embeds_list, indexes, attention_mask, gen_indicators=None):
        with text_encoding_progress(self.language_model.model.layers):
            out = self.language_model.model(
                inputs_embeds_list=input_embeds_list,
                indexes=indexes,
                attention_mask=attention_mask,
                use_cache=True,
                image_gen_indicators=gen_indicators.view(1, -1) if gen_indicators is not None else None
            )
            result = (out.past_key_values, out.last_hidden_state.device, out.last_hidden_state.dtype)
            del out
            return result

    def _think_prefix_forward(self, **kwargs):
        """Build the Think prefix cache without materialising per-token logits."""
        return self.language_model(use_cache=True, logits_to_keep=1, **kwargs)

    def _append_text_tokens_to_cache(self, cache, t_idx, input_ids):
        if input_ids.shape[1] == 0:
            return t_idx

        device = input_ids.device
        seq_len = input_ids.shape[1]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        t_indexes = torch.arange(t_idx + 1, t_idx + 1 + seq_len, dtype=torch.long, device=device)
        h_indexes = torch.zeros(seq_len, dtype=torch.long, device=device)
        w_indexes = torch.zeros(seq_len, dtype=torch.long, device=device)
        indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

        past_len = cache.get_seq_length()
        mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=device)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = torch.where(causal_mask == 1, 0.0, float('-inf'))
        mask[:, :, :, past_len:] = causal_mask
        attention_mask_dict = {"full_attention": mask}

        self.language_model.model(
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask=attention_mask_dict,
            past_key_values=cache,
            use_cache=True
        )
        return t_idx + seq_len

    def _generate_think(
        self,
        tokenizer,
        prefix_outputs,
        past_key_values,
        t_idx,
        IMG_START_TOKEN,
        max_think_tokens=1024,
    ):
        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        think_end_token_id = tokenizer.convert_tokens_to_ids('</think>')
        think_token_ids = []
        next_token = torch.argmax(prefix_outputs.logits[:, -1, :], dim=-1)

        for _ in range(max_think_tokens):
            token_item = next_token.item()
            if token_item == eos_token_id:
                break
            if token_item == think_end_token_id:
                self.language_model.model.current_index = t_idx
                outputs = self.language_model(
                    input_ids=next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                t_idx += 1
                think_token_ids.append(token_item)
                break

            think_token_ids.append(token_item)

            self.language_model.model.current_index = t_idx
            outputs = self.language_model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            t_idx += 1

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        append_ids = tokenizer(
            '\n\n' + IMG_START_TOKEN,
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids'].to(self.device)
        t_idx = self._append_text_tokens_to_cache(past_key_values, t_idx, append_ids)

        think_text = tokenizer.decode(think_token_ids, skip_special_tokens=False)

        return past_key_values, t_idx, think_text
    
    def _t2i_predict_v(self, input_embeds, indexes_image, attn_mask, past_key_values, t, z, image_token_num, timestep_embeddings=None, image_size=None):
        outputs = self.language_model.model(
            inputs_embeds=input_embeds,
            image_gen_indicators=torch.ones((input_embeds.shape[0], input_embeds.shape[1]), dtype=torch.bool, device=input_embeds.device),
            indexes=indexes_image,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=True,
        )
        hidden_states_list = [outputs.last_hidden_state]
        del outputs
        return self._t2i_hidden_to_v(hidden_states_list, t, z, image_token_num, image_size)

    def _t2i_predict_v_joint(self, input_embeds_list, branches, t, z, image_token_num, image_size=None):
        branch_inputs = [{"indexes": indexes, "attention_mask": attention_mask, "past_key_values": cache} for indexes, attention_mask, cache in branches]
        hidden_states = self.language_model.model.forward_gen_branches(input_embeds_list, branch_inputs)
        outputs = []
        while hidden_states:
            hidden_states_list = [hidden_states.pop(0)]
            outputs.append(self._t2i_hidden_to_v(hidden_states_list, t, z, image_token_num, image_size))
        return outputs

    def _t2i_hidden_to_v(self, hidden_states_list, t, z, image_token_num, image_size):
        hidden_states = _take_tensor(hidden_states_list)
        B, L = z.shape[0], z.shape[1]

        if self.use_pixel_head:
            merge_size = int(1 / self.downsample_ratio)
            token_h = image_size[1] // (self.patch_size * merge_size)
            token_w = image_size[0] // (self.patch_size * merge_size)

            img_2d = hidden_states[:, -image_token_num:].view(B, token_h, token_w, -1).permute(0, 3, 1, 2)
            del hidden_states
            img_2d_list = [img_2d]
            del img_2d
            img_2d = _contiguous_disposable(img_2d_list)
            img_2d_list = [img_2d]
            del img_2d
            smoothed_img_2d = self.fm_modules['fm_head'](img_2d_list)
                
            smoothed_reshaped = smoothed_img_2d.view(B, 3, token_h, self.patch_size * merge_size, token_w, self.patch_size * merge_size).permute(0, 2, 4, 3, 5, 1)
            del smoothed_img_2d
            smoothed_reshaped_list = [smoothed_reshaped]
            del smoothed_reshaped
            x_pred = _contiguous_disposable(smoothed_reshaped_list).view(B, L, self.patch_size * merge_size * self.patch_size * merge_size * 3)
        else:
            if self.use_deep_fm_head:
                x_pred = self.fm_modules["fm_head"](
                hidden_states[:, -image_token_num:].view(B*L, -1), t.expand(B*L)
                ).view(B, L, -1)
            else:
                x_pred = self.fm_modules["fm_head"](
                    hidden_states[:, -image_token_num:].view(B, L, -1)
                ).view(B, L, -1)
            
        
        return x_pred.sub_(z).div_((1 - t).clamp_min(self.config.t_eps))

    def _broadcast_timestep_embedding(self, t, noise_scale, batch_size, token_count):
        embedding = self.fm_modules['timestep_embedder'](t.reshape(1))
        if self.add_noise_scale_embedding:
            normalized_scale = t.new_tensor([noise_scale / self.noise_scale_max_value])
            embedding.add_(self.fm_modules['noise_scale_embedder'](normalized_scale))
        return embedding.view(1, 1, -1).expand(batch_size, token_count, -1)
    
    def _build_it2i_inputs(self, tokenizer, query, pixel_values=None, grid_hw=None):
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)

        indexes = self.get_thw_indexes(input_ids[0], grid_hw)

        attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values, grid_hw=grid_hw)
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)

        return input_embeds, indexes, attention_mask

    @torch.no_grad()
    def interleave_gen_image_only(
            self,
            tokenizer,
            prompt,
            gt_text,
            images=None,
            gt_images=None,
            cfg_scale=1.0,
            img_cfg_scale=1.0,
            cfg_norm='none',
            max_images=10,
            enable_timestep_shift=True,
            timestep_shift=1.0,
            image_size=(256, 256),
            num_steps=30,
            IMG_START_TOKEN='<img>',
            IMG_END_TOKEN='</img>',
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            method='euler',
            cfg_interval=(0, 1),
            t_eps=0.02,
            verbose=False,
            system_message='',
    ):
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.config.t_eps = t_eps

        if isinstance(image_size, tuple):
            image_size_list = [image_size] * max_images
        elif isinstance(image_size, list) and isinstance(image_size[0], tuple):
            image_size_list = image_size
            if len(image_size) < max_images:
                image_size_list += [image_size_list[-1]] * (max_images - len(image_size_list))
        else:
            assert False, "image size should be a tuple or a list of tuple"

        if images is None:
            images =[]

        image_token_count = prompt.count('<image>')
        assert len(images) >= image_token_count
        if len(images) > image_token_count:
            prompt = "<image>\n" * (len(images) - image_token_count) + prompt

        pixel_values =[]
        grid_hw =[]
        for image in images:
            cur_pixel_values, cur_grid_hw = load_image_native(image, self.patch_size, self.downsample_ratio, min_pixels=512*512, max_pixels=min(2048*2048, (4096*4096)//max(1, len(images))), upscale=False)
            grid_hw.append(cur_grid_hw.to(self.device))
            pixel_values.append(cur_pixel_values.to(self.device).to(torch.bfloat16))

        merge_size = int(1 / self.downsample_ratio)
        pv_tensor = torch.cat(pixel_values) if pixel_values else None
        ghw_tensor = torch.cat(grid_hw) if grid_hw else None

        # Condition Initial Cache
        template_cond = get_conv_template(self.template)
        template_cond.system_message = system_message
        template_cond.append_message(template_cond.roles[0], prompt)
        template_cond.append_message(template_cond.roles[1], None)
        query_cond = template_cond.get_prompt() + '<think>\n\n</think>\n\n'

        def replace_image_tokens(query, grid_hw_list):
            for i in range(len(grid_hw_list)):
                num_patch_token = int(grid_hw_list[i][0, 0] * grid_hw_list[i][0, 1] * self.downsample_ratio**2)
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            return query

        query_cond = replace_image_tokens(query_cond, grid_hw)
        input_embeds_cond, indexes_cond, attention_mask_cond = self._build_it2i_inputs(tokenizer, query_cond, pv_tensor, ghw_tensor)
        
        outputs_cond = self.language_model(inputs_embeds=input_embeds_cond, indexes=indexes_cond, attention_mask=attention_mask_cond, use_cache=True)
        past_key_values_cond = outputs_cond.past_key_values
        t_index_cond = indexes_cond[0].max().item()

        # Text Uncondition Cache Initial
        question_text_uncondition = '<image>' * len(images)
        template_tu = get_conv_template(self.template)
        template_tu.system_message = self.system_message
        template_tu.append_message(template_tu.roles[0], question_text_uncondition)
        template_tu.append_message(template_tu.roles[1], None)
        query_text_uncond = template_tu.get_prompt()
        query_text_uncond = replace_image_tokens(query_text_uncond, grid_hw)

        input_embeds_tu, indexes_tu, attention_mask_tu = self._build_it2i_inputs(tokenizer, query_text_uncond, pv_tensor, ghw_tensor)
        outputs_tu = self.language_model(inputs_embeds=input_embeds_tu, indexes=indexes_tu, attention_mask=attention_mask_tu, use_cache=True)
        past_key_values_tu = outputs_tu.past_key_values
        t_index_tu = indexes_tu[0].max().item()

        # Img Uncondition Cache Initial
        query_img_uncond = self._build_t2i_query("", append_text=IMG_START_TOKEN)
        input_embeds_iu, indexes_iu, attention_mask_iu = self._build_it2i_inputs(tokenizer, query_img_uncond)
        outputs_iu = self.language_model(inputs_embeds=input_embeds_iu, indexes=indexes_iu, attention_mask=attention_mask_iu, use_cache=True)
        past_key_values_iu = outputs_iu.past_key_values


        generated_images =[]
        img_count = 0
        device = self.device

        def append_ids_to_cache(cache, t_idx, input_ids):
            if input_ids.shape[1] == 0:
                return t_idx
            seq_len = input_ids.shape[1]
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            t_indexes = torch.arange(t_idx + 1, t_idx + 1 + seq_len, dtype=torch.long, device=device)
            h_indexes = torch.zeros(seq_len, dtype=torch.long, device=device)
            w_indexes = torch.zeros(seq_len, dtype=torch.long, device=device)
            indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
            
            past_len = cache.get_seq_length()
            mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=device)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            causal_mask = torch.where(causal_mask == 1, 0.0, float('-inf'))
            mask[:, :, :, past_len:] = causal_mask
            attention_mask_dict = {"full_attention": mask}
            
            self.language_model(
                inputs_embeds=inputs_embeds,
                indexes=indexes,
                attention_mask=attention_mask_dict,
                past_key_values=cache,
                use_cache=True
            )
            return t_idx + seq_len

        def append_image_to_cache(cache, t_idx, inputs_embeds_img, N_img_tokens, abs_pos_w, abs_pos_h):
            past_len = cache.get_seq_length()
            tgt_len = N_img_tokens + 1
            
            t_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
            t_indexes[:N_img_tokens] = t_idx + 1
            t_indexes[N_img_tokens] = t_idx + 2
            
            h_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
            w_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
            h_indexes[:N_img_tokens] = abs_pos_h
            w_indexes[:N_img_tokens] = abs_pos_w
            
            indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
            
            mask = torch.zeros(1, 1, tgt_len, past_len + tgt_len, device=device)
            mask[0, 0, :N_img_tokens, past_len + N_img_tokens] = float('-inf')
            attention_mask_dict = {"full_attention": mask}
            
            self.language_model(
                inputs_embeds=inputs_embeds_img,
                indexes=indexes,
                attention_mask=attention_mask_dict,
                past_key_values=cache,
                use_cache=True
            )
            return t_idx + 2

        parts = gt_text.split('<image>')
        img_start_id_tensor = torch.tensor([[self.img_start_token_id]], device=device)

        for i, part in enumerate(parts):
            if len(part) > 0:
                if verbose:
                    print(part, end='', flush=True)
                part_ids = tokenizer(part, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
                t_index_cond = append_ids_to_cache(past_key_values_cond, t_index_cond, part_ids)

            if i < len(parts) - 1:
                if img_count >= max_images:
                    break
                    
                if verbose:
                    print("<image>", end='', flush=True)

                t_index_cond = append_ids_to_cache(past_key_values_cond, t_index_cond, img_start_id_tensor)
                t_index_tu = append_ids_to_cache(past_key_values_tu, t_index_tu, img_start_id_tensor)

                cur_image_size = image_size_list[img_count]
                token_h = cur_image_size[1] // (self.patch_size * merge_size)
                token_w = cur_image_size[0] // (self.patch_size * merge_size)

                indexes_image_condition = self._build_t2i_image_indexes(token_h, token_w, t_index_cond + 1, device=device)
                indexes_image_text_uncondition = self._build_t2i_image_indexes(token_h, token_w, t_index_tu + 1, device=device)
                indexes_image_img_uncondition = self._build_t2i_image_indexes(token_h, token_w, indexes_iu[0].max() + 1, device=device)

                grid_h = cur_image_size[1] // self.patch_size
                grid_w = cur_image_size[0] // self.patch_size
                gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device)

                noise_scale = self.noise_scale
                if self.noise_scale_mode in ("resolution", "dynamic", 'dynamic_sqrt'):
                    noise_scale = math.sqrt((grid_h*grid_w)/(merge_size**2) / self.noise_scale_base_image_seq_len)
                    base = float(self.noise_scale_base_image_seq_len)
                    noise_scale = math.sqrt((grid_h*grid_w)/(merge_size**2)/base) * float(self.noise_scale)
                    if self.noise_scale_mode == 'dynamic_sqrt':
                        noise_scale = math.sqrt(noise_scale)
                noise_scale = min(noise_scale, self.noise_scale_max_value)
                image_prediction = noise_scale * torch.randn((1, 3, cur_image_size[1], cur_image_size[0]), device=device, dtype=outputs_cond.logits.dtype)

                past_key_values_cond_cfg = past_key_values_cond
                past_key_values_tu_cfg = past_key_values_tu
                past_key_values_iu_cfg = past_key_values_iu

                # attention_mask_condition = {"full_attention": torch.zeros(1, 1, token_h*token_w, past_key_values_cond.get_seq_length() + token_h*token_w, device=device)}
                # attention_mask_text_uncondition = {"full_attention": torch.zeros(1, 1, token_h*token_w, past_key_values_tu.get_seq_length() + token_h*token_w, device=device)}
                # attention_mask_img_uncondition = {"full_attention": torch.zeros(1, 1, token_h*token_w, past_key_values_iu.get_seq_length() + token_h*token_w, device=device)}
                attention_mask_condition = {"full_attention": None}
                attention_mask_text_uncondition = {"full_attention": None}
                attention_mask_img_uncondition = {"full_attention": None}

                timesteps = torch.linspace(0.0, 1.0, num_steps+1, device=device)
                if enable_timestep_shift:
                    timesteps = self._apply_time_schedule(timesteps, token_h*token_w, timestep_shift)

                step_iter = range(num_steps)
                if verbose:
                    try:
                        from tqdm import tqdm as _tqdm
                        step_iter = _tqdm(
                            step_iter,
                            desc=f"image {img_count + 1} ({image_size[0]}x{image_size[1]})",
                            total=num_steps,
                            leave=False,
                        )
                    except ImportError:
                        pass
                for step_i in step_iter:
                    t = timesteps[step_i]
                    t_next = timesteps[step_i + 1]

                    z = self.patchify(image_prediction, self.patch_size * merge_size)
                    image_input = self.patchify(image_prediction, self.patch_size, channel_first=True)
                    image_embeds = self.extract_feature(image_input.view(1 * grid_h*grid_w, -1), gen_model=True, grid_hw=gen_grid_hw).view(1, token_h*token_w, -1)
                    timestep_embeddings = self._broadcast_timestep_embedding(t, noise_scale, 1, token_h * token_w)
                    image_embeds.add_(timestep_embeddings)

                    use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0
                    out_cond = self._t2i_predict_v(image_embeds, indexes_image_condition, attention_mask_condition, past_key_values_cond_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                    if not use_cfg:
                        v_pred = out_cond
                    elif cfg_scale == 1 and img_cfg_scale == 1:
                        v_pred = out_cond
                    elif img_cfg_scale == 1:
                        out_img_cond = self._t2i_predict_v(image_embeds, indexes_image_text_uncondition, attention_mask_text_uncondition, past_key_values_tu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = out_img_cond + cfg_scale * (out_cond - out_img_cond)
                    elif cfg_scale == img_cfg_scale:
                        out_uncond = self._t2i_predict_v(image_embeds, indexes_image_img_uncondition, attention_mask_img_uncondition, past_key_values_iu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
                    else:
                        out_img_cond = self._t2i_predict_v(image_embeds, indexes_image_text_uncondition, attention_mask_text_uncondition, past_key_values_tu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        out_uncond = self._t2i_predict_v(image_embeds, indexes_image_img_uncondition, attention_mask_img_uncondition, past_key_values_iu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = (
                            out_uncond
                            + cfg_scale * (out_cond - out_img_cond)
                            + img_cfg_scale * (out_img_cond - out_uncond)
                        )
                    if (cfg_scale > 1 or img_cfg_scale > 1) and use_cfg:
                        if cfg_norm == 'global':
                            norm_v_condition = torch.norm(out_cond, dim=(1,2), keepdim=True)
                            norm_v_cfg = torch.norm(v_pred, dim=(1,2), keepdim=True)
                            scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                            v_pred = v_pred * scale
                        elif cfg_norm == 'channel':
                            norm_v_condition = torch.norm(out_cond, dim=-1, keepdim=True)
                            norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                            scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                            v_pred = v_pred * scale

                    z = z + (t_next - t) * v_pred
                    image_prediction = self.unpatchify(z, self.patch_size * merge_size, cur_image_size[1], cur_image_size[0])

                generated_images.append(image_prediction)


                if gt_images is not None and img_count < len(gt_images):
                    gt_img_pil = gt_images[img_count]
                    gt_pixel_values, gt_grid_hw = load_image_native(gt_img_pil, self.patch_size, self.downsample_ratio, min_pixels=512*512, max_pixels=(2048*2048), upscale=False)
                    gt_pixel_values = gt_pixel_values.to(device).to(torch.bfloat16)
                    
                    flatten_pixel_values = gt_pixel_values
                    gen_grid_hw_und = gt_grid_hw
                else:
                    pred_img = image_prediction[0].unsqueeze(0).to(torch.bfloat16)
                    raw_img = pred_img * 0.5 + 0.5
                    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=raw_img.dtype, device=device).view(1, 3, 1, 1)
                    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=raw_img.dtype, device=device).view(1, 3, 1, 1)
                    und_img = (raw_img - img_mean) / img_std
                    
                    c, h, w = und_img[0].shape
                    ps = self.patch_size
                    p_grid_h = h // ps
                    p_grid_w = w // ps
                    flatten_pixel_values = (
                        und_img[0].view(c, p_grid_h, ps, p_grid_w, ps)
                        .permute(1, 3, 0, 2, 4)
                        .reshape(p_grid_h * p_grid_w, c * ps ** 2)
                    )
                    gen_grid_hw_und = torch.tensor([[p_grid_h, p_grid_w]], device=device)

                vit_embeds = self.extract_feature(flatten_pixel_values, grid_hw=gen_grid_hw_und[:1]).unsqueeze(0)
                
                img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
                img_end_embed = self.language_model.get_input_embeddings()(torch.tensor([[img_end_id]], device=device))
                inputs_embeds_img = torch.cat([vit_embeds, img_end_embed], dim=1) # (1, N + 1, C)
                
                N_img_tokens = vit_embeds.shape[1]
                abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(gen_grid_hw_und[:1] // int(1 / self.downsample_ratio), device=device)

                t_index_cond = append_image_to_cache(past_key_values_cond, t_index_cond, inputs_embeds_img, N_img_tokens, abs_pos_w, abs_pos_h)
                t_index_tu = append_image_to_cache(past_key_values_tu, t_index_tu, inputs_embeds_img, N_img_tokens, abs_pos_w, abs_pos_h)

                img_count += 1

        return generated_images

    @torch.no_grad()
    def interleave_gen(
            self,
            tokenizer,
            prompt,
            images=None,
            generation_config=None,
            cfg_scale=1.0,
            img_cfg_scale=1.0,
            cfg_norm='none',
            max_images=10,
            enable_timestep_shift=True,
            timestep_shift=1.0,
            image_size=(256, 256),
            num_steps=30,
            IMG_START_TOKEN='<img>',
            IMG_END_TOKEN='</img>',
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            method='euler',
            cfg_interval=(0, 1),
            t_eps=0.02,
            verbose=False,
            system_message='',
            think_mode=False,
            seed=0,
    ):
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.config.t_eps = t_eps

        if isinstance(image_size, tuple):
            image_size_list = [image_size] * max_images
        elif isinstance(image_size, list) and isinstance(image_size[0], tuple):
            image_size_list = image_size
            if len(image_size) < max_images:
                image_size_list += [image_size_list[-1]] * (max_images - len(image_size_list))
        else:
            assert False, "image size should be a tuple or a list of tuple"

        if generation_config and hasattr(generation_config, 'max_new_tokens') and generation_config.max_new_tokens is not None:
            max_new_tokens = generation_config.max_new_tokens
        else:
            max_new_tokens = 8192

        current_generated_tokens = 0

        if images is None:
            images = []

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        image_token_count = prompt.count('<image>')
        assert len(images) >= image_token_count
        if len(images) > image_token_count:
            prompt = "<image>\n" * (len(images) - image_token_count) + prompt

        pixel_values =[]
        grid_hw =[]
        for image in images:
            cur_pixel_values, cur_grid_hw = load_image_native(image, self.patch_size, self.downsample_ratio, min_pixels=512*512, max_pixels=min(2048*2048, (4096*4096)//max(1, len(images))), upscale=False)
            grid_hw.append(cur_grid_hw.to(self.device))
            pixel_values.append(cur_pixel_values.to(self.device).to(torch.bfloat16))

        merge_size = int(1 / self.downsample_ratio)
        pv_tensor = torch.cat(pixel_values) if pixel_values else None
        ghw_tensor = torch.cat(grid_hw) if grid_hw else None

        # Condition
        template_cond = get_conv_template(self.template)
        template_cond.system_message = system_message
        template_cond.append_message(template_cond.roles[0], prompt)
        template_cond.append_message(template_cond.roles[1], None)
        query_cond = template_cond.get_prompt()

        if not think_mode:
            query_cond = query_cond + '<think>\n\n</think>\n\n'

        def replace_image_tokens(query, grid_hw_list):
            for i in range(len(grid_hw_list)):
                num_patch_token = int(grid_hw_list[i][0, 0] * grid_hw_list[i][0, 1] * self.downsample_ratio**2)
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            return query

        query_cond = replace_image_tokens(query_cond, grid_hw)
        input_embeds_cond, indexes_cond, attention_mask_cond = self._build_it2i_inputs(tokenizer, query_cond, pv_tensor, ghw_tensor)
        
        outputs_cond = self.language_model(inputs_embeds=input_embeds_cond, indexes=indexes_cond, attention_mask=attention_mask_cond, use_cache=True)
        past_key_values_cond = outputs_cond.past_key_values
        t_index_cond = indexes_cond[0].max().item()

        # Initialize Text Uncondition Cache
        question_text_uncondition = '<image>' * len(images)
        template_tu = get_conv_template(self.template)
        template_tu.system_message = self.system_message
        template_tu.append_message(template_tu.roles[0], question_text_uncondition)
        template_tu.append_message(template_tu.roles[1], None)
        query_text_uncond = template_tu.get_prompt()
        query_text_uncond = replace_image_tokens(query_text_uncond, grid_hw)

        input_embeds_tu, indexes_tu, attention_mask_tu = self._build_it2i_inputs(tokenizer, query_text_uncond, pv_tensor, ghw_tensor)
        outputs_tu = self.language_model(inputs_embeds=input_embeds_tu, indexes=indexes_tu, attention_mask=attention_mask_tu, use_cache=True)
        past_key_values_tu = outputs_tu.past_key_values
        t_index_tu = indexes_tu[0].max().item()

        # Initialize Img (ALL) Uncondition Cache
        query_img_uncond = self._build_t2i_query("", append_text=IMG_START_TOKEN)
        input_embeds_iu, indexes_iu, attention_mask_iu = self._build_it2i_inputs(tokenizer, query_img_uncond)
        outputs_iu = self.language_model(inputs_embeds=input_embeds_iu, indexes=indexes_iu, attention_mask=attention_mask_iu, use_cache=True)
        past_key_values_iu = outputs_iu.past_key_values


        generated_text = ""
        generated_images =[]
        max_images = 10
        img_count = 0

        next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)

        generator = torch.Generator(self.device).manual_seed(seed)
        while True:
            # text generation
            gen_tokens = []
            hit_max_tokens = False
            last_decoded = 0
            while True:
                token_item = next_token.item()
                if token_item == eos_token_id or token_item == self.img_start_token_id:
                    break
                gen_tokens.append(token_item)
                current_generated_tokens += 1

                self.language_model.model.current_index = t_index_cond
                outputs_cond = self.language_model(
                    input_ids=next_token.unsqueeze(0),
                    past_key_values=past_key_values_cond,
                    use_cache=True
                )
                past_key_values_cond = outputs_cond.past_key_values
                t_index_cond += 1
                next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)

                # Stream partial text so users see liveness during long runs
                # (e.g. low VRAM offload). Decode in 16-token chunks.
                if verbose and len(gen_tokens) - last_decoded >= 16:
                    partial = tokenizer.decode(gen_tokens[last_decoded:], skip_special_tokens=True)
                    print(partial, end='', flush=True)
                    last_decoded = len(gen_tokens)

                if current_generated_tokens >= max_new_tokens:
                    hit_max_tokens = True
                    break

            if len(gen_tokens) > 0:
                chunk_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                generated_text += chunk_text
                if verbose:
                    remaining = tokenizer.decode(gen_tokens[last_decoded:], skip_special_tokens=True)
                    if remaining:
                        print(remaining, end='', flush=True)

            if next_token.item() == eos_token_id or hit_max_tokens:
                break

            if next_token.item() == self.img_start_token_id:
                if img_count >= max_images:
                    break

                generated_text += "<image>"
                if verbose:
                    print(f"\n[image {img_count + 1}] preparing diffusion...", flush=True)

                # Add the img_start_token for condition and text_uncondition branch
                self.language_model.model.current_index = t_index_cond
                outputs_cond = self.language_model(input_ids=next_token.unsqueeze(0), past_key_values=past_key_values_cond, use_cache=True)
                past_key_values_cond = outputs_cond.past_key_values
                t_index_cond += 1

                self.language_model.model.current_index = t_index_tu
                outputs_tu = self.language_model(input_ids=next_token.unsqueeze(0), past_key_values=past_key_values_tu, use_cache=True)
                past_key_values_tu = outputs_tu.past_key_values
                t_index_tu += 1

                image_size = image_size_list[img_count]
                # Image Generation
                token_h = image_size[1] // (self.patch_size * merge_size)
                token_w = image_size[0] // (self.patch_size * merge_size)
                device = self.device

                indexes_image_condition = self._build_t2i_image_indexes(token_h, token_w, t_index_cond + 1, device=device)
                indexes_image_text_uncondition = self._build_t2i_image_indexes(token_h, token_w, t_index_tu + 1, device=device)
                indexes_image_img_uncondition = self._build_t2i_image_indexes(token_h, token_w, indexes_iu[0].max() + 1, device=device)

                grid_h = image_size[1] // self.patch_size
                grid_w = image_size[0] // self.patch_size
                gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device)

                noise_scale = self.noise_scale
                if self.noise_scale_mode in ("resolution", "dynamic", 'dynamic_sqrt'):
                    base = float(self.noise_scale_base_image_seq_len)
                    noise_scale = math.sqrt((grid_h*grid_w)/(merge_size**2)/base) * float(self.noise_scale)
                    if self.noise_scale_mode == 'dynamic_sqrt':
                        noise_scale = math.sqrt(noise_scale)
                noise_scale = min(noise_scale, self.noise_scale_max_value)
                image_prediction = noise_scale * torch.randn((1, 3, image_size[1], image_size[0]), device=device, dtype=outputs_cond.logits.dtype, generator=generator)

                past_key_values_cond_cfg = past_key_values_cond
                past_key_values_tu_cfg = past_key_values_tu
                past_key_values_iu_cfg = past_key_values_iu

                attention_mask_condition = {"full_attention": None}
                attention_mask_text_uncondition = {"full_attention": None}
                attention_mask_img_uncondition = {"full_attention": None}

                timesteps = torch.linspace(0.0, 1.0, num_steps+1, device=device)
                if enable_timestep_shift:
                    timesteps = self._apply_time_schedule(timesteps, token_h*token_w, timestep_shift)

                step_iter = range(num_steps)
                if verbose:
                    try:
                        from tqdm import tqdm as _tqdm
                        step_iter = _tqdm(
                            step_iter,
                            desc=f"image {img_count + 1} ({image_size[0]}x{image_size[1]})",
                            total=num_steps,
                            leave=False,
                        )
                    except ImportError:
                        pass
                for step_i in step_iter:
                    t = timesteps[step_i]
                    t_next = timesteps[step_i + 1]

                    z = self.patchify(image_prediction, self.patch_size * merge_size)
                    image_input = self.patchify(image_prediction, self.patch_size, channel_first=True)
                    image_embeds = self.extract_feature(image_input.view(1 * grid_h*grid_w, -1), gen_model=True, grid_hw=gen_grid_hw).view(1, token_h*token_w, -1)
                    timestep_embeddings = self._broadcast_timestep_embedding(t, noise_scale, 1, token_h * token_w)
                    image_embeds.add_(timestep_embeddings)

                    use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0
                    out_cond = self._t2i_predict_v(image_embeds, indexes_image_condition, attention_mask_condition, past_key_values_cond_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                    if not use_cfg:
                        v_pred = out_cond
                    elif cfg_scale == 1 and img_cfg_scale == 1:
                        v_pred = out_cond
                    elif img_cfg_scale == 1:
                        out_img_cond = self._t2i_predict_v(image_embeds, indexes_image_text_uncondition, attention_mask_text_uncondition, past_key_values_tu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = out_img_cond + cfg_scale * (out_cond - out_img_cond)
                    elif cfg_scale == img_cfg_scale:
                        out_uncond = self._t2i_predict_v(image_embeds, indexes_image_img_uncondition, attention_mask_img_uncondition, past_key_values_iu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
                    else:
                        out_img_cond = self._t2i_predict_v(image_embeds, indexes_image_text_uncondition, attention_mask_text_uncondition, past_key_values_tu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        out_uncond = self._t2i_predict_v(image_embeds, indexes_image_img_uncondition, attention_mask_img_uncondition, past_key_values_iu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = (
                            out_uncond
                            + cfg_scale * (out_cond - out_img_cond)
                            + img_cfg_scale * (out_img_cond - out_uncond)
                        )
                    if (cfg_scale > 1 or img_cfg_scale > 1 and use_cfg):
                        if cfg_norm == 'global':
                            norm_v_condition = torch.norm(out_cond, dim=(1,2), keepdim=True)
                            norm_v_cfg = torch.norm(v_pred, dim=(1,2), keepdim=True)
                            scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                            v_pred = v_pred * scale
                        elif cfg_norm == 'channel':
                            norm_v_condition = torch.norm(out_cond, dim=-1, keepdim=True)
                            norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                            scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                            v_pred = v_pred * scale

                    z = z + (t_next - t) * v_pred
                    image_prediction = self.unpatchify(z, self.patch_size * merge_size, image_size[1], image_size[0])

                generated_images.append(image_prediction)


                img_count += 1

                # re-encode the generated image using the und-branch
                pred_img = image_prediction[0].unsqueeze(0).to(torch.bfloat16)
                # re-normalize the image
                raw_img = pred_img * 0.5 + 0.5
                img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=raw_img.dtype, device=device).view(1, 3, 1, 1)
                img_std = torch.tensor([0.229, 0.224, 0.225], dtype=raw_img.dtype, device=device).view(1, 3, 1, 1)
                und_img = (raw_img - img_mean) / img_std
                c, h, w = und_img[0].shape
                ps = self.patch_size
                p_grid_h = h // ps
                p_grid_w = w // ps
                flatten_pixel_values = (
                    und_img[0].view(c, p_grid_h, ps, p_grid_w, ps)
                    .permute(1, 3, 0, 2, 4)  # [grid_h, grid_w, c, patch_size, patch_size]
                    .reshape(p_grid_h * p_grid_w, c * ps ** 2)
                )
                vit_embeds = self.extract_feature(flatten_pixel_values, grid_hw=gen_grid_hw[:1]).unsqueeze(0)
                
                img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
                img_end_embed = self.language_model.get_input_embeddings()(torch.tensor([[img_end_id]], device=device))
                inputs_embeds_img = torch.cat([vit_embeds, img_end_embed], dim=1) # (1, N + 1, C)
                
                N_img_tokens = vit_embeds.shape[1]
                abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(gen_grid_hw[:1] // int(1 / self.downsample_ratio), device=device)

                def append_image_to_cache(cache, t_idx):
                    past_len = cache.get_seq_length()
                    tgt_len = N_img_tokens + 1
                    
                    t_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
                    t_indexes[:N_img_tokens] = t_idx + 1
                    t_indexes[N_img_tokens] = t_idx + 2
                    
                    h_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
                    w_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
                    h_indexes[:N_img_tokens] = abs_pos_h
                    w_indexes[:N_img_tokens] = abs_pos_w
                    
                    indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
                    
                    mask = torch.zeros(1, 1, tgt_len, past_len + tgt_len, device=device)
                    mask[0, 0, :N_img_tokens, past_len + N_img_tokens] = float('-inf')
                    attention_mask_dict = {"full_attention": mask}
                    
                    outputs = self.language_model(
                        inputs_embeds=inputs_embeds_img,
                        indexes=indexes,
                        attention_mask=attention_mask_dict,
                        past_key_values=cache,
                        use_cache=True
                    )
                    return outputs, t_idx + 2

                outputs_cond, t_index_cond = append_image_to_cache(past_key_values_cond, t_index_cond)
                outputs_tu, t_index_tu = append_image_to_cache(past_key_values_tu, t_index_tu)

                next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)

        return generated_text, generated_images

    @torch.no_grad()
    def it2i_generate(self, tokenizer, prompt, images, cfg_scale=1, img_cfg_scale=1, cfg_norm='none', enable_timestep_shift=True, timestep_shift=1, image_size=(256, 256), num_steps=30, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', method='euler', cfg_interval=(0, 1), batch_size=1, t_eps=0.02, think_mode=False, seed=0, callback=None):
        assert cfg_norm in ['none', 'global', 'channel']
        self._notify_layer_offload_phase("prefix")

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.config.t_eps = t_eps

        image_token_count = prompt.count('<image>')
        assert len(images) >= image_token_count
        if len(images) > image_token_count:
            if image_token_count == 0 and len(images) > 1:
                prompt = "".join(f"Image-{i + 1}:<image>\n" for i in range(len(images))) + prompt
            else:
                prompt = "<image>\n" * (len(images) - image_token_count) + prompt

        pixel_values = []
        grid_hw = []
        for image in images:
            cur_pixel_values, cur_grid_hw = load_image_native(
                image,
                self.patch_size,
                self.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=min(2048*2048, (4096 * 4096) // len(images)),
                upscale=False,
            )
            cur_grid_hw = cur_grid_hw.to(self.device)
            cur_pixel_values = cur_pixel_values.to(self.device).to(torch.bfloat16)
            pixel_values.append(cur_pixel_values)
            grid_hw.append(cur_grid_hw)
        pixel_values = torch.cat(pixel_values)
        grid_hw = torch.cat(grid_hw)

        merge_size = int(1 / self.downsample_ratio)
        question_condition = f"{prompt}"
        think_text = ""
        needs_cfg = not (cfg_scale == 1 and img_cfg_scale == 1)
        needs_img_condition = needs_cfg and (img_cfg_scale == 1 or cfg_scale != img_cfg_scale)
        needs_uncondition = needs_cfg and img_cfg_scale != 1

        think_content = '<think>\n' if think_mode else '<think>\n\n</think>\n\n' + IMG_START_TOKEN
        query_condition = self._build_t2i_query(question_condition, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)
        query_img_condition = (
            self._build_t2i_query('<image>' * len(images), append_text=IMG_START_TOKEN)
            if needs_img_condition
            else None
        )
        query_uncondition = self._build_t2i_query("", append_text=IMG_START_TOKEN) if needs_uncondition else None

        for i in range(grid_hw.shape[0]):
            num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * self.downsample_ratio**2)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
            query_condition = query_condition.replace('<image>', image_tokens, 1)
            if query_img_condition is not None:
                query_img_condition = query_img_condition.replace('<image>', image_tokens, 1)

        input_embeds_condition, indexes_condition, attention_mask_condition_prefix = self._build_it2i_inputs(
            tokenizer, query_condition, pixel_values, grid_hw
        )
        if query_img_condition is not None:
            input_embeds_img_condition, indexes_img_condition, attention_mask_img_condition_prefix = self._build_it2i_inputs(
                tokenizer, query_img_condition, pixel_values, grid_hw
            )
        else:
            input_embeds_img_condition = indexes_img_condition = attention_mask_img_condition_prefix = None
        if query_uncondition is not None:
            input_embeds_uncondition, indexes_uncondition, attention_mask_uncondition_prefix = self._build_it2i_inputs(
                tokenizer, query_uncondition
            )
        else:
            input_embeds_uncondition = indexes_uncondition = attention_mask_uncondition_prefix = None

        token_h = image_size[1] // (self.patch_size * merge_size)
        token_w = image_size[0] // (self.patch_size * merge_size)

        indexes_image_condition = self._build_t2i_image_indexes(
            token_h, token_w, indexes_condition[0].max() + 1, device=input_embeds_condition.device
        )
        indexes_image_img_condition = (
            self._build_t2i_image_indexes(
                token_h, token_w, indexes_img_condition[0].max() + 1, device=input_embeds_img_condition.device
            )
            if indexes_img_condition is not None
            else None
        )
        indexes_image_uncondition = (
            self._build_t2i_image_indexes(
                token_h, token_w, indexes_uncondition[0].max() + 1, device=input_embeds_uncondition.device
            )
            if indexes_uncondition is not None
            else None
        )

        if think_mode:
            input_device = input_embeds_condition.device
            input_embeds_condition_list = [input_embeds_condition]
            del input_embeds_condition
            outputs_condition = self._think_prefix_forward(
                inputs_embeds_list=input_embeds_condition_list,
                indexes=indexes_condition,
                attention_mask=attention_mask_condition_prefix,
            )
            past_key_values_condition = outputs_condition.past_key_values
            device = outputs_condition.logits.device
            dtype = outputs_condition.logits.dtype
            t_index_condition = indexes_condition[0].max().item()
            past_key_values_condition, t_index_condition, think_text = self._generate_think(
                tokenizer,
                outputs_condition,
                past_key_values_condition,
                t_index_condition,
                IMG_START_TOKEN,
            )
            indexes_image_condition = self._build_t2i_image_indexes(
                token_h, token_w, t_index_condition + 1, device=input_device
            )
            del outputs_condition
        else:
            input_embeds_condition_list = [input_embeds_condition]
            del input_embeds_condition
            past_key_values_condition, device, dtype = self._it2i_prefix_forward(input_embeds_condition_list, indexes_condition, attention_mask_condition_prefix)
        past_key_values_img_condition = None
        if input_embeds_img_condition is not None:
            input_embeds_img_condition_list = [input_embeds_img_condition]
            del input_embeds_img_condition
            past_key_values_img_condition, _, _ = self._it2i_prefix_forward(input_embeds_img_condition_list, indexes_img_condition, attention_mask_img_condition_prefix)
        past_key_values_uncondition = None
        if input_embeds_uncondition is not None:
            input_embeds_uncondition_list = [input_embeds_uncondition]
            del input_embeds_uncondition
            past_key_values_uncondition, _, _ = self._it2i_prefix_forward(input_embeds_uncondition_list, indexes_uncondition, attention_mask_uncondition_prefix)

        del pixel_values, grid_hw
        del indexes_condition, attention_mask_condition_prefix
        if past_key_values_img_condition is not None:
            del indexes_img_condition, attention_mask_img_condition_prefix
        if past_key_values_uncondition is not None:
            del indexes_uncondition, attention_mask_uncondition_prefix
        set_phase_status("Preparing Denoising")
        self._notify_layer_offload_phase("denoise")

        for layer_idx in range(len(past_key_values_condition.layers)):
            past_key_values_condition.layers[layer_idx].keys = past_key_values_condition.layers[layer_idx].keys.expand(
                batch_size, *past_key_values_condition.layers[layer_idx].keys.shape[1:]
            )
            past_key_values_condition.layers[layer_idx].values = past_key_values_condition.layers[layer_idx].values.expand(
                batch_size, *past_key_values_condition.layers[layer_idx].values.shape[1:]
            )
            if past_key_values_img_condition is not None:
                past_key_values_img_condition.layers[layer_idx].keys = past_key_values_img_condition.layers[layer_idx].keys.expand(
                    batch_size, *past_key_values_img_condition.layers[layer_idx].keys.shape[1:]
                )
                past_key_values_img_condition.layers[layer_idx].values = past_key_values_img_condition.layers[layer_idx].values.expand(
                    batch_size, *past_key_values_img_condition.layers[layer_idx].values.shape[1:]
                )
            if past_key_values_uncondition is not None:
                past_key_values_uncondition.layers[layer_idx].keys = past_key_values_uncondition.layers[layer_idx].keys.expand(
                    batch_size, *past_key_values_uncondition.layers[layer_idx].keys.shape[1:]
                )
                past_key_values_uncondition.layers[layer_idx].values = past_key_values_uncondition.layers[layer_idx].values.expand(
                    batch_size, *past_key_values_uncondition.layers[layer_idx].values.shape[1:]
                )

        grid_h = image_size[1] // self.patch_size
        grid_w = image_size[0] // self.patch_size
        grid_hw = torch.tensor([[grid_h, grid_w]] * batch_size, device=device)

        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == 'dynamic_sqrt':
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, self.noise_scale_max_value)
        generator = torch.Generator(device).manual_seed(seed)
        image_prediction = noise_scale * torch.randn(
            (batch_size, 3, image_size[1], image_size[0]), device=device, dtype=dtype, generator=generator
        )

        attention_mask_condition = {"full_attention": None}
        attention_mask_img_condition = {"full_attention": None}
        attention_mask_uncondition = {"full_attention": None}

        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        if enable_timestep_shift:
            timesteps = self._apply_time_schedule(timesteps, token_h * token_w, timestep_shift)

        if callback is not None:
            callback(-1, None)

        for step_i in tqdm(range(num_steps), total=num_steps, desc="SenseNova-U1 denoising"):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]
            use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0

            z = self.patchify(image_prediction, self.patch_size * merge_size)
            image_prediction_list = [image_prediction]
            del image_prediction
            image_input = self.patchify_list(image_prediction_list, self.patch_size, channel_first=True)
            pixel_values_list = [image_input.view(batch_size * grid_h * grid_w, -1)]
            del image_input
            image_embeds = self.extract_feature_list(
                pixel_values_list,
                gen_model=True,
                grid_hw=grid_hw,
            ).view(batch_size, token_h * token_w, -1)
            timestep_embeddings = self._broadcast_timestep_embedding(t, noise_scale, batch_size, token_h * token_w)
            image_embeds.add_(timestep_embeddings)
            del timestep_embeddings
            image_embeds_list = [image_embeds]
            del image_embeds

            if not use_cfg or (cfg_scale == 1 and img_cfg_scale == 1):
                out_cond, = self._t2i_predict_v_joint(
                    image_embeds_list,
                    [(indexes_image_condition, attention_mask_condition, past_key_values_condition)],
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    image_size=image_size,
                )
                v_pred = out_cond
            elif img_cfg_scale == 1:
                out_cond, out_img_cond = self._t2i_predict_v_joint(
                    image_embeds_list,
                    [
                        (indexes_image_condition, attention_mask_condition, past_key_values_condition),
                        (indexes_image_img_condition, attention_mask_img_condition, past_key_values_img_condition),
                    ],
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    image_size=image_size,
                )
                v_pred = out_img_cond.mul_(1 - cfg_scale).add_(out_cond, alpha=cfg_scale)
                del out_img_cond
            elif cfg_scale == img_cfg_scale:
                out_cond, out_uncond = self._t2i_predict_v_joint(
                    image_embeds_list,
                    [
                        (indexes_image_condition, attention_mask_condition, past_key_values_condition),
                        (indexes_image_uncondition, attention_mask_uncondition, past_key_values_uncondition),
                    ],
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    image_size=image_size,
                )
                v_pred = out_uncond.mul_(1 - cfg_scale).add_(out_cond, alpha=cfg_scale)
                del out_uncond
            else:
                out_cond, out_img_cond, out_uncond = self._t2i_predict_v_joint(
                    image_embeds_list,
                    [
                        (indexes_image_condition, attention_mask_condition, past_key_values_condition),
                        (indexes_image_img_condition, attention_mask_img_condition, past_key_values_img_condition),
                        (indexes_image_uncondition, attention_mask_uncondition, past_key_values_uncondition),
                    ],
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    image_size=image_size,
                )
                v_pred = out_uncond.mul_(1 - img_cfg_scale).add_(out_img_cond, alpha=img_cfg_scale - cfg_scale).add_(out_cond, alpha=cfg_scale)
                del out_img_cond, out_uncond
            if (cfg_scale > 1 or img_cfg_scale > 1) and use_cfg:
                if cfg_norm == 'global':
                    norm_v_condition = torch.norm(out_cond, dim=(1, 2), keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred.mul_(scale)
                elif cfg_norm == 'channel':
                    norm_v_condition = torch.norm(out_cond, dim=-1, keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred.mul_(scale)

            del out_cond
            v_pred_list = [v_pred]
            del v_pred
            z = self._euler_step(v_pred_list, z, t, t_next)
            z_list = [z]
            del z
            image_prediction = self.unpatchify_list(z_list, self.patch_size * merge_size, image_size[1], image_size[0])
            if callback is not None:
                callback(step_i, image_prediction)

        self.last_think_content = think_text
        if think_mode:
            return image_prediction, think_text
        return image_prediction

    @torch.no_grad()
    def self_correction_generate(self, tokenizer, prompt, images=None, cfg_scale=4.0, timestep_shift=3.0, image_size=(1024, 1024), num_steps=50, seed=0, callback=None, max_images=4, max_text_tokens=1024,
                                 always_review=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
        """Native self-correction (upstream examples/interleave/self_correction.py: interleave_gen without Think): the model draws the image,
        writes a critique of it (text fixes located by boxes) and draws a revision with the prompt, its draft and the critique in context, and
        may review that revision again, until it ends its answer or max_images were drawn (upstream: 10; each round is a whole generation).
        max_text_tokens: per critique.
        always_review: the model may not end its answer right after the draft (it may otherwise stop with visible errors left: an 8-step 4K
        infographic draft, end token 78-99 % whatever the review size, gave an exact critique and a corrected revision once forced).
        Returns (images in the order drawn, text segments written before each image and after the last)."""
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_start_token_id = img_start_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        eos_id = tokenizer.convert_tokens_to_ids(get_conv_template(self.template).sep.strip())
        self._notify_layer_offload_phase("prefix")
        images = images or []
        pixel_values = grid_hw = None
        if images:
            if len(images) > prompt.count('<image>'):
                prompt = "<image>\n" * (len(images) - prompt.count('<image>')) + prompt
            loaded = [load_image_native(image, self.patch_size, self.downsample_ratio, min_pixels=512 * 512, max_pixels=min(2048 * 2048, (4096 * 4096) // len(images)), upscale=False) for image in images]
            pixel_values = torch.cat([values.to(self.device, torch.bfloat16) for values, _ in loaded])
            grid_hw = torch.cat([hw.to(self.device) for _, hw in loaded])
            del loaded

        def prefix_inputs(query):
            for i in range(0 if grid_hw is None else grid_hw.shape[0]):
                query = query.replace('<image>', IMG_START_TOKEN + IMG_CONTEXT_TOKEN * int(grid_hw[i, 0] * grid_hw[i, 1] * self.downsample_ratio**2) + IMG_END_TOKEN, 1)
            embeds, indexes, mask = self._build_it2i_inputs(tokenizer, query, pixel_values, grid_hw)
            return [embeds], indexes, mask

        # condition: the generation system message, no Think; text-unconditioned branch (CFG): the references only, default system message.
        # Caches sized for a draft and one fix, both read back with a critique each (the model reviews its fix too, then usually ends), grown
        # only for further rounds
        model = self.language_model.model
        factor = self.patch_size * int(1 / self.downsample_ratio)
        review_h, review_w = self._review_size(image_size[1], image_size[0])
        review_tokens = (review_h // factor) * (review_w // factor) + 1  # with </img>
        rotary, rotary_hw = model.layers[0].self_attn.rotary_emb, model.layers[0].self_attn.rotary_emb_hw
        device = self.device
        # the prompts end with <img>, as in t2i_generate / it2i_generate: the draft is always drawn, as the plain generation draws it
        embeds_list, indexes, mask = prefix_inputs(self._build_t2i_query(prompt, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text='<think>\n\n</think>\n\n' + IMG_START_TOKEN))
        cache_cond = StaticKVCache(model.config, indexes.shape[1] + 2 + 2 * review_tokens + 2 * max_text_tokens, rotary_hw, device)
        with text_encoding_progress(model.layers):
            model(inputs_embeds_list=embeds_list, indexes=indexes, attention_mask=mask, past_key_values=cache_cond, use_cache=True)
        t_cond = indexes[0].max().item()
        del embeds_list, indexes, mask
        cache_tu = t_tu = None
        if cfg_scale != 1:
            embeds_list, indexes, mask = prefix_inputs(self._build_t2i_query('<image>' * len(images), append_text=IMG_START_TOKEN))
            t_tu = indexes[0].max().item()
            cache_tu = StaticKVCache(model.config, indexes.shape[1] + 2 + 2 * review_tokens, rotary_hw, device)  # no written text
            with text_encoding_progress(model.layers):
                model(inputs_embeds_list=embeds_list, indexes=indexes, attention_mask=mask, past_key_values=cache_tu, use_cache=True)
            del embeds_list, indexes, mask
        del pixel_values, grid_hw

        # the written tokens of the conditioned branch: graphed decode steps (MMGP stage: one graph per step with the layers in VRAM, else a
        # graph per layer); the text-unconditioned branch only gets the image starts, eagerly
        kit = AutoRegressiveCudaGraphKit("sensenova_self_correction", owner=self, resident=model)
        kit.register_state(cache_cond)  # a no-op when the model has no CUDA graphs

        def write_token(cache, t_idx, token_id, graphed):  # one token into a branch's cache at time position t_idx + 1, its final hidden state
            embeds = self.language_model.get_input_embeddings()(torch.tensor([[token_id]], device="cpu"))  # looked up in RAM
            cos, sin = rotary(embeds, torch.full((1, 1), t_idx + 1, dtype=torch.long, device=device))
            cache.begin_decode()
            with cuda_graphs.scope(gated=False):
                graphed = graphed and cuda_graphs.is_enabled(self)  # the model's graphs (off: eager, not the kit's own graphs)
            hidden = kit.run("decode", lambda e, c, s: model.decode_step(e, c, s, cache), embeds, cos, sin) if graphed else model.decode_step(embeds, cos, sin, cache)
            cache.end_decode()
            return hidden

        def next_logits(hidden):
            return self.language_model.lm_head(hidden)[:, -1]

        generator = torch.Generator(device).manual_seed(seed)
        drawn, segments, token_ids = [], [], []
        progress = bar = None  # the written text's progress (UI bar and abort checks at each token, tqdm), up to max_text_tokens

        def end_writing():
            nonlocal progress, bar
            if progress is not None:
                bar.close()
                progress.__exit__(None, None, None)
                progress = bar = None

        try:
            draw = True  # the draft first: both caches end with <img>
            while True:
                if not draw:
                    token_id = torch.argmax(logits, dim=-1).item()
                    if token_id != img_start_id:
                        if token_id == eos_id or len(token_ids) >= max_text_tokens:
                            break
                        if progress is None:
                            progress = PhaseProgress(max_text_tokens, "Writing Critique", "tokens").__enter__()
                            bar = tqdm(total=max_text_tokens, desc="Writing Critique", unit="tok", leave=False)
                        token_ids.append(token_id)
                        logits = next_logits(write_token(cache_cond, t_cond, token_id, True))
                        t_cond += 1
                        bar.update(1)
                        progress.advance()
                        continue
                    end_writing()
                    write_token(cache_cond, t_cond, token_id, True)
                    t_cond += 1
                    if cache_tu is not None:
                        write_token(cache_tu, t_tu, token_id, False)
                        t_tu += 1
                draw = False
                segments.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())
                token_ids = []
                branches = [(cache_cond, t_cond)] + ([(cache_tu, t_tu)] if cache_tu is not None else [])
                set_phase_status("Drawing Revision" if drawn else "Drawing Draft")
                drawn.append(self._self_correction_denoise(branches, image_size, num_steps, timestep_shift, cfg_scale, generator, callback))
                if len(drawn) == max_images:
                    break
                self._notify_layer_offload_phase("prefix")
                set_phase_status("Reviewing Revision" if len(drawn) > 1 else "Reviewing Draft")
                if cache_cond.reserve(review_tokens + max_text_tokens + 1):  # a further round: this image, a critique, the next image start
                    kit.register_state(cache_cond)
                logits = self._append_drawn_image(cache_cond, t_cond, drawn[-1], img_end_id)
                if always_review and len(drawn) == 1:  # the model may not end its answer right after the draft: it writes its review
                    logits[:, eos_id] = float("-inf")
                t_cond += 2
                if cache_tu is not None:
                    cache_tu.reserve(review_tokens + 1)
                    self._append_drawn_image(cache_tu, t_tu, drawn[-1], img_end_id)
                    t_tu += 2
            end_writing()
        finally:
            if bar is not None:  # aborted while writing
                bar.close()
            kit.release()  # forgets the cache, with the model's graphs that read it
        segments.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())
        return drawn, segments

    def _self_correction_denoise(self, branches, image_size, num_steps, timestep_shift, cfg_scale, generator, callback):
        """Draws one image of the interleaved sequence on the conditioned branch and, with CFG, the text-unconditioned one (both caches end with <img>)."""
        merge_size = int(1 / self.downsample_ratio)
        token_h, token_w = image_size[1] // (self.patch_size * merge_size), image_size[0] // (self.patch_size * merge_size)
        grid_h, grid_w = image_size[1] // self.patch_size, image_size[0] // self.patch_size
        device = self.device
        dtype = branches[0][0].layers[0].keys.dtype
        branches = [(self._build_t2i_image_indexes(token_h, token_w, t_idx + 1, device=device), {"full_attention": None}, cache) for cache, t_idx in branches]
        grid_hw = torch.tensor([[grid_h, grid_w]], device=device)
        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            noise_scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / float(self.noise_scale_base_image_seq_len)) * float(self.noise_scale)
            if self.noise_scale_mode == 'dynamic_sqrt':
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, self.noise_scale_max_value)
        image_prediction = noise_scale * torch.randn((1, 3, image_size[1], image_size[0]), device=device, dtype=dtype, generator=generator)
        timesteps = self._apply_time_schedule(torch.linspace(0.0, 1.0, num_steps + 1, device=device), token_h * token_w, timestep_shift)
        self._notify_layer_offload_phase("denoise")
        if callback is not None:
            callback(-1, None)
        for step_i in tqdm(range(num_steps), total=num_steps, desc="SenseNova-U1 denoising"):
            t, t_next = timesteps[step_i], timesteps[step_i + 1]
            z = self.patchify(image_prediction, self.patch_size * merge_size)
            image_input = self.patchify_list([image_prediction], self.patch_size, channel_first=True)
            del image_prediction
            pixel_values_list = [image_input.view(grid_h * grid_w, -1)]
            del image_input
            image_embeds = self.extract_feature_list(pixel_values_list, gen_model=True, grid_hw=grid_hw).view(1, token_h * token_w, -1)
            image_embeds.add_(self._broadcast_timestep_embedding(t, noise_scale, 1, token_h * token_w))
            image_embeds_list = [image_embeds]
            del image_embeds
            predictions = self._t2i_predict_v_joint(image_embeds_list, branches, t, z, image_token_num=token_h * token_w, image_size=image_size)
            v_pred = predictions[0] if len(predictions) == 1 else predictions[1].mul_(1 - cfg_scale).add_(predictions[0], alpha=cfg_scale)
            del predictions
            z = self._euler_step([v_pred], z, t, t_next)
            del v_pred
            image_prediction = self.unpatchify_list([z], self.patch_size * merge_size, image_size[1], image_size[0])
            del z
            if callback is not None:
                callback(step_i, image_prediction)
        return image_prediction

    def _review_size(self, height, width):
        if height * width <= REVIEW_MAX_PIXELS:
            return height, width
        return smart_resize(height, width, factor=self.patch_size * int(1 / self.downsample_ratio), max_pixels=REVIEW_MAX_PIXELS)

    def _append_drawn_image(self, cache, t_idx, image, img_end_id):
        """Appends a drawn image, re-encoded by the understanding branch and closed by </img>, to a branch's cache (upstream interleave_gen);
        returns the logits of the next token. Above REVIEW_MAX_PIXELS the image is read back downscaled: at 16 MP the model ends its answer
        without reviewing the draft (end token 99.7 % after the draft), at 4 MP it writes its critique."""
        device = image.device
        size = self._review_size(*image.shape[-2:])
        if size != tuple(image.shape[-2:]):
            image = torch.nn.functional.interpolate(image[:1].float(), size=size, mode="bicubic", antialias=True)
        und = image[0].to(torch.bfloat16) * 0.5 + 0.5
        und = (und - torch.tensor([0.485, 0.456, 0.406], dtype=und.dtype, device=device).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], dtype=und.dtype, device=device).view(3, 1, 1)
        channels, height, width = und.shape
        ps = self.patch_size
        pixel_values = und.view(channels, height // ps, ps, width // ps, ps).permute(1, 3, 0, 2, 4).reshape((height // ps) * (width // ps), channels * ps * ps)
        del und
        grid_hw = torch.tensor([[height // ps, width // ps]], device=device)
        vit_embeds = self.extract_feature(pixel_values, grid_hw=grid_hw).unsqueeze(0)
        count = vit_embeds.shape[1]
        embeds = torch.cat([vit_embeds, self.language_model.get_input_embeddings()(torch.tensor([[img_end_id]], device=device))], dim=1)
        del vit_embeds
        abs_w, abs_h = build_abs_positions_from_grid_hw(grid_hw // int(1 / self.downsample_ratio), device=device)
        t_indexes = torch.full((count + 1,), t_idx + 1, dtype=torch.long, device=device)
        t_indexes[count] = t_idx + 2
        h_indexes = torch.zeros(count + 1, dtype=torch.long, device=device)
        w_indexes = torch.zeros(count + 1, dtype=torch.long, device=device)
        h_indexes[:count], w_indexes[:count] = abs_h, abs_w
        past = cache.get_seq_length()
        mask = torch.zeros(1, 1, count + 1, past + count + 1, device=device)
        mask[0, 0, :count, past + count] = float('-inf')  # the image tokens do not see </img>
        return self.language_model(inputs_embeds=embeds, indexes=torch.stack([t_indexes, h_indexes, w_indexes]), attention_mask={"full_attention": mask}, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]

    @torch.no_grad()
    def t2i_generate(self, tokenizer, prompt, cfg_scale=1, timestep_shift=1, enable_timestep_shift=True, cfg_norm='none', image_size=(256, 256), num_steps=30, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', method='euler', cfg_interval=(0, 1), batch_size=1, t_eps=0.02, think_mode=False, seed=0, callback=None):
        assert self.concat_time_token_num == 0
        assert cfg_norm in ['cfg_zero_star', 'global', 'none', 'channel']
        self._notify_layer_offload_phase("prefix")
        merge_size = int(1 / self.downsample_ratio)

        self.config.t_eps = t_eps
        # question_condition = f"Please generate an image based on the following description: {prompt}"
        question_condition = f"{prompt}"
        # question_condition += f"\nThe resolution of the image should be {image_size}"

        think_text = ""
        needs_cfg = cfg_scale > 1

        think_content = '<think>\n' if think_mode else '<think>\n\n</think>\n\n' + IMG_START_TOKEN
        query_condition = self._build_t2i_query(question_condition, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)
        query_uncondition = self._build_t2i_query("", append_text=IMG_START_TOKEN) if needs_cfg else None

        input_ids_condition, indexes_condition, attention_mask_condition_prefix = self._build_t2i_text_inputs(tokenizer, query_condition)
        if query_uncondition is not None:
            input_ids_uncondition, indexes_uncondition, attention_mask_uncondition_prefix = self._build_t2i_text_inputs(tokenizer, query_uncondition)
        else:
            input_ids_uncondition = indexes_uncondition = attention_mask_uncondition_prefix = None
       
        token_h = image_size[1] // (self.patch_size * merge_size)
        token_w = image_size[0] // (self.patch_size * merge_size)

        indexes_image_condition = self._build_t2i_image_indexes(token_h, token_w, indexes_condition.shape[1], device=input_ids_condition.device)
        indexes_image_uncondition = (
            self._build_t2i_image_indexes(token_h, token_w, indexes_uncondition.shape[1], device=input_ids_uncondition.device)
            if indexes_uncondition is not None
            else None
        )

        if think_mode:
            outputs_condition = self._think_prefix_forward(
                input_ids=input_ids_condition,
                indexes=indexes_condition,
                attention_mask=attention_mask_condition_prefix,
            )
            past_key_values_condition = outputs_condition.past_key_values
            device = outputs_condition.logits.device
            dtype = outputs_condition.logits.dtype
            t_index_condition = indexes_condition[0].max().item()
            past_key_values_condition, t_index_condition, think_text = self._generate_think(
                tokenizer,
                outputs_condition,
                past_key_values_condition,
                t_index_condition,
                IMG_START_TOKEN,
            )
            indexes_image_condition = self._build_t2i_image_indexes(
                token_h, token_w, t_index_condition + 1, device=input_ids_condition.device
            )
            del outputs_condition
        else:
            past_key_values_condition, device, dtype = self._t2i_prefix_forward(input_ids_condition, indexes_condition, attention_mask_condition_prefix)
        past_key_values_uncondition = None
        if input_ids_uncondition is not None:
            past_key_values_uncondition, _, _ = self._t2i_prefix_forward(input_ids_uncondition, indexes_uncondition, attention_mask_uncondition_prefix)

        del input_ids_condition, indexes_condition, attention_mask_condition_prefix
        if input_ids_uncondition is not None:
            del input_ids_uncondition, indexes_uncondition, attention_mask_uncondition_prefix
        set_phase_status("Preparing Denoising")
        self._notify_layer_offload_phase("denoise")

        for layer_idx in range(len(past_key_values_condition.layers)):
            past_key_values_condition.layers[layer_idx].keys = past_key_values_condition.layers[layer_idx].keys.expand(batch_size, *past_key_values_condition.layers[layer_idx].keys.shape[1:])
            past_key_values_condition.layers[layer_idx].values = past_key_values_condition.layers[layer_idx].values.expand(batch_size, *past_key_values_condition.layers[layer_idx].values.shape[1:])
            if past_key_values_uncondition is not None:
                past_key_values_uncondition.layers[layer_idx].keys = past_key_values_uncondition.layers[layer_idx].keys.expand(batch_size, *past_key_values_uncondition.layers[layer_idx].keys.shape[1:])
                past_key_values_uncondition.layers[layer_idx].values = past_key_values_uncondition.layers[layer_idx].values.expand(batch_size, *past_key_values_uncondition.layers[layer_idx].values.shape[1:])

        # init noise image tokens
        grid_h = image_size[1] // self.patch_size
        grid_w = image_size[0] // self.patch_size
        grid_hw = torch.tensor([[grid_h, grid_w]]*batch_size, device=device)

        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", 'dynamic_sqrt'):
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h*grid_w)/(merge_size**2)/base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == 'dynamic_sqrt':
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, self.noise_scale_max_value)
        generator = torch.Generator(device).manual_seed(seed)
        image_prediction = noise_scale * torch.randn((batch_size, 3, image_size[1], image_size[0]), device=device, dtype=dtype, generator=generator)

        attention_mask_condition = {"full_attention": None}
        attention_mask_uncondition = {"full_attention": None}

        timesteps = torch.linspace(0.0, 1.0, num_steps+1, device=device)
        if enable_timestep_shift:
            timesteps = self._apply_time_schedule(timesteps, token_h*token_w, timestep_shift)

        if callback is not None:
            callback(-1, None)

        for step_i in tqdm(range(num_steps), total=num_steps, desc="SenseNova-U1 denoising"):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]

            z = self.patchify(image_prediction, self.patch_size * merge_size)
            image_prediction_list = [image_prediction]
            del image_prediction
            image_input = self.patchify_list(image_prediction_list, self.patch_size, channel_first=True)
            pixel_values_list = [image_input.view(batch_size * grid_h * grid_w, -1)]
            del image_input
            image_embeds = self.extract_feature_list(pixel_values_list, gen_model=True, grid_hw=grid_hw).view(batch_size, token_h * token_w, -1)
            timestep_embeddings = self._broadcast_timestep_embedding(t, noise_scale, batch_size, token_h * token_w)
            image_embeds.add_(timestep_embeddings)
            del timestep_embeddings
            image_embeds_list = [image_embeds]
            del image_embeds

            cfg_active = t >= cfg_interval[0] and t <= cfg_interval[1] and cfg_scale > 1
            if cfg_active:
                v_pred_condition, v_pred_uncondition = self._t2i_predict_v_joint(
                    image_embeds_list,
                    [
                        (indexes_image_condition, attention_mask_condition, past_key_values_condition),
                        (indexes_image_uncondition, attention_mask_uncondition, past_key_values_uncondition),
                    ],
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    image_size=image_size,
                )
                if cfg_norm == 'cfg_zero_star':
                    positive_flat = v_pred_condition.view(batch_size, -1)  
                    negative_flat = v_pred_uncondition.view(batch_size, -1)  

                    alpha = optimized_scale(positive_flat,negative_flat)
                    alpha = alpha.view(batch_size, *([1] * (len(v_pred_condition.shape) - 1)))
                    alpha = alpha.to(positive_flat.dtype)
                    del positive_flat, negative_flat

                    if (step_i <= 0):
                        v_pred = v_pred_condition.mul_(0)
                    else:
                        v_pred = v_pred_uncondition.mul_(alpha).mul_(1 - cfg_scale).add_(v_pred_condition, alpha=cfg_scale)
                else: 
                    v_pred = v_pred_uncondition.mul_(1 - cfg_scale).add_(v_pred_condition, alpha=cfg_scale)
                    if cfg_norm == 'global':
                        norm_v_condition = torch.norm(v_pred_condition, dim=(1,2), keepdim=True)
                        norm_v_cfg = torch.norm(v_pred, dim=(1,2), keepdim=True)
                        scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                        v_pred.mul_(scale)
                    elif cfg_norm == 'channel':
                        norm_v_condition = torch.norm(v_pred_condition, dim=-1, keepdim=True)
                        norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                        scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                        v_pred.mul_(scale)
                del v_pred_condition, v_pred_uncondition
            else:
                v_pred, = self._t2i_predict_v_joint(
                    image_embeds_list,
                    [(indexes_image_condition, attention_mask_condition, past_key_values_condition)],
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    image_size=image_size,
                )

            v_pred_list = [v_pred]
            del v_pred
            z = self._euler_step(v_pred_list, z, t, t_next)
            z_list = [z]
            del z
            image_prediction = self.unpatchify_list(z_list, self.patch_size * merge_size, image_size[1], image_size[0])
            if callback is not None:
                callback(step_i, image_prediction)

        self.last_think_content = think_text
        if think_mode:
            return image_prediction, think_text
        return image_prediction

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False, grid_hw=None, 
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            print(f'dynamic image size: {grid_hw[0] * self.patch_size}')

        for i in range(grid_hw.shape[0]):
            num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * self.downsample_ratio**2)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            grid_hw=grid_hw,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            grid_hw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert input_ids.shape[0] == 1
        assert self.img_context_token_id is not None
        indexes = self.get_thw_indexes(input_ids[0], grid_hw)
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values, grid_hw=grid_hw)
        
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
    
    def get_thw_indexes(self, input_ids, grid_hw=None):
        img_start_shift = torch.cat([torch.zeros(1, dtype=torch.long).to(input_ids.device), 
                                     (input_ids == self.img_start_token_id).long()], dim=0)[:-1]
        not_img_token = (input_ids != self.img_context_token_id).long()
        t_indexes = ((img_start_shift + not_img_token).cumsum(0) - 1)
        h_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)
        w_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)

        if grid_hw is not None:
            selected = (input_ids == self.img_context_token_id)
            if selected.long().sum() > 0:
                abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
                    grid_hw // int(1 / self.downsample_ratio), device=t_indexes.device)
                h_indexes[selected] = abs_pos_h.to(t_indexes.device, t_indexes.dtype)
                w_indexes[selected] = abs_pos_w.to(t_indexes.device, t_indexes.dtype)
        return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
