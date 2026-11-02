# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), Copyright (c) 2025 Xingkai Yu,
# MIT License (see docs/third_party_licenses/MIT.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Optional, Callable, Any

from ..sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        is_unconditional: bool = False,
        conditional_seq=None,
        prompt_embeds=None,
        prompt_position_ids=None,
        position_offset: int = 0,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.cfg_scale = sampling_params.cfg_scale
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.min_p = sampling_params.min_p
        self.repetition_penalty = sampling_params.repetition_penalty
        self.predictive_penalty = sampling_params.predictive_penalty
        self.repetition_penalty_start = self.num_prompt_tokens
        # Optional repetition-penalty rules (set by callers that know the tokenizer): token ids never penalized nor recorded (symbols, digits,
        # identifier pieces), and the text of the tokens that open or close spans (backticks, double quotes, newlines) inside which the penalty
        # is suspended and nothing is recorded: code, paths and ids are copied, not repeated by mistake.
        self.repetition_exempt_ids: Optional[frozenset] = None
        self.repetition_span_marks: Optional[dict] = None
        # For CFG: mark if this is an unconditional sequence
        self.is_unconditional = is_unconditional
        # For CFG: reference to the corresponding conditional sequence (if this is unconditional)
        # For conditional sequences, this points to the unconditional sequence
        self.paired_seq = conditional_seq  # For conditional seq, points to uncond; for uncond seq, points to cond
        # For constrained decoding: logits processor and state update callback
        self.logits_processor: Optional[Any] = sampling_params.logits_processor
        self.logits_processor_update_state: Optional[Callable[[int], None]] = sampling_params.logits_processor_update_state
        self.logits_bias: Optional[Any] = sampling_params.logits_bias
        self.prompt_embeds = prompt_embeds
        self.prompt_position_ids = prompt_position_ids
        self.position_offset = int(position_offset or 0)
        self.overlay_slot = -1  # mixed-precision KV cache: the sequence's exact-overlay slot (layers.kv_formats), -1 without one
        self.ring_high = 0  # highest position written to the cache + 1: the exact ring holds [ring_high - recent, ring_high)

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def clear_prompt_data(self):
        self.prompt_embeds = None
        self.prompt_position_ids = None

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
