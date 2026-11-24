# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Hashable

import torch
from shared.utils.phase_progress import text_encoding_prompts


@dataclass
class _CacheEntry:
    value: Any
    size_bytes: int


class TextEncoderCache:
    def __init__(self, max_size_mb: float = 100) -> None:
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._entries: "OrderedDict[Hashable, _CacheEntry]" = OrderedDict()
        self._size_bytes = 0

    def encode(
        self,
        encode_fn: Callable[[list[str]], list[Any]],
        prompts: Iterable[str] | str,
        device: torch.device | str | None = None,
        parallel: bool = False,
        cache_keys: Iterable[Hashable] | Hashable | None = None,
    ) -> list[Any]:
        if isinstance(prompts, str):
            prompts_list = [prompts]
        else:
            prompts_list = list(prompts)
        if not prompts_list:
            return []
        if cache_keys is None:
            keys_list = prompts_list
        else:
            if len(prompts_list) == 1 and not isinstance(cache_keys, list):
                keys_list = [cache_keys]
            else:
                keys_list = list(cache_keys)
            if len(keys_list) != len(prompts_list):
                raise ValueError("cache_keys must match the number of prompts.")

        if not parallel:
            with text_encoding_prompts(sum(key not in self._entries for key in dict.fromkeys(keys_list)), inherit=True):
                results: list[Any] = []
                for prompt, cache_key in zip(prompts_list, keys_list):
                    cached = self._entries.get(cache_key)
                    if cached is not None:
                        self._entries.move_to_end(cache_key)
                        results.append(self._to_device(cached.value, device))
                        continue
                    encoded = encode_fn([prompt])
                    if isinstance(encoded, (list, tuple)):
                        if not encoded:
                            raise ValueError("encode_fn returned empty embeddings.")
                        encoded_item = encoded[0]
                    else:
                        encoded_item = encoded
                    results.append(self._store(cache_key, encoded_item, device))
                return results

        results = [None] * len(prompts_list)
        missing_prompts: list[str] = []
        missing_indices: list[int] = []
        missing_keys: list[Hashable] = []

        for idx, (prompt, cache_key) in enumerate(zip(prompts_list, keys_list)):
            cached = self._entries.get(cache_key)
            if cached is None:
                missing_prompts.append(prompt)
                missing_indices.append(idx)
                missing_keys.append(cache_key)
                continue
            self._entries.move_to_end(cache_key)
            results[idx] = self._to_device(cached.value, device)

        if missing_prompts:
            with text_encoding_prompts(len(missing_prompts), inherit=True):
                encoded_batch = encode_fn(missing_prompts)
            if not isinstance(encoded_batch, list):
                encoded_batch = list(encoded_batch)
            if len(encoded_batch) != len(missing_prompts):
                raise ValueError("encode_fn returned unexpected number of embeddings.")
            for cache_key, idx, encoded in zip(missing_keys, missing_indices, encoded_batch):
                results[idx] = self._store(cache_key, encoded, device)

        return results

    def __contains__(self, cache_key: Hashable) -> bool:
        """Whether an encoding is cached (encode would reuse it)."""
        return cache_key in self._entries

    def _store(self, cache_key: Hashable, encoded: Any, device: torch.device | str | None) -> Any:
        cached_value = self._detach_to_cpu(encoded)
        size_bytes = self._estimate_size_bytes(cached_value)
        if size_bytes <= self.max_size_bytes:
            existing = self._entries.pop(cache_key, None)
            if existing is not None:
                self._size_bytes -= existing.size_bytes
            self._entries[cache_key] = _CacheEntry(cached_value, size_bytes)
            self._size_bytes += size_bytes
            self._purge_if_needed()
        else:
            if cache_key in self._entries:
                self._entries.move_to_end(cache_key)
        return self._to_device(encoded, device)

    def _purge_if_needed(self) -> None:
        if self._size_bytes <= self.max_size_bytes:
            return
        while self._entries and self._size_bytes > self.max_size_bytes:
            _, entry = self._entries.popitem(last=False)
            self._size_bytes -= entry.size_bytes

    def _estimate_size_bytes(self, value: Any) -> int:
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, dict):
            return sum(self._estimate_size_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._estimate_size_bytes(v) for v in value)
        return 0

    def _detach_to_cpu(self, value: Any) -> Any:
        if torch.is_tensor(value):
            if value.device.type == "cpu":
                return value.detach()
            return value.detach().to("cpu")
        if isinstance(value, dict):
            return {k: self._detach_to_cpu(v) for k, v in value.items()}
        if isinstance(value, tuple):
            items = [self._detach_to_cpu(v) for v in value]
            if hasattr(value, "_fields"):
                return value.__class__(*items)
            return tuple(items)
        if isinstance(value, list):
            return [self._detach_to_cpu(v) for v in value]
        return value

    def _to_device(self, value: Any, device: torch.device | str | None) -> Any:
        if device is None:
            return value
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {k: self._to_device(v, device) for k, v in value.items()}
        if isinstance(value, tuple):
            items = [self._to_device(v, device) for v in value]
            if hasattr(value, "_fields"):
                return value.__class__(*items)
            return tuple(items)
        if isinstance(value, list):
            return [self._to_device(v, device) for v in value]
        return value
