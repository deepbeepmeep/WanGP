# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch import nn


@dataclass
class MMArg:
    vid: Any
    txt: Any


def get_args(key: str, args: List[Any]) -> List[Any]:
    return [getattr(v, key) if isinstance(v, MMArg) else v for v in args]


def get_kwargs(key: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: getattr(v, key) if isinstance(v, MMArg) else v for k, v in kwargs.items()}


class MMModule(nn.Module):
    def __init__(
        self,
        module: Callable[..., nn.Module],
        *args,
        shared_weights: bool = False,
        vid_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.shared_weights = shared_weights
        self.vid_only = vid_only
        if self.shared_weights:
            assert get_args("vid", args) == get_args("txt", args)
            assert get_kwargs("vid", kwargs) == get_kwargs("txt", kwargs)
            self.all = module(*get_args("vid", args), **get_kwargs("vid", kwargs))
        else:
            self.vid = module(*get_args("vid", args), **get_kwargs("vid", kwargs))
            self.txt = (
                module(*get_args("txt", args), **get_kwargs("txt", kwargs))
                if not vid_only
                else None
            )

    def forward(
        self,
        vid: torch.FloatTensor,
        txt: torch.FloatTensor,
        *args,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_module = self.vid if not self.shared_weights else self.all
        vid = vid_module(vid, *get_args("vid", args), **get_kwargs("vid", kwargs))
        if not self.vid_only:
            txt_module = self.txt if not self.shared_weights else self.all
            txt = txt_module(txt, *get_args("txt", args), **get_kwargs("txt", kwargs))
        return vid, txt

    def forward_disposable(
        self,
        vid_list: List[torch.FloatTensor],
        txt_list: List[torch.FloatTensor],
        *args,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_module = self.vid if not self.shared_weights else self.all
        txt = txt_list.pop()
        if not self.vid_only:
            txt_module = self.txt if not self.shared_weights else self.all
            txt = txt_module(txt, *get_args("txt", args), **get_kwargs("txt", kwargs))
        vid = vid_module(vid_list.pop(), *get_args("vid", args), **get_kwargs("vid", kwargs))
        return vid, txt

    def module(self, branch: str):
        """The vid or txt module (the shared one when the weights are shared); None for the text of a video-only module."""
        return None if branch == "txt" and self.vid_only else self.all if self.shared_weights else getattr(self, branch)

    def forward_vid_disposable(self, vid_list: List[torch.FloatTensor], *args, **kwargs) -> torch.FloatTensor:
        module = self.vid if not self.shared_weights else self.all
        return module(vid_list.pop(), *get_args("vid", args), **get_kwargs("vid", kwargs))

    def forward_txt_disposable(self, txt_list: List[torch.FloatTensor], *args, **kwargs) -> torch.FloatTensor:
        module = self.txt if not self.shared_weights else self.all
        return module(txt_list.pop(), *get_args("txt", args), **get_kwargs("txt", kwargs))
