"""
Pydantic models for prompt enhancement endpoint.

Supports multiple prompt enhancer backends:
  - qwen35_9b_abliterated / qwen35_4b_abliterated (CUDA required — vLLM kernels)
  - florence2_llama32 / florence2_joycaption (standard PyTorch, may work on MPS)
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class PromptEnhancerModel(str, Enum):
    """Supported models for prompt enhancement.

    Qwen3.5 models use CUDA vLLM kernels and require an NVIDIA GPU.
    Florence2-based models use standard PyTorch + SDPA attention and
    may work on Apple Silicon (MPS), though performance will be slower.
    """

    QWEN35_9B_ABLITERATED = "qwen35_9b_abliterated"
    QWEN35_4B_ABLITERATED = "qwen35_4b_abliterated"
    FLORENCE2_LLAMA32 = "florence2_llama32"
    FLORENCE2_JOYCAPTION = "florence2_joycaption"

    @property
    def enhancer_enabled(self) -> int:
        """Get the enhancer_enabled value for the shared.prompt_enhancer.loader.

        Maps to shared.prompt_enhancer.loader expectations:
          4 -> Qwen3.5-9B Abliterated (CUDA vLLM)
          3 -> Qwen3.5-4B Abliterated (CUDA vLLM)
          1 -> Florence2 + Llama3.2 (standard HF + SDPA)
          2 -> Florence2 + JoyCaption (standard HF + SDPA)
        """
        return {
            PromptEnhancerModel.QWEN35_9B_ABLITERATED: 4,
            PromptEnhancerModel.QWEN35_4B_ABLITERATED: 3,
            PromptEnhancerModel.FLORENCE2_LLAMA32: 1,
            PromptEnhancerModel.FLORENCE2_JOYCAPTION: 2,
        }[self]

    @property
    def display_name(self) -> str:
        return {
            PromptEnhancerModel.QWEN35_9B_ABLITERATED: "Qwen3.5-9B Abliterated",
            PromptEnhancerModel.QWEN35_4B_ABLITERATED: "Qwen3.5-4B Abliterated",
            PromptEnhancerModel.FLORENCE2_LLAMA32: "Florence2 + Llama3.2",
            PromptEnhancerModel.FLORENCE2_JOYCAPTION: "Florence2 + JoyCaption",
        }[self]

    @property
    def requires_cuda(self) -> bool:
        """Whether this model requires CUDA (vs potentially working on MPS/CPU)."""
        return self in (PromptEnhancerModel.QWEN35_9B_ABLITERATED, PromptEnhancerModel.QWEN35_4B_ABLITERATED)


class PromptEnhanceRequest(BaseModel):
    """Request model for prompt enhancement."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The text prompt to enhance",
    )
    model: PromptEnhancerModel = Field(
        default=PromptEnhancerModel.FLORENCE2_LLAMA32,
        description="The prompt enhancer model to use",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Maximum number of tokens to generate",
    )
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    )
    seed: int = Field(
        default=-1,
        description="Random seed (-1 for random)",
    )


class PromptEnhanceResponse(BaseModel):
    """Response model for prompt enhancement."""

    enhanced_prompt: str = Field(
        ...,
        description="The enhanced prompt",
    )
    model_used: str = Field(
        ...,
        description="The model that was used for enhancement",
    )
    seed_used: Optional[int] = Field(
        default=None,
        description="The seed used for generation",
    )


class DeepyMessageRequest(BaseModel):
    """Request model for Deepy (LLM) messaging endpoint."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The text prompt to send to the LLM",
    )
    image: Optional[str] = Field(
        default=None,
        description="Base64-encoded image (data URI: data:image/png;base64,...)",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Maximum number of tokens to generate",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    )
    seed: int = Field(
        default=-1,
        description="Random seed (-1 for random)",
    )


class DeepyMessageResponse(BaseModel):
    """Response model for Deepy message."""

    response: str = Field(
        ...,
        description="The LLM's response text",
    )
    model_used: str = Field(
        ...,
        description="The model that was used (Qwen3.5 VL)",
    )
    seed_used: Optional[int] = Field(
        default=None,
        description="The seed used for generation",
    )
