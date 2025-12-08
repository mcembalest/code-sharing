"""Model configuration presets with architectural parameters."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a model architecture."""

    name: str
    params_billions: float
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int  # FFN hidden dim (usually ~4x hidden_size)
    vocab_size: int

    @property
    def params(self) -> int:
        """Total parameters (approximate)."""
        return int(self.params_billions * 1e9)

    def __str__(self) -> str:
        return f"{self.name} ({self.params_billions}B)"


# Model presets - real architectures with correct dimensions
MODEL_PRESETS: Dict[str, ModelConfig] = {
    # Tutorial / Small models
    "gpt2_small": ModelConfig(
        name="GPT-2 Small",
        params_billions=0.124,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        vocab_size=50257,
    ),
    "gpt2_medium": ModelConfig(
        name="GPT-2 Medium",
        params_billions=0.355,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096,
        vocab_size=50257,
    ),
    "gpt2_large": ModelConfig(
        name="GPT-2 Large",
        params_billions=0.774,
        hidden_size=1280,
        num_layers=36,
        num_heads=20,
        intermediate_size=5120,
        vocab_size=50257,
    ),

    # LLaMA family
    "llama_1b": ModelConfig(
        name="LLaMA 1B",
        params_billions=1.0,
        hidden_size=2048,
        num_layers=16,
        num_heads=16,
        intermediate_size=5504,
        vocab_size=32000,
    ),
    "llama_3b": ModelConfig(
        name="LLaMA 3B",
        params_billions=3.0,
        hidden_size=3200,
        num_layers=26,
        num_heads=32,
        intermediate_size=8640,
        vocab_size=32000,
    ),
    "llama_7b": ModelConfig(
        name="LLaMA 7B",
        params_billions=7.0,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        intermediate_size=11008,
        vocab_size=32000,
    ),
    "llama_13b": ModelConfig(
        name="LLaMA 13B",
        params_billions=13.0,
        hidden_size=5120,
        num_layers=40,
        num_heads=40,
        intermediate_size=13824,
        vocab_size=32000,
    ),
    "llama_70b": ModelConfig(
        name="LLaMA 70B",
        params_billions=70.0,
        hidden_size=8192,
        num_layers=80,
        num_heads=64,
        intermediate_size=28672,
        vocab_size=32000,
    ),

    # Mistral
    "mistral_7b": ModelConfig(
        name="Mistral 7B",
        params_billions=7.3,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        intermediate_size=14336,
        vocab_size=32000,
    ),

    # Qwen
    "qwen_1.5b": ModelConfig(
        name="Qwen 1.5B",
        params_billions=1.5,
        hidden_size=1536,
        num_layers=28,
        num_heads=12,
        intermediate_size=8960,
        vocab_size=151936,
    ),
    "qwen_7b": ModelConfig(
        name="Qwen 7B",
        params_billions=7.0,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        intermediate_size=11008,
        vocab_size=151936,
    ),
}


def get_model(name: str) -> ModelConfig:
    """Get a model config by name, with helpful error message."""
    if name not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODEL_PRESETS[name]
