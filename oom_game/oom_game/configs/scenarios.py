"""Pre-built game scenarios with varying difficulty."""

from dataclasses import dataclass
from typing import Optional, List

from .gpus import GPUConfig, GPU_PRESETS
from .models import ModelConfig, MODEL_PRESETS


@dataclass
class Scenario:
    """A game challenge presented to the player."""

    id: str
    description: str
    difficulty: str  # "tutorial", "easy", "medium", "hard", "expert"
    gpu: GPUConfig
    num_gpus: int
    model: ModelConfig
    target_utilization: float = 0.95

    # Optional constraints for specific challenges
    min_batch_size: Optional[int] = None
    max_batch_size: Optional[int] = None
    min_seq_len: Optional[int] = None
    max_seq_len: Optional[int] = None
    allowed_parallelism: Optional[List[str]] = None
    require_gradient_checkpointing: Optional[bool] = None


# Curated scenarios that teach different concepts
SCENARIOS: List[Scenario] = [
    # ===== TUTORIAL =====
    # Start small, learn the basics
    Scenario(
        id="tutorial_1",
        description="Baby Steps: GPT-2 Small on RTX 4090",
        difficulty="tutorial",
        gpu=GPU_PRESETS["RTX_4090"],
        num_gpus=1,
        model=MODEL_PRESETS["gpt2_small"],
        target_utilization=0.90,  # More forgiving
    ),
    Scenario(
        id="tutorial_2",
        description="Stepping Up: GPT-2 Large on RTX 4090",
        difficulty="tutorial",
        gpu=GPU_PRESETS["RTX_4090"],
        num_gpus=1,
        model=MODEL_PRESETS["gpt2_large"],
        target_utilization=0.90,
    ),

    # ===== EASY =====
    # Single GPU, manageable models
    Scenario(
        id="easy_1",
        description="Consumer Setup: LLaMA 1B on RTX 3090",
        difficulty="easy",
        gpu=GPU_PRESETS["RTX_3090"],
        num_gpus=1,
        model=MODEL_PRESETS["llama_1b"],
    ),
    Scenario(
        id="easy_2",
        description="Cloud Basic: LLaMA 1B on A10G",
        difficulty="easy",
        gpu=GPU_PRESETS["A10G"],
        num_gpus=1,
        model=MODEL_PRESETS["llama_1b"],
    ),
    Scenario(
        id="easy_3",
        description="Mid-range: Qwen 1.5B on RTX 4090",
        difficulty="easy",
        gpu=GPU_PRESETS["RTX_4090"],
        num_gpus=1,
        model=MODEL_PRESETS["qwen_1.5b"],
    ),

    # ===== MEDIUM =====
    # Introduction to multi-GPU and larger models
    Scenario(
        id="medium_1",
        description="First Multi-GPU: LLaMA 1B on 2x RTX 4090",
        difficulty="medium",
        gpu=GPU_PRESETS["RTX_4090"],
        num_gpus=2,
        model=MODEL_PRESETS["llama_1b"],
    ),
    Scenario(
        id="medium_2",
        description="Data Center Entry: LLaMA 3B on A100 80GB",
        difficulty="medium",
        gpu=GPU_PRESETS["A100_80GB"],
        num_gpus=1,
        model=MODEL_PRESETS["llama_3b"],
    ),
    Scenario(
        id="medium_3",
        description="FSDP Introduction: LLaMA 3B on 4x RTX 4090",
        difficulty="medium",
        gpu=GPU_PRESETS["RTX_4090"],
        num_gpus=4,
        model=MODEL_PRESETS["llama_3b"],
    ),

    # ===== HARD =====
    # Larger models, tighter constraints
    Scenario(
        id="hard_1",
        description="Scaling Up: LLaMA 7B on 8x A100 40GB",
        difficulty="hard",
        gpu=GPU_PRESETS["A100_40GB"],
        num_gpus=8,
        model=MODEL_PRESETS["llama_7b"],
    ),
    Scenario(
        id="hard_2",
        description="Data Center Pro: LLaMA 7B on 4x A100 80GB",
        difficulty="hard",
        gpu=GPU_PRESETS["A100_80GB"],
        num_gpus=4,
        model=MODEL_PRESETS["llama_7b"],
    ),
    Scenario(
        id="hard_3",
        description="Budget Multi-GPU: LLaMA 3B on 8x RTX 3090",
        difficulty="hard",
        gpu=GPU_PRESETS["RTX_3090"],
        num_gpus=8,
        model=MODEL_PRESETS["llama_3b"],
    ),

    # ===== EXPERT =====
    # The big leagues - pushing limits
    Scenario(
        id="expert_1",
        description="The Beast: LLaMA 13B on 8x H100",
        difficulty="expert",
        gpu=GPU_PRESETS["H100_80GB"],
        num_gpus=8,
        model=MODEL_PRESETS["llama_13b"],
    ),
    Scenario(
        id="expert_2",
        description="Memory Master: LLaMA 13B on 8x A100 80GB",
        difficulty="expert",
        gpu=GPU_PRESETS["A100_80GB"],
        num_gpus=8,
        model=MODEL_PRESETS["llama_13b"],
    ),
    Scenario(
        id="expert_3",
        description="Impossible Dream: LLaMA 70B on 8x H100 (FSDP required)",
        difficulty="expert",
        gpu=GPU_PRESETS["H100_80GB"],
        num_gpus=8,
        model=MODEL_PRESETS["llama_70b"],
    ),
]


def get_scenario(scenario_id: str) -> Scenario:
    """Get a scenario by ID."""
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    available = [s.id for s in SCENARIOS]
    raise ValueError(f"Unknown scenario: {scenario_id}. Available: {available}")


def get_scenarios_by_difficulty(difficulty: str) -> List[Scenario]:
    """Get all scenarios of a given difficulty."""
    return [s for s in SCENARIOS if s.difficulty == difficulty]


def list_scenarios() -> None:
    """Print all available scenarios."""
    current_difficulty = None
    for s in SCENARIOS:
        if s.difficulty != current_difficulty:
            current_difficulty = s.difficulty
            print(f"\n{current_difficulty.upper()}:")
        print(f"  {s.id}: {s.description}")
