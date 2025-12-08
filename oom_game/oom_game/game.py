"""Core game data structures and enums."""

from dataclasses import dataclass
from enum import Enum
from typing import List


class Precision(Enum):
    """Training precision options."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    @property
    def bytes_per_param(self) -> int:
        """Bytes per parameter for this precision."""
        if self == Precision.FP32:
            return 4
        return 2  # FP16 and BF16

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return self.value.upper()


class ParallelismStrategy(Enum):
    """Distributed training strategies."""

    NONE = "none"   # Single GPU
    DDP = "ddp"     # Distributed Data Parallel
    FSDP = "fsdp"   # Fully Sharded Data Parallel

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        if self == ParallelismStrategy.NONE:
            return "Single GPU"
        return self.value.upper()

    @property
    def description(self) -> str:
        """Short description of the strategy."""
        descriptions = {
            ParallelismStrategy.NONE: "Full model on one GPU",
            ParallelismStrategy.DDP: "Full model replicated on each GPU, gradients synchronized",
            ParallelismStrategy.FSDP: "Model/optimizer/gradients sharded across GPUs",
        }
        return descriptions[self]


@dataclass
class TrainingConfig:
    """Player-chosen training hyperparameters."""

    num_gpus: int
    parallelism: ParallelismStrategy
    batch_size: int       # Per-GPU batch size
    seq_len: int          # Sequence length
    precision: Precision
    gradient_checkpointing: bool = False

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors = []

        if self.num_gpus < 1:
            errors.append("Must have at least 1 GPU")

        if self.num_gpus == 1 and self.parallelism != ParallelismStrategy.NONE:
            errors.append(f"Single GPU cannot use {self.parallelism.display_name}")

        if self.num_gpus > 1 and self.parallelism == ParallelismStrategy.NONE:
            errors.append("Multi-GPU requires DDP or FSDP parallelism")

        if self.batch_size < 1:
            errors.append("Batch size must be >= 1")

        if self.seq_len < 1:
            errors.append("Sequence length must be >= 1")

        return errors

    @property
    def global_batch_size(self) -> int:
        """Total batch size across all GPUs."""
        return self.batch_size * self.num_gpus
