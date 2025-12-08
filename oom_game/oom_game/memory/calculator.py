"""Memory calculator with parallelism support."""

from dataclasses import dataclass
from typing import Literal

from ..configs.gpus import GPUConfig
from ..configs.models import ModelConfig
from ..game import TrainingConfig, ParallelismStrategy
from .formulas import MemoryFormulas, EasyModeFormulas, HardModeFormulas, MemoryBreakdown


@dataclass
class MemoryResult:
    """Result of memory calculation for a training run."""

    per_gpu_memory: int       # Bytes required per GPU
    total_cluster_memory: int  # Total across all GPUs
    vram_available: int       # VRAM per GPU in bytes
    breakdown: MemoryBreakdown
    mode: str                 # "Easy" or "Hard"

    @property
    def utilization(self) -> float:
        """Memory utilization as a fraction (0.0 to 1.0+)."""
        return self.per_gpu_memory / self.vram_available

    @property
    def utilization_percent(self) -> float:
        """Memory utilization as a percentage."""
        return self.utilization * 100

    @property
    def is_oom(self) -> bool:
        """True if memory exceeds available VRAM."""
        return self.per_gpu_memory > self.vram_available

    @property
    def headroom_bytes(self) -> int:
        """Spare memory (positive) or deficit (negative)."""
        return self.vram_available - self.per_gpu_memory

    @property
    def headroom_gb(self) -> float:
        """Headroom in GB."""
        return self.headroom_bytes / (1024 ** 3)


class MemoryCalculator:
    """
    Main calculator that computes memory requirements.
    Handles parallelism strategies and mode switching.
    """

    def __init__(self, mode: Literal["easy", "hard"] = "easy"):
        self.mode = mode
        self.formulas: MemoryFormulas = (
            EasyModeFormulas() if mode == "easy" else HardModeFormulas()
        )

    def calculate(
        self,
        gpu: GPUConfig,
        model: ModelConfig,
        training: TrainingConfig,
    ) -> MemoryResult:
        """
        Calculate memory requirements for the given configuration.

        Args:
            gpu: GPU configuration (VRAM, specs)
            model: Model architecture (params, hidden_size, etc.)
            training: Training hyperparameters (batch_size, parallelism, etc.)

        Returns:
            MemoryResult with per-GPU memory, utilization, and breakdown
        """
        bytes_per_param = training.precision.bytes_per_param
        params = model.params

        # Base memory calculations (before parallelism)
        model_weights = self.formulas.calc_model_weights(params, bytes_per_param)
        optimizer_states = self.formulas.calc_optimizer_states(params)
        gradients = self.formulas.calc_gradients(params, bytes_per_param)
        activations = self.formulas.calc_activations(
            batch_size=training.batch_size,
            seq_len=training.seq_len,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
            num_heads=model.num_heads,
            vocab_size=model.vocab_size,
            bytes_per_elem=bytes_per_param,
            gradient_checkpointing=training.gradient_checkpointing,
        )

        # Apply parallelism effects
        per_gpu = self._apply_parallelism(
            model_weights=model_weights,
            optimizer_states=optimizer_states,
            gradients=gradients,
            activations=activations,
            strategy=training.parallelism,
            num_gpus=training.num_gpus,
        )

        breakdown = MemoryBreakdown(
            model_weights=per_gpu["model_weights"],
            optimizer_states=per_gpu["optimizer_states"],
            gradients=per_gpu["gradients"],
            activations=per_gpu["activations"],
            formula_explanations=self.formulas.get_explanations(),
        )

        return MemoryResult(
            per_gpu_memory=breakdown.total,
            total_cluster_memory=breakdown.total * training.num_gpus,
            vram_available=gpu.vram_bytes,
            breakdown=breakdown,
            mode=self.formulas.mode_name,
        )

    def _apply_parallelism(
        self,
        model_weights: int,
        optimizer_states: int,
        gradients: int,
        activations: int,
        strategy: ParallelismStrategy,
        num_gpus: int,
    ) -> dict:
        """
        Apply parallelism strategy to memory requirements.

        DDP (Distributed Data Parallel):
        - Full model replicated on each GPU
        - Gradients synchronized via all-reduce
        - Memory per GPU = full model + activations for local batch

        FSDP (Fully Sharded Data Parallel):
        - Model weights, optimizer states, and gradients sharded across GPUs
        - Each GPU holds 1/N of model state
        - Activations NOT sharded (each GPU processes its own batch)
        - Small overhead (1.1x) for gather/scatter communication buffers
        """
        if strategy == ParallelismStrategy.NONE or num_gpus == 1:
            # Single GPU: full memory on one device
            return {
                "model_weights": model_weights,
                "optimizer_states": optimizer_states,
                "gradients": gradients,
                "activations": activations,
            }

        elif strategy == ParallelismStrategy.DDP:
            # DDP: Full model replicated, only data parallelism
            # Each GPU has full copy of model, optimizer, gradients
            # Activations are per-GPU (different data shards)
            return {
                "model_weights": model_weights,
                "optimizer_states": optimizer_states,
                "gradients": gradients,
                "activations": activations,
            }

        elif strategy == ParallelismStrategy.FSDP:
            # FSDP: Model state sharded across GPUs
            # ZeRO-3 style: weights, optimizer, gradients all sharded
            # Communication overhead: ~10% for gather/scatter buffers
            shard_factor = num_gpus
            overhead = 1.1  # 10% overhead for communication buffers

            return {
                "model_weights": int(model_weights / shard_factor * overhead),
                "optimizer_states": int(optimizer_states / shard_factor * overhead),
                "gradients": int(gradients / shard_factor * overhead),
                "activations": activations,  # NOT sharded - each GPU has its batch
            }

        raise ValueError(f"Unknown parallelism strategy: {strategy}")
