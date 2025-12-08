"""Memory calculation formulas for easy and hard modes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
import math


@dataclass
class MemoryBreakdown:
    """Detailed memory breakdown in bytes with explanations."""

    model_weights: int
    optimizer_states: int
    gradients: int
    activations: int
    formula_explanations: Dict[str, str]

    @property
    def total(self) -> int:
        """Total memory in bytes."""
        return (
            self.model_weights
            + self.optimizer_states
            + self.gradients
            + self.activations
        )

    def to_gb(self) -> Dict[str, float]:
        """Convert all values to GB for display."""
        gb = 1024 ** 3
        return {
            "model_weights": self.model_weights / gb,
            "optimizer_states": self.optimizer_states / gb,
            "gradients": self.gradients / gb,
            "activations": self.activations / gb,
            "total": self.total / gb,
        }


class MemoryFormulas(ABC):
    """Abstract base for memory calculation formulas."""

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Name of this formula mode."""
        pass

    @abstractmethod
    def calc_model_weights(self, params: int, bytes_per_param: int) -> int:
        """Calculate model weights memory."""
        pass

    @abstractmethod
    def calc_optimizer_states(self, params: int) -> int:
        """Calculate optimizer states memory (assumes Adam)."""
        pass

    @abstractmethod
    def calc_gradients(self, params: int, bytes_per_param: int) -> int:
        """Calculate gradients memory."""
        pass

    @abstractmethod
    def calc_activations(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int,
        bytes_per_elem: int,
        gradient_checkpointing: bool,
    ) -> int:
        """Calculate activation memory."""
        pass

    @abstractmethod
    def get_explanations(self) -> Dict[str, str]:
        """Get formula explanations for educational display."""
        pass


class EasyModeFormulas(MemoryFormulas):
    """
    Simplified formulas for learning the basics.
    Uses clean rule-of-thumb multipliers that are easy to mental-math.
    """

    @property
    def mode_name(self) -> str:
        return "Easy"

    def calc_model_weights(self, params: int, bytes_per_param: int) -> int:
        """Simple: params x bytes_per_param."""
        return params * bytes_per_param

    def calc_optimizer_states(self, params: int) -> int:
        """
        Adam optimizer stores:
        - Momentum (m): FP32 = 4 bytes
        - Variance (v): FP32 = 4 bytes
        - Master weights: FP32 = 4 bytes
        Total: 12 bytes per parameter

        Easy to remember: "3x the model size in FP32"
        """
        return params * 12

    def calc_gradients(self, params: int, bytes_per_param: int) -> int:
        """Same size as model weights."""
        return params * bytes_per_param

    def calc_activations(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int,
        bytes_per_elem: int,
        gradient_checkpointing: bool,
    ) -> int:
        """
        Simplified rule of thumb:
        batch_size x seq_len x hidden_size x num_layers x 34 x bytes

        The "34" comes from storing intermediate activations:
        - Attention: Q, K, V projections, attention weights, output
        - FFN: two linear layers with intermediate activations
        - LayerNorm: 2 per layer
        - Residual connections
        """
        base = batch_size * seq_len * hidden_size * num_layers * 34 * bytes_per_elem

        if gradient_checkpointing:
            # Checkpointing stores only ~sqrt(layers) worth of activations
            # We recompute the rest during backward pass
            checkpoint_factor = math.sqrt(num_layers)
            base = int(base / checkpoint_factor)

        return base

    def get_explanations(self) -> Dict[str, str]:
        return {
            "model_weights": "params x bytes_per_param",
            "optimizer_states": "params x 12 bytes (Adam: m + v + master weights)",
            "gradients": "params x bytes_per_param",
            "activations": "batch x seq x hidden x layers x 34 x bytes",
        }


class HardModeFormulas(MemoryFormulas):
    """
    Realistic formulas based on transformer architecture.

    References:
    - "Reducing Activation Recomputation in Large Transformer Models" (NVIDIA)
    - EleutherAI Transformer Math 101
    - nanoGPT memory analysis
    """

    @property
    def mode_name(self) -> str:
        return "Hard"

    def calc_model_weights(self, params: int, bytes_per_param: int) -> int:
        """
        Model weights + buffers (layernorm running stats, etc.)
        Buffers typically add ~2% overhead.
        """
        buffer_overhead = 1.02
        return int(params * bytes_per_param * buffer_overhead)

    def calc_optimizer_states(self, params: int) -> int:
        """
        Adam with mixed precision training:
        - FP32 master weights: 4 bytes (for precision during updates)
        - FP32 momentum (m): 4 bytes (first moment estimate)
        - FP32 variance (v): 4 bytes (second moment estimate)
        Total: 12 bytes per parameter
        """
        return params * 12

    def calc_gradients(self, params: int, bytes_per_param: int) -> int:
        """Gradients stored in same precision as forward pass."""
        return params * bytes_per_param

    def calc_activations(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int,
        bytes_per_elem: int,
        gradient_checkpointing: bool,
    ) -> int:
        """
        Detailed activation memory based on transformer architecture.

        Per-layer breakdown:
        - Input to attention: batch x seq x hidden (saved for backward)
        - Q, K, V projections: 3 x batch x seq x hidden
        - Attention scores: batch x heads x seq x seq (quadratic in seq_len!)
        - Attention output: batch x seq x hidden
        - Post-attention dropout mask: batch x seq x hidden / 8 (bits)
        - FFN input: batch x seq x hidden
        - FFN intermediate: batch x seq x intermediate_size (4x hidden)
        - FFN activation (GeLU input saved): batch x seq x intermediate
        - FFN output dropout mask: batch x seq x hidden / 8

        Simplified formula from NVIDIA paper:
        Per layer: sbh(34 + 5as/h) bytes for fp16
        Where: s=seq_len, b=batch, h=hidden, a=num_heads

        We use a slightly simplified version:
        36 * N_e + 6 * N_a per layer
        Where: N_e = batch x seq x hidden, N_a = batch x heads x seq^2
        """
        # Embedding tensor elements per layer
        N_e = batch_size * seq_len * hidden_size

        # Attention matrix elements (the quadratic term!)
        N_a = batch_size * num_heads * seq_len * seq_len

        # Per-layer activation memory
        per_layer = (36 * N_e + 6 * N_a) * bytes_per_elem

        # Output layer (logits over vocabulary)
        N_logits = batch_size * seq_len * vocab_size
        output_activations = 4 * N_logits * bytes_per_elem  # logits, softmax, etc.

        # Embedding layer activations
        embedding_activations = 4 * N_e * bytes_per_elem

        total = (num_layers * per_layer) + output_activations + embedding_activations

        if gradient_checkpointing:
            # With checkpointing, we only store activations at checkpoint boundaries
            # Typically sqrt(layers) checkpoints, so we store ~sqrt(layers) layers
            # Plus we need to store the embeddings and output always
            checkpoint_layers = int(math.sqrt(num_layers)) + 1
            total = (
                (checkpoint_layers * per_layer)
                + output_activations
                + embedding_activations
            )

        return total

    def get_explanations(self) -> Dict[str, str]:
        return {
            "model_weights": "params x bytes x 1.02 (buffer overhead)",
            "optimizer_states": "params x 12 (Adam FP32: master + m + v)",
            "gradients": "params x bytes_per_param",
            "activations": "layers x (36 x batch x seq x hidden + 6 x batch x heads x seq^2) x bytes",
        }
