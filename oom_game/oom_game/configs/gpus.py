"""GPU configuration presets."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class GPUConfig:
    """Configuration for a GPU type."""

    name: str
    vram_gb: float
    memory_bandwidth_gbps: float  # GB/s - for future throughput calculations
    tflops_fp16: float  # FP16 tensor core TFLOPS

    @property
    def vram_bytes(self) -> int:
        """VRAM in bytes."""
        return int(self.vram_gb * (1024 ** 3))

    def __str__(self) -> str:
        return f"{self.name} ({self.vram_gb}GB)"


# GPU presets - real-world specs
GPU_PRESETS: Dict[str, GPUConfig] = {
    # Data center GPUs
    "A100_40GB": GPUConfig(
        name="NVIDIA A100 40GB",
        vram_gb=40,
        memory_bandwidth_gbps=1555,
        tflops_fp16=312,
    ),
    "A100_80GB": GPUConfig(
        name="NVIDIA A100 80GB",
        vram_gb=80,
        memory_bandwidth_gbps=2039,
        tflops_fp16=312,
    ),
    "H100_80GB": GPUConfig(
        name="NVIDIA H100 80GB",
        vram_gb=80,
        memory_bandwidth_gbps=3350,
        tflops_fp16=989,
    ),
    "H100_SXM": GPUConfig(
        name="NVIDIA H100 SXM",
        vram_gb=80,
        memory_bandwidth_gbps=3350,
        tflops_fp16=989,
    ),
    "A10G": GPUConfig(
        name="NVIDIA A10G",
        vram_gb=24,
        memory_bandwidth_gbps=600,
        tflops_fp16=31.2,
    ),
    "V100_32GB": GPUConfig(
        name="NVIDIA V100 32GB",
        vram_gb=32,
        memory_bandwidth_gbps=900,
        tflops_fp16=125,
    ),
    "V100_16GB": GPUConfig(
        name="NVIDIA V100 16GB",
        vram_gb=16,
        memory_bandwidth_gbps=900,
        tflops_fp16=125,
    ),

    # Consumer GPUs
    "RTX_4090": GPUConfig(
        name="NVIDIA RTX 4090",
        vram_gb=24,
        memory_bandwidth_gbps=1008,
        tflops_fp16=82.6,
    ),
    "RTX_4080": GPUConfig(
        name="NVIDIA RTX 4080",
        vram_gb=16,
        memory_bandwidth_gbps=716,
        tflops_fp16=48.7,
    ),
    "RTX_3090": GPUConfig(
        name="NVIDIA RTX 3090",
        vram_gb=24,
        memory_bandwidth_gbps=936,
        tflops_fp16=35.6,
    ),
    "RTX_3080": GPUConfig(
        name="NVIDIA RTX 3080",
        vram_gb=10,
        memory_bandwidth_gbps=760,
        tflops_fp16=29.8,
    ),
}


def get_gpu(name: str) -> GPUConfig:
    """Get a GPU config by name, with helpful error message."""
    if name not in GPU_PRESETS:
        available = ", ".join(GPU_PRESETS.keys())
        raise ValueError(f"Unknown GPU: {name}. Available: {available}")
    return GPU_PRESETS[name]
