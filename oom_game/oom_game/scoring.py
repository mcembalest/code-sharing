"""Scoring algorithm for the OOM game."""

from dataclasses import dataclass
from .memory.calculator import MemoryResult


@dataclass
class Score:
    """Game score with detailed breakdown."""

    points: int
    grade: str  # S, A, B, C, D, F, or OOM
    utilization_percent: float
    feedback: str
    hint: str

    # Score breakdown
    base_score: int
    efficiency_bonus: int
    penalty: int


def calculate_score(result: MemoryResult, target_utilization: float = 0.95) -> Score:
    """
    Calculate score based on how close to target utilization.

    Scoring algorithm:
    - Base: 100 points for hitting target utilization exactly
    - Penalty: -2 points per percentage point away from target
    - Bonus: +10 points for being in the "optimal zone" (90-98%)
    - OOM: Automatic 0 points

    Grades:
    - S: 95+ points (expert level - you really know your stuff)
    - A: 85-94 points (great job)
    - B: 70-84 points (good)
    - C: 50-69 points (okay, room for improvement)
    - D: 25-49 points (needs work)
    - F: 1-24 points (poor)
    - OOM: 0 points (crashed!)

    Args:
        result: Memory calculation result
        target_utilization: Target utilization (default 95%)

    Returns:
        Score with points, grade, and feedback
    """
    utilization_pct = result.utilization_percent
    target_pct = target_utilization * 100

    # OOM is an instant fail
    if result.is_oom:
        deficit_gb = -result.headroom_gb
        return Score(
            points=0,
            grade="OOM",
            utilization_percent=utilization_pct,
            feedback=f"CUDA OUT OF MEMORY! You needed {deficit_gb:.1f} GB more VRAM.",
            hint=_get_oom_hint(result),
            base_score=0,
            efficiency_bonus=0,
            penalty=-100,
        )

    # Calculate distance from target
    distance = abs(utilization_pct - target_pct)

    # Base score: start at 100, lose 2 points per % away from target
    base_score = max(0, int(100 - distance * 2))

    # Efficiency bonus: reward being in the sweet spot (90-98%)
    efficiency_bonus = 0
    if 90 <= utilization_pct <= 98:
        efficiency_bonus = 10
    elif 85 <= utilization_pct < 90:
        efficiency_bonus = 5

    # Penalty for very low utilization (wasting expensive GPU resources)
    waste_penalty = 0
    if utilization_pct < 50:
        waste_penalty = int((50 - utilization_pct) * 0.5)

    total_points = max(0, base_score + efficiency_bonus - waste_penalty)

    # Determine grade
    grade = _get_grade(total_points)

    # Generate feedback and hint
    feedback = _get_feedback(utilization_pct, result.headroom_gb)
    hint = _get_optimization_hint(utilization_pct, result)

    return Score(
        points=total_points,
        grade=grade,
        utilization_percent=utilization_pct,
        feedback=feedback,
        hint=hint,
        base_score=base_score,
        efficiency_bonus=efficiency_bonus,
        penalty=-waste_penalty,
    )


def _get_grade(points: int) -> str:
    """Convert points to letter grade."""
    if points >= 95:
        return "S"
    elif points >= 85:
        return "A"
    elif points >= 70:
        return "B"
    elif points >= 50:
        return "C"
    elif points >= 25:
        return "D"
    else:
        return "F"


def _get_feedback(utilization_pct: float, headroom_gb: float) -> str:
    """Generate feedback message based on utilization."""
    if utilization_pct > 98:
        return "Living dangerously! You're right on the edge of OOM."
    elif utilization_pct >= 93:
        return "Excellent! Near-optimal GPU utilization."
    elif utilization_pct >= 85:
        return f"Great job! {headroom_gb:.1f} GB headroom remaining."
    elif utilization_pct >= 70:
        return f"Good, but {headroom_gb:.1f} GB unused. Could fit more."
    elif utilization_pct >= 50:
        return f"Decent, but {headroom_gb:.1f} GB wasted. Try larger batch/seq."
    else:
        return f"Very conservative! {headroom_gb:.1f} GB sitting idle."


def _get_optimization_hint(utilization_pct: float, result: MemoryResult) -> str:
    """Provide actionable optimization hint."""
    if utilization_pct > 98:
        return "Consider gradient checkpointing for safety margin."
    elif utilization_pct >= 90:
        return "Perfect zone! Minor tweaks won't make much difference."
    elif utilization_pct >= 70:
        return "Try increasing batch_size or seq_len to use more VRAM."
    elif utilization_pct >= 50:
        gb = result.breakdown.to_gb()
        if gb["activations"] < gb["optimizer_states"] * 0.5:
            return "Activations are small - double or triple your batch size!"
        return "Lots of room! Try 2x batch_size."
    else:
        return "GPU is barely working. Increase batch_size significantly!"


def _get_oom_hint(result: MemoryResult) -> str:
    """Provide hint for OOM situations."""
    gb = result.breakdown.to_gb()

    # Check what's dominating memory
    total = gb["total"]
    opt_pct = gb["optimizer_states"] / total * 100
    act_pct = gb["activations"] / total * 100

    hints = []

    if opt_pct > 60:
        hints.append("Optimizer states dominate - try FSDP to shard them")

    if act_pct > 30:
        hints.append("Activations are large - try gradient checkpointing or smaller batch/seq")

    if not hints:
        hints.append("Try FSDP with more GPUs, or a smaller model")

    return hints[0]
