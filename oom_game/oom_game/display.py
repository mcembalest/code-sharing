"""Terminal display utilities for the OOM game."""

from typing import Optional, List, TYPE_CHECKING
import sys

from .memory.calculator import MemoryResult
from .scoring import Score

if TYPE_CHECKING:
    from .timer import RoundTimer


# Box drawing characters
BOX_H = "─"
BOX_V = "│"
BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"


def print_header(text: str, width: int = 60) -> None:
    """Print a styled header box."""
    padding = (width - len(text) - 2) // 2
    print()
    print(BOX_TL + BOX_H * (width - 2) + BOX_TR)
    print(BOX_V + " " * padding + text + " " * (width - len(text) - padding - 2) + BOX_V)
    print(BOX_BL + BOX_H * (width - 2) + BOX_BR)


def print_subheader(text: str) -> None:
    """Print a section subheader."""
    print(f"\n{text}")
    print(BOX_H * len(text))


def print_scenario(scenario) -> None:
    """Print scenario description."""
    print(f"\nSCENARIO: {scenario.description}")
    print(f"  GPU:   {scenario.num_gpus}x {scenario.gpu.name} ({scenario.gpu.vram_gb}GB each)")
    print(f"  Model: {scenario.model.name} ({scenario.model.params_billions}B params)")
    print(f"  Goal:  Get as close to {int(scenario.target_utilization * 100)}% utilization without OOM!")


def print_memory_breakdown(result: MemoryResult, score: Score) -> None:
    """Print detailed memory breakdown and score."""
    gb = result.breakdown.to_gb()
    vram_gb = result.vram_available / (1024 ** 3)

    print_subheader("Memory Breakdown")

    # Memory components
    print(f"  Model Weights:    {gb['model_weights']:>8.2f} GB")
    print(f"  Optimizer States: {gb['optimizer_states']:>8.2f} GB")
    print(f"  Gradients:        {gb['gradients']:>8.2f} GB")
    print(f"  Activations:      {gb['activations']:>8.2f} GB")
    print(f"  {BOX_H * 34}")
    print(f"  TOTAL:            {gb['total']:>8.2f} GB")
    print(f"  VRAM Available:   {vram_gb:>8.2f} GB")

    # Utilization bar
    print()
    print_utilization_bar(result.utilization_percent, result.is_oom)

    # Score display
    print()
    if score.grade == "OOM":
        print("  >>> OUT OF MEMORY <<<")
        print(f"  {score.feedback}")
    else:
        print(f"  Utilization: {score.utilization_percent:.1f}%")
        print(f"  {score.feedback}")

    print()
    print(f"  Grade: {score.grade}  |  Score: {score.points}/100")
    print(f"  Hint: {score.hint}")


def print_utilization_bar(percent: float, is_oom: bool, width: int = 40) -> None:
    """Print a visual utilization bar."""
    # Clamp for display
    display_pct = min(percent, 150)  # Cap at 150% for display
    filled = int(width * min(display_pct / 100, 1.0))
    overflow = int(width * max(0, (display_pct - 100) / 50))  # Overflow section

    bar = "█" * filled + "░" * (width - filled)

    if is_oom:
        status = "OOM!"
        marker = " " * width + " <<< EXCEEDED"
    else:
        status = f"{percent:.1f}%"
        marker_pos = int(width * 0.95)  # 95% target marker
        marker = " " * marker_pos + "▲ 95%"

    print(f"  [{bar}] {status}")
    print(f"   {marker}")


def print_config_summary(config, scenario) -> None:
    """Print the player's configuration choices."""
    print_subheader("Your Configuration")
    print(f"  Parallelism:   {config.parallelism.display_name}")
    print(f"  Precision:     {config.precision.display_name}")
    print(f"  Batch Size:    {config.batch_size} (global: {config.global_batch_size})")
    print(f"  Seq Length:    {config.seq_len}")
    print(f"  Grad Ckpt:     {'Yes' if config.gradient_checkpointing else 'No'}")


def print_formula_explanations(result: MemoryResult) -> None:
    """Print the formulas used (educational)."""
    print_subheader(f"Formulas ({result.mode} Mode)")
    for component, formula in result.breakdown.formula_explanations.items():
        name = component.replace("_", " ").title()
        print(f"  {name}: {formula}")


def print_level_header(level_num: int, scenario_name: str, attempt: int) -> None:
    """Print level header with attempt counter."""
    print(f"\n{'═' * 60}")
    print(f"  LEVEL {level_num}: {scenario_name}                    Attempt: {attempt}")
    print(f"{'═' * 60}")


def print_try_again(score: int, required: int = 50) -> None:
    """Print try again message when level not beaten."""
    print()
    print("  ╔════════════════════════════════════╗")
    print("  ║           TRY AGAIN!               ║")
    print(f"  ║   Need {required}+ points to advance       ║")
    print("  ╚════════════════════════════════════╝")


def print_level_complete(level_num: int, score: int, attempts: int) -> None:
    """Print celebration when level is beaten."""
    print()
    print("  ╔════════════════════════════════════╗")
    print("  ║        LEVEL COMPLETE!             ║")
    print(f"  ║   Score: {score:>3}  |  Attempts: {attempts:<3}     ║")
    print("  ╚════════════════════════════════════╝")


def print_game_over(total_score: int, rounds_played: int, high_scores: List[int]) -> None:
    """Print game over summary."""
    print_header("GAME OVER")

    avg_score = total_score / rounds_played if rounds_played > 0 else 0

    print(f"  Rounds Played: {rounds_played}")
    print(f"  Total Score:   {total_score}")
    print(f"  Average Score: {avg_score:.1f}")

    if high_scores:
        print()
        print("  Round Scores:")
        for i, score in enumerate(high_scores, 1):
            grade = _points_to_grade(score)
            print(f"    Round {i}: {score:>3} ({grade})")

    # Final assessment
    print()
    if avg_score >= 90:
        print("  Assessment: GPU Memory Master!")
    elif avg_score >= 75:
        print("  Assessment: Solid understanding of GPU memory.")
    elif avg_score >= 50:
        print("  Assessment: Getting there! Keep practicing.")
    else:
        print("  Assessment: Review the formulas and try again!")


def _points_to_grade(points: int) -> str:
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
    elif points > 0:
        return "F"
    else:
        return "OOM"


# Interactive prompts

def prompt_choice(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    """Interactive choice prompt with validation."""
    choice_str = "/".join(choices)
    default_hint = f" [{default}]" if default else ""

    while True:
        try:
            user_input = input(f"  {prompt} ({choice_str}){default_hint}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            sys.exit(0)

        if not user_input and default:
            return default

        # Allow partial matches
        matches = [c for c in choices if c.startswith(user_input)]
        if len(matches) == 1:
            return matches[0]

        if user_input in choices:
            return user_input

        print(f"  Invalid choice. Please enter one of: {choice_str}")


def prompt_int(
    prompt: str,
    min_val: int = 1,
    max_val: int = 10000,
    default: Optional[int] = None,
) -> int:
    """Interactive integer prompt with validation."""
    default_hint = f" [{default}]" if default is not None else ""

    while True:
        try:
            user_input = input(f"  {prompt} ({min_val}-{max_val}){default_hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            sys.exit(0)

        if not user_input and default is not None:
            return default

        try:
            value = int(user_input)
            if min_val <= value <= max_val:
                return value
            print(f"  Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("  Please enter a valid integer")


def prompt_continue(message: str = "Continue?", default: str = "yes") -> bool:
    """Ask if player wants to continue."""
    choice = prompt_choice(message, ["yes", "no"], default=default)
    return choice == "yes"


# ============================================================
# TIMED PROMPTS (for arcade mode)
# ============================================================

def print_timer_bar(timer: "RoundTimer") -> None:
    """Print the current timer status."""
    if timer and timer.running:
        bar = timer.render_bar()
        print(f"  {bar}")


def timed_prompt_choice(
    prompt: str,
    choices: List[str],
    default: str,
    timer: Optional["RoundTimer"] = None,
) -> tuple:
    """
    Interactive choice prompt with optional timer.

    Returns:
        (choice, timed_out) tuple
    """
    if timer is None or not timer.running:
        return (prompt_choice(prompt, choices, default), False)

    choice_str = "/".join(choices)
    default_hint = f" [{default}]"

    # Show timer before prompt
    print_timer_bar(timer)

    while not timer.expired:
        try:
            # Simple timed input - check timer periodically
            sys.stdout.write(f"  {prompt} ({choice_str}){default_hint}: ")
            sys.stdout.flush()

            # Use simple polling approach
            user_input = _get_input_with_timeout(timer)

            if user_input is None:
                # Timed out
                print()  # Newline after prompt
                return (default, True)

            user_input = user_input.strip().lower()

            if not user_input:
                return (default, False)

            # Allow partial matches
            matches = [c for c in choices if c.startswith(user_input)]
            if len(matches) == 1:
                return (matches[0], False)

            if user_input in choices:
                return (user_input, False)

            print(f"  Invalid choice. Please enter one of: {choice_str}")

        except (EOFError, KeyboardInterrupt):
            return (default, False)

    # Timer expired during validation loop
    return (default, True)


def timed_prompt_int(
    prompt: str,
    min_val: int,
    max_val: int,
    default: int,
    timer: Optional["RoundTimer"] = None,
) -> tuple:
    """
    Interactive integer prompt with optional timer.

    Returns:
        (value, timed_out) tuple
    """
    if timer is None or not timer.running:
        return (prompt_int(prompt, min_val, max_val, default), False)

    default_hint = f" [{default}]"

    # Show timer before prompt
    print_timer_bar(timer)

    while not timer.expired:
        try:
            sys.stdout.write(f"  {prompt} ({min_val}-{max_val}){default_hint}: ")
            sys.stdout.flush()

            user_input = _get_input_with_timeout(timer)

            if user_input is None:
                print()
                return (default, True)

            user_input = user_input.strip()

            if not user_input:
                return (default, False)

            try:
                value = int(user_input)
                if min_val <= value <= max_val:
                    return (value, False)
                print(f"  Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("  Please enter a valid integer")

        except (EOFError, KeyboardInterrupt):
            return (default, False)

    return (default, True)


def _get_input_with_timeout(timer: "RoundTimer") -> Optional[str]:
    """
    Get a line of input, respecting the timer.

    Returns None if timer expires before input is received.
    """
    import select

    buffer = []

    try:
        while timer.running and not timer.expired:
            # Check if input is ready (0.2 sec intervals)
            if hasattr(select, 'select'):
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready:
                    line = sys.stdin.readline()
                    return line.rstrip('\n')
            else:
                # Fallback: blocking read (timer won't interrupt)
                # This is a limitation on Windows
                return input()

        return None  # Timer expired

    except Exception:
        return None
