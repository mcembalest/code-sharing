"""Main CLI entry point for OOM: The Game."""

import argparse
import sys
from typing import Optional, List

from .game import TrainingConfig, ParallelismStrategy, Precision
from .memory.calculator import MemoryCalculator
from .scoring import calculate_score
from .configs.scenarios import (
    SCENARIOS,
    Scenario,
    get_scenario,
    get_scenarios_by_difficulty,
    list_scenarios,
)
from .display import (
    print_header,
    print_scenario,
    print_memory_breakdown,
    print_config_summary,
    print_formula_explanations,
    print_game_over,
    print_level_header,
    print_try_again,
    print_level_complete,
    print_timer_bar,
    prompt_choice,
    prompt_int,
    prompt_continue,
    timed_prompt_choice,
    timed_prompt_int,
)
from .timer import RoundTimer, print_times_up

# Game constants
PASS_THRESHOLD = 50  # Minimum score to beat a level
DEFAULT_TIMER_SECONDS = 30  # Timer per round in arcade mode


def get_training_config(
    scenario: Scenario,
    timer: Optional[RoundTimer] = None,
) -> tuple:
    """
    Interactive prompts to get player's training configuration.

    Returns:
        (TrainingConfig, timed_out) tuple
    """
    print("\nConfigure your training run:")
    print("─" * 40)

    timed_out = False

    # Show initial timer status if using timer
    if timer and timer.running:
        print_timer_bar(timer)

    # Parallelism strategy
    if scenario.num_gpus == 1:
        parallelism = ParallelismStrategy.NONE
        print(f"  Parallelism: Single GPU (fixed)")
    else:
        allowed = scenario.allowed_parallelism or ["ddp", "fsdp"]
        if timer:
            choice, timed_out = timed_prompt_choice("Parallelism", allowed, "fsdp", timer)
        else:
            choice = prompt_choice("Parallelism", allowed, default="fsdp")
        parallelism = ParallelismStrategy(choice)

    if timed_out:
        print_times_up()
        return _build_default_config(scenario), True

    # Precision
    if timer:
        precision_choice, timed_out = timed_prompt_choice(
            "Precision", ["fp32", "fp16", "bf16"], "bf16", timer
        )
    else:
        precision_choice = prompt_choice("Precision", ["fp32", "fp16", "bf16"], default="bf16")
    precision = Precision(precision_choice)

    if timed_out:
        print_times_up()
        return _build_default_config(scenario), True

    # Batch size
    min_batch = scenario.min_batch_size or 1
    max_batch = scenario.max_batch_size or 64
    if timer:
        batch_size, timed_out = timed_prompt_int(
            "Per-GPU batch size", min_batch, max_batch, 4, timer
        )
    else:
        batch_size = prompt_int("Per-GPU batch size", min_batch, max_batch, default=4)

    if timed_out:
        print_times_up()
        return _build_default_config(scenario), True

    # Sequence length
    min_seq = scenario.min_seq_len or 128
    max_seq = scenario.max_seq_len or 8192
    default_seq = max(min_seq, 2048) if min_seq > 2048 else 2048
    if timer:
        seq_len, timed_out = timed_prompt_int(
            "Sequence length", min_seq, max_seq, default_seq, timer
        )
    else:
        seq_len = prompt_int("Sequence length", min_seq, max_seq, default=default_seq)

    if timed_out:
        print_times_up()
        return _build_default_config(scenario), True

    # Gradient checkpointing
    if scenario.require_gradient_checkpointing is not None:
        gradient_checkpointing = scenario.require_gradient_checkpointing
        print(f"  Gradient checkpointing: {'Yes' if gradient_checkpointing else 'No'} (fixed)")
    else:
        if timer:
            gc_choice, timed_out = timed_prompt_choice(
                "Gradient checkpointing", ["yes", "no"], "no", timer
            )
        else:
            gc_choice = prompt_choice("Gradient checkpointing", ["yes", "no"], default="no")
        gradient_checkpointing = gc_choice == "yes"

    if timed_out:
        print_times_up()
        return _build_default_config(scenario), True

    # Stop timer now that all inputs are collected
    if timer:
        timer.stop()

    config = TrainingConfig(
        num_gpus=scenario.num_gpus,
        parallelism=parallelism,
        batch_size=batch_size,
        seq_len=seq_len,
        precision=precision,
        gradient_checkpointing=gradient_checkpointing,
    )
    return config, False


def _build_default_config(scenario: Scenario) -> TrainingConfig:
    """Build a default training config for timeout scenarios."""
    return TrainingConfig(
        num_gpus=scenario.num_gpus,
        parallelism=ParallelismStrategy.FSDP if scenario.num_gpus > 1 else ParallelismStrategy.NONE,
        batch_size=4,
        seq_len=2048,
        precision=Precision.BF16,
        gradient_checkpointing=False,
    )


def play_round(
    scenario: Scenario,
    calculator: MemoryCalculator,
    show_formulas: bool = False,
    use_timer: bool = False,
    timer_seconds: int = DEFAULT_TIMER_SECONDS,
) -> int:
    """
    Play a single round of the game.

    Returns:
        Score points achieved (0-100+)
    """
    print_scenario(scenario)

    # Create timer if enabled
    timer = None
    if use_timer:
        timer = RoundTimer(seconds=timer_seconds)
        timer.start()

    # Get player's configuration
    config, timed_out = get_training_config(scenario, timer)

    # Stop timer if still running
    if timer:
        timer.stop()

    # Validate configuration
    errors = config.validate()
    if errors:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"  - {error}")
        return 0

    # Show what they configured
    print_config_summary(config, scenario)

    # Calculate memory
    result = calculator.calculate(
        gpu=scenario.gpu,
        model=scenario.model,
        training=config,
    )

    # Calculate score
    score = calculate_score(result, scenario.target_utilization)

    # Display results
    print_memory_breakdown(result, score)

    # Optionally show formulas
    if show_formulas:
        print_formula_explanations(result)

    return score.points


def play_level(
    level_num: int,
    scenario: Scenario,
    calculator: MemoryCalculator,
    show_formulas: bool = False,
    use_timer: bool = False,
    timer_seconds: int = DEFAULT_TIMER_SECONDS,
) -> tuple:
    """
    Play a single level with retry until beaten.

    Returns:
        (final_score, attempts) tuple
    """
    attempt = 1

    while True:
        # Print level header with attempt counter
        print_level_header(level_num, scenario.description.split(":")[0], attempt)

        # Play the round
        score = play_round(
            scenario,
            calculator,
            show_formulas=show_formulas,
            use_timer=use_timer,
            timer_seconds=timer_seconds,
        )

        # Check if level beaten
        if score >= PASS_THRESHOLD:
            print_level_complete(level_num, score, attempt)
            return (score, attempt)

        # Not beaten - try again
        print_try_again(score, PASS_THRESHOLD)

        # Ask to retry or quit
        print()
        retry = prompt_choice("Try again?", ["yes", "quit"], default="yes")
        if retry == "quit":
            return (score, attempt)

        attempt += 1


def play_game(
    mode: str = "easy",
    difficulty: Optional[str] = None,
    scenario_id: Optional[str] = None,
    show_formulas: bool = False,
    use_timer: bool = False,
    timer_seconds: int = DEFAULT_TIMER_SECONDS,
) -> None:
    """
    Main game loop.

    Args:
        mode: Formula difficulty ("easy" or "hard")
        difficulty: Scenario difficulty filter (tutorial/easy/medium/hard/expert)
        scenario_id: Specific scenario to play
        show_formulas: Whether to display formulas after each round
        use_timer: Whether to enable per-round countdown timer
        timer_seconds: Seconds per round when timer is enabled
    """
    calculator = MemoryCalculator(mode=mode)

    print_header("OOM: THE GAME")
    print(f"  Formula Mode: {mode.upper()}")
    if use_timer:
        print(f"  Timer: {timer_seconds}s per round (ARCADE MODE)")
    print(f"  Goal: Maximize GPU utilization without going OOM!")
    print(f"  Target: 95% utilization = perfect score")
    print()
    print("  Rules:")
    print(f"    - Score {PASS_THRESHOLD}+ points to beat each level")
    print("    - Retry until you pass or quit")
    if use_timer:
        print(f"    - {timer_seconds} seconds to configure each attempt!")
    print()
    print("  Scoring:")
    print("    S (95+): Expert   |  A (85-94): Great")
    print("    B (70-84): Good   |  C (50-69): Okay")
    print("    D (25-49): Poor   |  F (<25): Fail")
    print("    OOM: Crashed!")

    total_score = 0
    levels_beaten = 0
    total_attempts = 0
    level_results: List[tuple] = []  # (score, attempts) per level

    # Determine scenarios to play
    if scenario_id:
        scenarios = [get_scenario(scenario_id)]
    elif difficulty:
        scenarios = get_scenarios_by_difficulty(difficulty)
        if not scenarios:
            print(f"\nNo scenarios found for difficulty: {difficulty}")
            return
    else:
        # Default: play tutorial + easy
        scenarios = (
            get_scenarios_by_difficulty("tutorial")
            + get_scenarios_by_difficulty("easy")
        )

    # Play each level (with retry loop)
    for i, scenario in enumerate(scenarios):
        level_num = i + 1

        score, attempts = play_level(
            level_num=level_num,
            scenario=scenario,
            calculator=calculator,
            show_formulas=show_formulas,
            use_timer=use_timer,
            timer_seconds=timer_seconds,
        )

        level_results.append((score, attempts))
        total_score += score
        total_attempts += attempts

        if score >= PASS_THRESHOLD:
            levels_beaten += 1

            # Ask to continue to next level if not the last
            if i < len(scenarios) - 1:
                print()
                if not prompt_continue("Continue to next level?"):
                    break
        else:
            # Player quit the level
            break

    # Game over summary
    print_game_over_arcade(levels_beaten, len(scenarios), total_score, level_results)


def print_game_over_arcade(
    levels_beaten: int,
    total_levels: int,
    total_score: int,
    level_results: List[tuple],
) -> None:
    """Print arcade mode game over summary."""
    print_header("GAME OVER")

    print(f"  Levels Beaten: {levels_beaten}/{total_levels}")
    print(f"  Total Score:   {total_score}")

    if level_results:
        print()
        print("  Level Results:")
        for i, (score, attempts) in enumerate(level_results, 1):
            status = "PASS" if score >= PASS_THRESHOLD else "FAIL"
            print(f"    Level {i}: {score:>3} pts ({attempts} attempt{'s' if attempts > 1 else ''}) [{status}]")

    # Final assessment
    print()
    if levels_beaten == total_levels:
        print("  Assessment: GPU Memory Master! All levels beaten!")
    elif levels_beaten >= total_levels * 0.75:
        print("  Assessment: Excellent! Almost there!")
    elif levels_beaten >= total_levels * 0.5:
        print("  Assessment: Good progress! Keep practicing!")
    else:
        print("  Assessment: Keep learning the formulas!")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OOM: The Game - Learn GPU memory estimation through gameplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  oom                           # Play tutorial + easy scenarios
  oom --difficulty medium       # Play medium difficulty scenarios
  oom --scenario hard_1         # Play a specific scenario
  oom --mode hard               # Use realistic memory formulas
  oom --list                    # List all available scenarios
  oom --formulas                # Show formulas after each round
        """,
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["easy", "hard"],
        default="easy",
        help="Formula complexity (easy=simplified, hard=realistic)",
    )

    parser.add_argument(
        "--difficulty", "-d",
        choices=["tutorial", "easy", "medium", "hard", "expert"],
        help="Play scenarios of a specific difficulty",
    )

    parser.add_argument(
        "--scenario", "-s",
        type=str,
        help="Play a specific scenario by ID",
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available scenarios",
    )

    parser.add_argument(
        "--formulas", "-f",
        action="store_true",
        help="Show formulas used after each round (educational)",
    )

    parser.add_argument(
        "--timer", "-t",
        action="store_true",
        help="Enable arcade mode with countdown timer per round",
    )

    parser.add_argument(
        "--timer-seconds",
        type=int,
        default=DEFAULT_TIMER_SECONDS,
        help=f"Seconds per round in timer mode (default: {DEFAULT_TIMER_SECONDS})",
    )

    args = parser.parse_args()

    if args.list:
        print_header("AVAILABLE SCENARIOS")
        list_scenarios()
        return

    try:
        play_game(
            mode=args.mode,
            difficulty=args.difficulty,
            scenario_id=args.scenario,
            show_formulas=args.formulas,
            use_timer=args.timer,
            timer_seconds=args.timer_seconds,
        )
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
