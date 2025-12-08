"""Timer module for arcade mode countdown."""

import sys
import threading
import time
from typing import Optional, Callable

# ANSI escape codes for terminal control
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[1A"
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"


class RoundTimer:
    """
    Countdown timer for arcade mode.

    Runs in a background thread and can trigger a callback when expired.
    """

    def __init__(self, seconds: int = 30, on_expire: Optional[Callable] = None):
        self.total_seconds = seconds
        self.remaining = seconds
        self.expired = False
        self.running = False
        self.on_expire = on_expire
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._display_enabled = True

    def start(self) -> None:
        """Start the countdown timer in a background thread."""
        self.remaining = self.total_seconds
        self.expired = False
        self.running = True
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._countdown, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the timer."""
        self.running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)

    def pause_display(self) -> None:
        """Temporarily pause timer display updates."""
        self._display_enabled = False

    def resume_display(self) -> None:
        """Resume timer display updates."""
        self._display_enabled = True

    def _countdown(self) -> None:
        """Background countdown loop."""
        while self.remaining > 0 and not self._stop_event.is_set():
            self._stop_event.wait(1.0)
            if not self._stop_event.is_set():
                self.remaining -= 1

        if self.remaining <= 0 and not self._stop_event.is_set():
            self.expired = True
            self.running = False
            if self.on_expire:
                self.on_expire()

    def render_bar(self, width: int = 30) -> str:
        """Render the timer bar as a string."""
        if self.total_seconds == 0:
            return ""

        fraction = self.remaining / self.total_seconds
        filled = int(width * fraction)
        empty = width - filled

        # Color coding based on time remaining
        if self.remaining <= 5:
            color = "\033[91m"  # Red
        elif self.remaining <= 10:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[92m"  # Green

        reset = "\033[0m"

        bar = f"{color}{'█' * filled}{'░' * empty}{reset}"
        return f"⏱ [{bar}] {self.remaining}s"


def timed_input(
    prompt: str,
    timeout: int,
    default: str,
    timer: Optional[RoundTimer] = None,
) -> tuple:
    """
    Get input with a timeout.

    Returns:
        (user_input, timed_out) tuple
    """
    if timer is None:
        # No timer - just regular input
        try:
            result = input(prompt)
            return (result if result else default, False)
        except (EOFError, KeyboardInterrupt):
            return (default, False)

    # Check if timer already expired
    if timer.expired:
        return (default, True)

    # For simplicity in v1, we'll use a polling approach
    # This isn't ideal but works cross-platform
    import select

    try:
        # Print prompt
        sys.stdout.write(prompt)
        sys.stdout.flush()

        # Use select for Unix, fallback for others
        if hasattr(select, 'select'):
            result = _timed_input_select(timer, default)
        else:
            result = _timed_input_fallback(timer, default)

        return result

    except Exception:
        return (default, timer.expired)


def _timed_input_select(timer: RoundTimer, default: str) -> tuple:
    """Unix implementation using select."""
    import select

    buffer = []

    while timer.running and not timer.expired:
        # Check if input is available (0.5 sec timeout)
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)

        if ready:
            char = sys.stdin.read(1)
            if char == '\n':
                result = ''.join(buffer).strip()
                return (result if result else default, False)
            buffer.append(char)

    # Timer expired
    if buffer:
        result = ''.join(buffer).strip()
        return (result if result else default, True)
    return (default, True)


def _timed_input_fallback(timer: RoundTimer, default: str) -> tuple:
    """Fallback implementation using threading."""
    result = [default]
    done = threading.Event()

    def read_input():
        try:
            line = sys.stdin.readline().strip()
            result[0] = line if line else default
        except:
            pass
        finally:
            done.set()

    input_thread = threading.Thread(target=read_input, daemon=True)
    input_thread.start()

    # Wait for either input or timer expiry
    while timer.running and not timer.expired:
        if done.wait(timeout=0.5):
            break

    return (result[0], timer.expired and not done.is_set())


def print_timer_status(timer: RoundTimer) -> None:
    """Print current timer status."""
    if timer and timer.running:
        bar = timer.render_bar()
        print(f"  {bar}")


def print_times_up() -> None:
    """Print time's up message."""
    print()
    print("  \033[91m⏱ TIME'S UP! Using defaults...\033[0m")
    print()
