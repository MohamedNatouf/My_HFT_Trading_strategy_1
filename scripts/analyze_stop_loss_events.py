import argparse
from pathlib import Path


def analyze_log(path: Path) -> None:
    if not path.exists():
        print(f"Log file not found: {path}")
        return

    pending_event = None  # type: ignore[assignment]
    last_rebalance_line = None
    last_alloc_line = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.rstrip("\n")

            # Track last rebalance and allocation lines for context
            if "rebalance_stats" in stripped:
                last_rebalance_line = (line_no, stripped)
            elif "strategy_allocation" in stripped:
                # If we were waiting for the post-stop-loss allocation, use this
                if pending_event is not None:
                    trigger_line_no, trigger_line, before_reb, before_alloc = pending_event
                    print("=" * 120)
                    print(f"STOP-LOSS EVENT (log: {path}, trigger_line: {trigger_line_no})")
                    print("Trigger:")
                    print(f"  {trigger_line}")
                    if before_reb is not None:
                        print("Before (last rebalance_stats before trigger):")
                        print(f"  [line {before_reb[0]}] {before_reb[1]}")
                    if before_alloc is not None:
                        print("Before (last strategy_allocation before trigger):")
                        print(f"  [line {before_alloc[0]}] {before_alloc[1]}")
                    print("After (first strategy_allocation after trigger):")
                    print(f"  [line {line_no}] {stripped}")
                    print("=" * 120)
                    print()
                    pending_event = None
                # Update last allocation regardless
                last_alloc_line = (line_no, stripped)
                continue

            # Detect allocation stop-loss trigger
            if "allocation_stop_loss_triggered" in stripped:
                pending_event = (
                    line_no,
                    stripped,
                    last_rebalance_line,
                    last_alloc_line,
                )

    if pending_event is not None:
        trigger_line_no, trigger_line, before_reb, before_alloc = pending_event
        print("=" * 120)
        print(f"STOP-LOSS EVENT WITHOUT FOLLOWING ALLOCATION (log: {path}, trigger_line: {trigger_line_no})")
        print("Trigger:")
        print(f"  {trigger_line}")
        if before_reb is not None:
            print("Before (last rebalance_stats before trigger):")
            print(f"  [line {before_reb[0]}] {before_reb[1]}")
        if before_alloc is not None:
            print("Before (last strategy_allocation before trigger):")
            print(f"  [line {before_alloc[0]}] {before_alloc[1]}")
        print("(No subsequent strategy_allocation line was found in this log.)")
        print("=" * 120)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a trading log for allocation_stop_loss_triggered events and "
            "print each with before/after allocation context."
        )
    )
    parser.add_argument(
        "logfile",
        type=str,
        help="Path to log file (e.g. logs/live_01.log or logs/sweep_*.log)",
    )
    args = parser.parse_args()
    analyze_log(Path(args.logfile))


if __name__ == "__main__":
    main()
