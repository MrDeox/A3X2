"""Loop/daemon para executar seeds automaticamente sem intervenção humana."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .seed_runner import SeedRunner


@dataclass
class DaemonResult:
    runs: int
    completed: int
    failed: int


def run_loop(
    backlog: str | Path = "seed/backlog.yaml",
    *,
    config: str | Path = "configs/sample.yaml",
    interval: float = 30.0,
    max_runs: int | None = None,
    stop_when_idle: bool = True,
    log_path: str | Path | None = None,
) -> DaemonResult:
    runner = SeedRunner(backlog)
    log_file = None
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    runs = 0
    completed = 0
    failed = 0

    while True:
        result = runner.run_next(default_config=config)
        if result is None:
            if stop_when_idle:
                break
            _maybe_sleep(interval)
            continue
        runs += 1
        if result.completed:
            completed += 1
        else:
            failed += 1
        _log_entry(log_file, result)
        if max_runs is not None and runs >= max_runs:
            break
        if interval > 0:
            _maybe_sleep(interval)
    return DaemonResult(runs=runs, completed=completed, failed=failed)


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Daemon de execução contínua de seeds")
    parser.add_argument("--backlog", default="seed/backlog.yaml")
    parser.add_argument("--config", default="configs/sample.yaml")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--no-stop-when-idle", action="store_true")
    args = parser.parse_args(argv)

    res = run_loop(
        backlog=args.backlog,
        config=args.config,
        interval=args.interval,
        max_runs=args.max_runs,
        stop_when_idle=not args.no_stop_when_idle,
        log_path="seed/daemon.log",
    )
    print(f"runs={res.runs} completed={res.completed} failed={res.failed}")
    return 0 if res.failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))


def _maybe_sleep(interval: float) -> None:
    try:
        if interval > 0:
            time.sleep(interval)
    except KeyboardInterrupt:
        raise


def _log_entry(log_file: Path | None, result) -> None:
    if not log_file:
        return
    from datetime import datetime, timezone

    entry = (
        f"[{datetime.now(timezone.utc).isoformat()}] seed={result.seed_id}"
        f" status={result.status} completed={result.completed} notes={result.notes}"
    )
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(entry + "\n")
