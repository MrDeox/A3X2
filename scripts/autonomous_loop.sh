#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load environment variables (e.g., OPENROUTER_API_KEY)
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

VENV="${A3X_VENV:-.venv}"
AUTOPILOT_CYCLES="${A3X_AUTOPILOT_CYCLES:-3}"
GOAL_ROTATION="${A3X_GOAL_ROTATION:-seed/goal_rotation.yaml}"
BACKLOG="${A3X_BACKLOG:-seed/backlog.yaml}"
SEED_CONFIG="${A3X_SEED_CONFIG:-configs/sample.yaml}"
SEED_MAX="${A3X_SEED_MAX:-5}"
SEED_MAX_STEPS="${A3X_SEED_MAX_STEPS:-6}"
AUTOPILOT_INTERVAL="${A3X_AUTOPILOT_INTERVAL:-30}"
WATCH_INTERVAL="${A3X_WATCH_INTERVAL:-10}"

AUTOPILOT_BIN="$ROOT_DIR/$VENV/bin/a3x"

run_autopilot_cycle() {
  "$AUTOPILOT_BIN" autopilot \
    --cycles "$AUTOPILOT_CYCLES" \
    --goals "$GOAL_ROTATION" \
    --backlog "$BACKLOG" \
    --seed-default-config "$SEED_CONFIG" \
    --seed-max "$SEED_MAX" \
    --seed-max-steps "$SEED_MAX_STEPS"
}

run_seed_watch() {
  "$AUTOPILOT_BIN" seed watch \
    --backlog "$BACKLOG" \
    --config "$SEED_CONFIG" \
    --interval "$WATCH_INTERVAL" \
    --no-stop-when-idle
}

# Start seed watch in background loop
(
  while true; do
    run_seed_watch || true
    sleep "$WATCH_INTERVAL"
  done
) &
WATCH_PID=$!

trap 'echo "Stopping autonomous loop"; kill "$WATCH_PID" 2>/dev/null || true' EXIT

while true; do
  run_autopilot_cycle || true
  sleep "$AUTOPILOT_INTERVAL"
done
