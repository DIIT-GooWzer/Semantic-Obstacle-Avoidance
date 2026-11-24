#!/bin/bash
set -euo pipefail

# Determine script and project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/hexapod_terrain.usd" ]]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/../hexapod_terrain.usd" ]]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    PROJECT_DIR="/home/user/Desktop/llmforhexapod"
fi

STAGE_USD="$PROJECT_DIR/hexapod_terrain.usd"
SCRIPTS_DIR="$PROJECT_DIR/Scripts"
CONTROLLER_SCRIPT="llmattempt2.py"
ISAAC_SIM_SH="${ISAACSIM_PATH:-/home/user/isaacsim}/isaac-sim.sh"

if [[ ! -x "$ISAAC_SIM_SH" ]]; then
    echo "Error: Isaac Sim launcher not found at: $ISAAC_SIM_SH" >&2
    exit 1
fi

if [[ ! -f "$STAGE_USD" ]]; then
    echo "Error: Terrain USD file not found at: $STAGE_USD" >&2
    exit 1
fi

if [[ ! -f "$SCRIPTS_DIR/$CONTROLLER_SCRIPT" ]]; then
    echo "Error: Controller script not found at: $SCRIPTS_DIR/$CONTROLLER_SCRIPT" >&2
    exit 1
fi

echo "=========================================================="
echo " Starting Isaac Sim with Autonomous Hexapod Controller"
echo "=========================================================="
echo " Stage:      $STAGE_USD"
echo " Scripts:    $SCRIPTS_DIR"
echo " Entry:      $CONTROLLER_SCRIPT"
echo " Threshold:  0.8 m"
echo " VLM Engine: LM Studio (Qwen 3.8 / port 1234)"
echo "=========================================================="

exec "$ISAAC_SIM_SH" \
    "$STAGE_USD" \
    --/app/python/scriptFolders/0="$SCRIPTS_DIR" \
    --exec "$CONTROLLER_SCRIPT" \
    "$@"
