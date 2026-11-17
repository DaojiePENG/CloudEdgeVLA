#!/bin/bash
# LIBERO closed-loop evaluation of CloudEdgeVLA under simulated observation delay.
#
# For every (suite, delay) cell this runs `experiments/robot/libero/run_libero_eval.py`
# with the released model settings. Delay is sampled uniformly from {1, ..., d_max}
# per environment step on the cloud-side observation, while the edge keeps the
# current frame. d_max=0 disables delay sampling and is the synchronous reference.
#
# The checkpoint may be a local directory (merged weights + action head + proprio
# projector + dataset_statistics.json) or a Hugging Face Hub repo id, e.g.
#   DaojiePENG/cloudedgevla-libero-spatial-object-goal-10
#
# Usage:
#   bash scripts/eval_libero_delay_sweep.sh --checkpoint <dir-or-repo-id>
#
# Options:
#   --checkpoint PATH|ID   Required.
#   --suites CSV           Default libero_spatial,libero_object,libero_goal,libero_10.
#   --delays CSV           Default 0,10,20,40 (the paper's delay grid).
#   --num-trials N         Rollouts per task (default 10, the paper protocol).
#   --seed N               Evaluation seed (default 7; the paper uses 7, 8, 9).
#   --gpu ID               CUDA device for the whole sweep (default 0).
#   --out-dir DIR          Log and summary root (default eval_logs).
#   --save-rollouts BOOL   Encode rollout videos (default false).
#
# Already-finished cells are skipped, so an interrupted sweep can be restarted
# with the same command. Per-cell success rates are appended to
# <out-dir>/summary.csv; finished cells are marked with an empty <cell>.ok file.
#
# Environment:
#   MUJOCO_GL=egl|osmesa   Rendering backend (default egl). Use osmesa on
#                          machines without an EGL-capable GPU stack.
#   PYTHON                 Python interpreter (default python).
set -euo pipefail

CHECKPOINT=""
SUITES_CSV="libero_spatial,libero_object,libero_goal,libero_10"
DELAYS_CSV="0,10,20,40"
NUM_TRIALS=10
SEED=7
GPU_ID=0
OUT_DIR="eval_logs"
SAVE_ROLLOUTS="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)     CHECKPOINT="$2"; shift 2 ;;
        --suites)         SUITES_CSV="$2"; shift 2 ;;
        --delays)         DELAYS_CSV="$2"; shift 2 ;;
        --num-trials)     NUM_TRIALS="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --gpu)            GPU_ID="$2"; shift 2 ;;
        --out-dir)        OUT_DIR="$2"; shift 2 ;;
        --save-rollouts)  SAVE_ROLLOUTS="$2"; shift 2 ;;
        -h|--help)        sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required (local directory or HF Hub repo id)." >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"
EVAL_SCRIPT="experiments/robot/libero/run_libero_eval.py"
if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "ERROR: run this script from the repository root." >&2
    exit 1
fi

# Rendering backend. EGL needs a GPU; osmesa is software rendering.
MUJOCO_GL_MODE="${MUJOCO_GL:-egl}"
export MUJOCO_GL="$MUJOCO_GL_MODE"
if [[ "$MUJOCO_GL_MODE" == "osmesa" ]]; then
    export PYOPENGL_PLATFORM=osmesa
else
    export PYOPENGL_PLATFORM=egl
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# robosuite requires the EGL device id to be one of the ids exposed by
# CUDA_VISIBLE_DEVICES (raw bus ids, not the remapped ordinals). Override with
# EGL_DEVICE_ID on stacks whose EGL enumeration does not match the CUDA order,
# e.g. software mesa rendering, which exposes a single device.
export MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE_ID:-${GPU_ID%%,*}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

IFS=',' read -r -a SUITES <<< "$SUITES_CSV"
IFS=',' read -r -a DELAYS <<< "$DELAYS_CSV"

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "suite,delay,num_trials,seed,success_rate,episodes,log" > "$SUMMARY"
fi

echo "============================================================"
echo "CloudEdgeVLA delay sweep"
echo "  checkpoint : $CHECKPOINT"
echo "  suites     : ${SUITES[*]}"
echo "  delays     : ${DELAYS[*]}"
echo "  trials/task: $NUM_TRIALS   seed: $SEED   gpu: $GPU_ID"
echo "  backend    : MUJOCO_GL=$MUJOCO_GL_MODE"
echo "  output     : $OUT_DIR"
echo "============================================================"

for SUITE in "${SUITES[@]}"; do
    for DELAY in "${DELAYS[@]}"; do
        CELL="${SUITE}_d${DELAY}_s${SEED}_n${NUM_TRIALS}"
        CELL_DIR="$OUT_DIR/$CELL"
        mkdir -p "$CELL_DIR"

        if [[ -f "$CELL_DIR/.ok" ]]; then
            echo "[skip] $CELL (already finished)"
            continue
        fi

        if [[ "$DELAY" == "0" ]]; then
            DELAY_ARGS=(--use_frame_delay_eval false)
        else
            DELAY_ARGS=(--use_frame_delay_eval true --max_delay_steps_eval "$DELAY")
        fi

        echo "[run ] $CELL  $(date '+%Y-%m-%d %H:%M:%S')"
        set +e
        "$PYTHON" "$EVAL_SCRIPT" \
            --pretrained_checkpoint "$CHECKPOINT" \
            --model_family openvla \
            --task_suite_name "$SUITE" \
            --num_trials_per_task "$NUM_TRIALS" \
            --seed "$SEED" \
            --center_crop True \
            --num_open_loop_steps 8 \
            --num_images_in_input 2 \
            --use_proprio True \
            --use_l1_regression True \
            --use_diffusion False \
            --use_vision_action_head True \
            --action_head_vision_encoder siglip-base \
            --freeze_action_head_vision True \
            --action_head_num_views 2 \
            --lora_rank 16 \
            --save_rollouts "$SAVE_ROLLOUTS" \
            --local_log_dir "$CELL_DIR" \
            "${DELAY_ARGS[@]}" \
            > "$CELL_DIR/stdout.log" 2>&1
        RC=$?
        set -e

        LOG_FILE=$(ls "$CELL_DIR"/EVAL-"$SUITE"-*.txt 2>/dev/null | head -n 1 || true)
        if [[ $RC -ne 0 || -z "$LOG_FILE" || ! -f "$LOG_FILE" ]]; then
            echo "[fail] $CELL (exit $RC); see $CELL_DIR/stdout.log" >&2
            continue
        fi
        if grep -q "INVALID RESULT" "$LOG_FILE"; then
            echo "[fail] $CELL: episodes raised during evaluation; see $LOG_FILE" >&2
            continue
        fi

        python3 - "$SUITE" "$DELAY" "$NUM_TRIALS" "$SEED" "$LOG_FILE" "$SUMMARY" <<'PY'
import re, sys

suite, delay, trials, seed, log_path, summary_path = sys.argv[1:7]
text = open(log_path).read()
match = re.search(r"Overall success rate: ([0-9.]+)", text)
episodes = re.search(r"Total episodes: (\d+)", text)
if not match:
    raise SystemExit(f"no 'Overall success rate' line in {log_path}")
with open(summary_path, "a") as fh:
    fh.write(f"{suite},{delay},{trials},{seed},{float(match.group(1)):.4f},"
             f"{episodes.group(1) if episodes else ''},{log_path}\n")
print(f"       success rate: {match.group(1)}")
PY

        touch "$CELL_DIR/.ok"
        echo "[done] $CELL"
    done
done

echo "============================================================"
echo "Sweep finished. Results: $SUMMARY"
echo "============================================================"
