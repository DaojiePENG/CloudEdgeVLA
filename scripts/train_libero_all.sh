#!/bin/bash
# Paired-frame dual-path training for CloudEdgeVLA on LIBERO.
#
# This is the launcher used for the released checkpoint
# `DaojiePENG/cloudedgevla-libero-spatial-object-goal-10`: one model trained
# jointly on all four LIBERO suites (Spatial, Object, Goal, Long) with a
# 21-frame window, i.e. delays up to 20 steps are in-distribution; d_max=40 in
# the paper is extrapolation.
#
# Required environment variables (no defaults, no machine-specific paths):
#   VLA_PATH       Base checkpoint to start from. The paper starts from the
#                  merged OpenVLA-OFT LIBERO checkpoint
#                  `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`.
#   DATA_ROOT_DIR  Directory containing the RLDS datasets (modified_libero_rlds).
#   RUN_ROOT_DIR   Output directory for checkpoints and logs.
#
# Optional:
#   NUM_GPUS           Number of GPUs for `torchrun` (default 8).
#   BATCH_SIZE         Per-GPU batch size (default 8). Paired-frame training
#                      runs two backbone forwards per step, so memory is roughly
#                      twice that of single-path OpenVLA-OFT: 8 needs ~90 GB
#                      (H200-class); use 4 on 80 GB cards.
#   MAX_STEPS          Optimization steps (default 100000; with 8 GPUs x
#                      batch 8 = effective batch 64 this is ~23.4 epochs over
#                      the 273,465 LIBERO transitions).
#   SAVE_FREQ          Checkpoint interval (default 5000).
#   SHUFFLE_BUFFER     Dataloader shuffle buffer (default 25000; lower it if
#                      the host runs out of RAM).
#   WANDB_ENTITY       W&B entity. If unset, W&B runs offline.
#   WANDB_PROJECT      W&B project name (default cloudedgevla).
#
# Flags:
#   --suite all|spatial|object|goal|10   Dataset to train on (default all).
#   --resume                             Resume the latest checkpoint of this run.
#   --run_id_note TEXT                   Suffix appended to the run id.
#
# Example:
#   VLA_PATH=.../openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
#   DATA_ROOT_DIR=.../modified_libero_rlds RUN_ROOT_DIR=.../runs \
#   NUM_GPUS=8 bash scripts/train_libero_all.sh
set -euo pipefail

SUITE="all"
RESUME="false"
RUN_ID_NOTE="libero_all_frame_delay_w21_eff64_lr5e4_100k"

while [[ $# -gt 0 ]]; do
    case $1 in
        --suite)      SUITE="$2"; shift 2 ;;
        --resume)     RESUME="true"; shift ;;
        --run_id_note) RUN_ID_NOTE="$2"; shift 2 ;;
        -h|--help)    sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

: "${VLA_PATH:?set VLA_PATH to the base OpenVLA-OFT checkpoint}"
: "${DATA_ROOT_DIR:?set DATA_ROOT_DIR to the directory containing modified_libero_rlds}"
: "${RUN_ROOT_DIR:?set RUN_ROOT_DIR to the checkpoint output directory}"

NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-25000}"
WANDB_PROJECT="${WANDB_PROJECT:-cloudedgevla}"

# The merged four-suite mixture is registered in
# prismatic/vla/datasets/rlds/oxe/mixtures.py; the per-suite names reproduce the
# paper's single-suite variants.
case "$SUITE" in
    all)     DATASET_NAME="libero_4_task_suites_no_noops" ;;
    spatial) DATASET_NAME="libero_spatial_no_noops" ;;
    object)  DATASET_NAME="libero_object_no_noops" ;;
    goal)    DATASET_NAME="libero_goal_no_noops" ;;
    10)      DATASET_NAME="libero_10_no_noops" ;;
    *) echo "ERROR: --suite must be one of all|spatial|object|goal|10" >&2; exit 1 ;;
esac

if [[ -z "${WANDB_ENTITY:-}" ]]; then
    echo "WANDB_ENTITY is not set; running Weights & Biases in offline mode."
    export WANDB_MODE="${WANDB_MODE:-offline}"
    WANDB_ENTITY="your-wandb-entity"
fi

RESUME_ARGS=()
if [[ "$RESUME" == "true" ]]; then
    RESUME_ARGS=(--resume true --resume_from_checkpoint auto --resume_root_dir "$RUN_ROOT_DIR")
fi

# 0.64 * max_steps follows the paper's learning-rate decay schedule.
DECAY_STEPS="${DECAY_STEPS:-$(( MAX_STEPS * 64 / 100 ))}"
# λ ramps from 0 to λ_max over the first 40% of training (paper recipe).
WARMUP_STEPS="${WARMUP_STEPS:-$(( MAX_STEPS * 40 / 100 ))}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================================"
echo "CloudEdgeVLA paired-frame training"
echo "  dataset        : $DATASET_NAME"
echo "  vla_path       : $VLA_PATH"
echo "  run_root_dir   : $RUN_ROOT_DIR"
echo "  gpus           : $NUM_GPUS x batch $BATCH_SIZE (effective batch $(( NUM_GPUS * BATCH_SIZE )))"
echo "  max_steps      : $MAX_STEPS (decay at $DECAY_STEPS)"
echo "  window / delay : window_size 21, training delays d ~ U{1..20}"
echo "  lambda schedule: 0 -> 0.5 over $WARMUP_STEPS steps"
echo "============================================================"

torchrun --nproc_per_node="$NUM_GPUS" --master_port="${MASTER_PORT:-29500}" vla-scripts/finetune.py \
    --vla_path "$VLA_PATH" \
    --data_root_dir "$DATA_ROOT_DIR" \
    --dataset_name "$DATASET_NAME" \
    --run_root_dir "$RUN_ROOT_DIR" \
    "${RESUME_ARGS[@]}" \
    --batch_size "$BATCH_SIZE" \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --lora_rank 16 \
    --lora_dropout 0.0 \
    --use_lora true \
    --image_aug true \
    --max_steps "$MAX_STEPS" \
    --num_steps_before_decay "$DECAY_STEPS" \
    --save_freq "$SAVE_FREQ" \
    --save_latest_checkpoint_only false \
    --shuffle_buffer_size "$SHUFFLE_BUFFER" \
    --num_images_in_input 2 \
    --use_proprio true \
    --use_l1_regression true \
    --use_diffusion false \
    --use_vision_action_head true \
    --action_head_vision_encoder siglip-base \
    --freeze_action_head_vision true \
    --action_head_num_views 2 \
    --use_frame_delay true \
    --window_size 21 \
    --stale_loss_lambda_max 0.5 \
    --stale_loss_warmup_steps "$WARMUP_STEPS" \
    --merge_lora_during_training true \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --run_id_note "$RUN_ID_NOTE"
