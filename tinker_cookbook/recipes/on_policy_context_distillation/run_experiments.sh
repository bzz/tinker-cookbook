#!/usr/bin/env bash
# On-Policy Context Distillation Experiments
#
# Compares three approaches to context distillation for language classification:
#   1. Off-policy only (SL on teacher data)
#   2. On-policy only (KL against teacher with context)
#   3. Off-policy followed by on-policy
#
# Prerequisites:
#   - Tinker API key set in environment
#   - tinker and tinker_cookbook installed

set -euo pipefail

DATA_DIR="${DATA_DIR:-/tmp/tinker-datasets/context_distillation}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/tinker-results/context_distillation}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
RENDERER="${RENDERER:-qwen3_disable_thinking}"
WANDB_PROJECT="${WANDB_PROJECT:-context_distillation}"
LORA_RANK="${LORA_RANK:-32}"
LR="${LR:-1e-4}"

mkdir -p "$DATA_DIR" "$RESULTS_DIR"

echo "============================================"
echo "Step 0: Generate data"
echo "============================================"
python -m tinker_cookbook.recipes.on_policy_context_distillation.create_data \
    output_dir="$DATA_DIR" \
    model_name="$MODEL" \
    renderer_name="$RENDERER"

echo ""
echo "============================================"
echo "Step 1: Evaluate base model (no training)"
echo "============================================"
python -m tinker_cookbook.recipes.on_policy_context_distillation.eval \
    model_name="$MODEL" \
    renderer_name="$RENDERER" \
    eval_file="$DATA_DIR/eval_prompts.jsonl" \
    output_file="$RESULTS_DIR/eval_base.json" \
    label=base

echo ""
echo "============================================"
echo "Experiment 1: Off-policy only"
echo "============================================"
python -m tinker_cookbook.recipes.on_policy_context_distillation.train_off_policy \
    file_path="$DATA_DIR/train.jsonl" \
    model_name="$MODEL" \
    renderer_name="$RENDERER" \
    learning_rate="$LR" \
    lora_rank="$LORA_RANK" \
    batch_size=64 \
    num_epochs=4 \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="exp1-offpolicy" \
    behavior_if_log_dir_exists=overwrite

# NOTE: After training, get the checkpoint path from the training logs.
# Set OFF_POLICY_CHECKPOINT to the path printed at the end of training.
# Example: OFF_POLICY_CHECKPOINT="tinker://<id>/weights/final"
echo "Set OFF_POLICY_CHECKPOINT to the path from training logs, then continue."
echo "Example: export OFF_POLICY_CHECKPOINT='tinker://<id>/weights/final'"

echo ""
echo "============================================"
echo "Experiment 1: Evaluate off-policy model"
echo "============================================"
if [ -n "${OFF_POLICY_CHECKPOINT:-}" ]; then
    python -m tinker_cookbook.recipes.on_policy_context_distillation.eval \
        model_name="$MODEL" \
        renderer_name="$RENDERER" \
        load_checkpoint_path="$OFF_POLICY_CHECKPOINT" \
        eval_file="$DATA_DIR/eval_prompts.jsonl" \
        output_file="$RESULTS_DIR/eval_offpolicy.json" \
        label=off_policy
fi

echo ""
echo "============================================"
echo "Experiment 2: On-policy only"
echo "============================================"
python -m tinker_cookbook.recipes.on_policy_context_distillation.train_on_policy \
    model_name="$MODEL" \
    renderer_name="$RENDERER" \
    teacher_model="$MODEL" \
    prompts_file="$DATA_DIR/train_prompts.jsonl" \
    learning_rate="$LR" \
    lora_rank="$LORA_RANK" \
    groups_per_batch=64 \
    group_size=4 \
    max_step=50 \
    kl_penalty_coef=1.0 \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="exp2-onpolicy" \
    behavior_if_log_dir_exists=overwrite

echo "Set ON_POLICY_CHECKPOINT to the path from training logs, then continue."

echo ""
echo "============================================"
echo "Experiment 2: Evaluate on-policy model"
echo "============================================"
if [ -n "${ON_POLICY_CHECKPOINT:-}" ]; then
    python -m tinker_cookbook.recipes.on_policy_context_distillation.eval \
        model_name="$MODEL" \
        renderer_name="$RENDERER" \
        load_checkpoint_path="$ON_POLICY_CHECKPOINT" \
        eval_file="$DATA_DIR/eval_prompts.jsonl" \
        output_file="$RESULTS_DIR/eval_onpolicy.json" \
        label=on_policy
fi

echo ""
echo "============================================"
echo "Experiment 3: Off-policy then on-policy"
echo "============================================"
if [ -n "${OFF_POLICY_CHECKPOINT:-}" ]; then
    python -m tinker_cookbook.recipes.on_policy_context_distillation.train_on_policy \
        model_name="$MODEL" \
        renderer_name="$RENDERER" \
        teacher_model="$MODEL" \
        load_checkpoint_path="$OFF_POLICY_CHECKPOINT" \
        prompts_file="$DATA_DIR/train_prompts.jsonl" \
        learning_rate="$LR" \
        lora_rank="$LORA_RANK" \
        groups_per_batch=64 \
        group_size=4 \
        max_step=50 \
        kl_penalty_coef=1.0 \
        wandb_project="$WANDB_PROJECT" \
        wandb_name="exp3-combined" \
        behavior_if_log_dir_exists=overwrite

    echo "Set COMBINED_CHECKPOINT to the path from training logs, then continue."
fi

echo ""
echo "============================================"
echo "Experiment 3: Evaluate combined model"
echo "============================================"
if [ -n "${COMBINED_CHECKPOINT:-}" ]; then
    python -m tinker_cookbook.recipes.on_policy_context_distillation.eval \
        model_name="$MODEL" \
        renderer_name="$RENDERER" \
        load_checkpoint_path="$COMBINED_CHECKPOINT" \
        eval_file="$DATA_DIR/eval_prompts.jsonl" \
        output_file="$RESULTS_DIR/eval_combined.json" \
        label=combined
fi

echo ""
echo "============================================"
echo "Done! Check results in $RESULTS_DIR"
echo "============================================"
