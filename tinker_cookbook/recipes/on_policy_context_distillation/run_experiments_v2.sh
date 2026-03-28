#!/usr/bin/env bash
# On-Policy Context Distillation v2 Experiments
#
# v2 design: Student sees task definition + output format (STUDENT_CONTEXT).
# Teacher additionally sees detailed classification instructions (TEACHER_ONLY_INSTRUCTIONS).
# This fixes the v1 issue where the student had no idea it was doing classification.
#
# Re-uses off-policy results from v1 (same checkpoint).
#
# Experiments:
#   2v2. On-policy only (student sees task def, teacher sees task def + instructions)
#   3v2. Off-policy → on-policy (load v1 off-policy checkpoint, then on-policy v2)
#
# Prerequisites:
#   - Tinker API key set in environment
#   - tinker and tinker_cookbook installed
#   - Data already generated via create_data.py (from v1)

set -euo pipefail

DATA_DIR="${DATA_DIR:-/tmp/tinker-datasets/context_distillation}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/tinker-results/context_distillation_v2}"
LOG_DIR="${LOG_DIR:-$HOME/tinker-examples/context_distillation_v2}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
RENDERER="${RENDERER:-qwen3_disable_thinking}"
WANDB_PROJECT="${WANDB_PROJECT:-context_distillation_v2}"
LORA_RANK="${LORA_RANK:-32}"
LR="${LR:-1e-4}"

# v1 off-policy checkpoint (reused, not re-trained)
OFF_POLICY_CHECKPOINT="${OFF_POLICY_CHECKPOINT:-tinker://1b910180-56eb-534a-bcf1-335be750c89e:train:0/sampler_weights/final}"

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$LOG_DIR"

# Ensure data exists
if [ ! -f "$DATA_DIR/train_prompts.jsonl" ]; then
    echo "============================================"
    echo "Step 0: Generate data (needed for on-policy)"
    echo "============================================"
    python -m tinker_cookbook.recipes.on_policy_context_distillation.create_data \
        output_dir="$DATA_DIR" \
        model_name="$MODEL" \
        renderer_name="$RENDERER"
fi

echo ""
echo "============================================"
echo "Experiment 2v2: On-policy only (v2 design)"
echo "============================================"
echo "Student sees: task definition + output format (STUDENT_CONTEXT)"
echo "Teacher sees: STUDENT_CONTEXT + detailed instructions"
echo ""

python -m tinker_cookbook.recipes.on_policy_context_distillation.train_on_policy_v2 \
    model_name="$MODEL" \
    renderer_name="$RENDERER" \
    teacher_model="$MODEL" \
    prompts_file="$DATA_DIR/train_prompts.jsonl" \
    learning_rate="$LR" \
    lora_rank="$LORA_RANK" \
    groups_per_batch=64 \
    group_size=4 \
    max_step=26 \
    kl_penalty_coef=1.0 \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="exp2v2-onpolicy" \
    log_path="$LOG_DIR/exp2v2-onpolicy" \
    behavior_if_log_dir_exists=overwrite \
    2>&1 | tee "$RESULTS_DIR/train_exp2v2.log"

echo ""
echo "Set ON_POLICY_V2_CHECKPOINT to the checkpoint path, then continue."
echo "Example: export ON_POLICY_V2_CHECKPOINT='tinker://<id>/sampler_weights/final'"

echo ""
echo "============================================"
echo "Experiment 2v2: Evaluate on-policy v2 model"
echo "============================================"
if [ -n "${ON_POLICY_V2_CHECKPOINT:-}" ]; then
    python -m tinker_cookbook.recipes.on_policy_context_distillation.eval_v2 \
        model_name="$MODEL" \
        renderer_name="$RENDERER" \
        load_checkpoint_path="$ON_POLICY_V2_CHECKPOINT" \
        eval_file="$DATA_DIR/eval_prompts.jsonl" \
        output_file="$RESULTS_DIR/eval_onpolicy_v2.json" \
        label=on_policy_v2 \
        2>&1 | tee "$RESULTS_DIR/eval_exp2v2.log"
fi

echo ""
echo "============================================"
echo "Experiment 3v2: Off-policy → on-policy (v2)"
echo "============================================"
echo "Loading off-policy checkpoint: $OFF_POLICY_CHECKPOINT"
echo ""

python -m tinker_cookbook.recipes.on_policy_context_distillation.train_on_policy_v2 \
    model_name="$MODEL" \
    renderer_name="$RENDERER" \
    teacher_model="$MODEL" \
    load_checkpoint_path="$OFF_POLICY_CHECKPOINT" \
    prompts_file="$DATA_DIR/train_prompts.jsonl" \
    learning_rate="$LR" \
    lora_rank="$LORA_RANK" \
    groups_per_batch=64 \
    group_size=4 \
    max_step=26 \
    kl_penalty_coef=1.0 \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="exp3v2-combined" \
    log_path="$LOG_DIR/exp3v2-combined" \
    behavior_if_log_dir_exists=overwrite \
    2>&1 | tee "$RESULTS_DIR/train_exp3v2.log"

echo ""
echo "Set COMBINED_V2_CHECKPOINT to the checkpoint path, then continue."

echo ""
echo "============================================"
echo "Experiment 3v2: Evaluate combined v2 model"
echo "============================================"
if [ -n "${COMBINED_V2_CHECKPOINT:-}" ]; then
    python -m tinker_cookbook.recipes.on_policy_context_distillation.eval_v2 \
        model_name="$MODEL" \
        renderer_name="$RENDERER" \
        load_checkpoint_path="$COMBINED_V2_CHECKPOINT" \
        eval_file="$DATA_DIR/eval_prompts.jsonl" \
        output_file="$RESULTS_DIR/eval_combined_v2.json" \
        label=combined_v2 \
        2>&1 | tee "$RESULTS_DIR/eval_exp3v2.log"
fi

echo ""
echo "============================================"
echo "Also evaluate base model and off-policy with v2 eval (task context)"
echo "============================================"
python -m tinker_cookbook.recipes.on_policy_context_distillation.eval_v2 \
    model_name="$MODEL" \
    renderer_name="$RENDERER" \
    eval_file="$DATA_DIR/eval_prompts.jsonl" \
    output_file="$RESULTS_DIR/eval_base_v2.json" \
    label=base_v2 \
    2>&1 | tee "$RESULTS_DIR/eval_base_v2.log"

python -m tinker_cookbook.recipes.on_policy_context_distillation.eval_v2 \
    model_name="$MODEL" \
    renderer_name="$RENDERER" \
    load_checkpoint_path="$OFF_POLICY_CHECKPOINT" \
    eval_file="$DATA_DIR/eval_prompts.jsonl" \
    output_file="$RESULTS_DIR/eval_offpolicy_v2.json" \
    label=off_policy_v2 \
    2>&1 | tee "$RESULTS_DIR/eval_offpolicy_v2.log"

echo ""
echo "============================================"
echo "Done! Results in $RESULTS_DIR"
echo "============================================"
ls -la "$RESULTS_DIR"
