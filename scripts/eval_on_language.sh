#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=true
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

############################################
# Helper functions
############################################

die() {
  echo -e "\nError: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $0 --checkpoint-path PATH [options]

Required arguments:
  --checkpoint-path PATH   Path to model checkpoint (local or HuggingFace Hub ID).

Optional arguments:
  --n-gpus N             Number of GPUs to use (default: 1).
  --bsz B               Batch size for evaluation (default: 4).
  --limit N              Limit number of samples to evaluate (default: null).
  --bf16                 Use bfloat16 precision instead of float32.
  --thinking_mode        Enable thinking mode for evaluation.
  --greedy               Enable greedy decoding for evaluation.
  --max-new-tokens N     Cap generated tokens for BOTH legs: overrides EvalScope's
                         per-task table, and replaces LM-Eval's gen_kwargs default of 300.
  --seed N               Random seed for both harnesses (default: 1).
  --evalscope-tasks TASKS  Comma-separated list of EvalScope tasks (default: ceval,arc,hellaswag,winogrande,mmlu_redux).
  --lm-eval-tasks TASKS   Comma-separated list of LM-Eval tasks (default: piqa).
  --skip-lm-eval         Skip LM-Evaluation harness evals.
  --skip-evalscope       Skip EvalScope evals.
  --no-apply-chat-template  Disable --apply_chat_template in lm_eval (default: enabled).
  --evalscope-output DIR   Output dir for evalscope results (default: ./eval_outputs/evalscope).
  --lm-eval-output DIR     Output dir for lm-eval results (default: ./eval_outputs/lm-eval).
  -h, --help             Show this help message and exit.

Example usage:
  bash scripts/eval_on_language.sh --checkpoint-path ./checkpoints/hybrid-kda --limit 2 --bsz 1
  bash scripts/eval_on_language.sh --checkpoint-path <hf-repo-id-or-local-path> --bf16 --greedy
EOF
}

############################################
# Arg parsing
############################################
NUM_GPUS=1
BATCH_SIZE=4
USE_BF16=false
USE_THINKING_MODE=false
USE_GREEDY=false
MAX_NEW_TOKENS=""
LIMIT_SAMPLES=null
RUN_EVALSCOPE=true
RUN_LM_EVAL=true
EVALSCOPE_TASKS="ceval,arc,hellaswag,winogrande,mmlu_redux"
LM_EVAL_TASKS="piqa"
SEED=1
APPLY_CHAT_TEMPLATE=true
EVALSCOPE_OUTPUT_DIR="./eval_outputs/evalscope"
LM_EVAL_OUTPUT_DIR="./eval_outputs/lm-eval"
CKPT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-path)
      CKPT_PATH="${2:-}"; shift 2;;
    --n-gpus)
      NUM_GPUS="${2:-}"; shift 2;;
    --bsz)
      BATCH_SIZE="${2:-}"; shift 2;;
    --limit)
      LIMIT_SAMPLES="${2:-}"; shift 2;;
    --bf16)
      USE_BF16=true; shift 1;;
    --thinking_mode)
      USE_THINKING_MODE=true; shift 1;;
    --greedy)
      USE_GREEDY=true; shift 1;;
    --max-new-tokens)
      MAX_NEW_TOKENS="${2:-}"; shift 2;;
    --evalscope-tasks)
      EVALSCOPE_TASKS="${2:-}"; shift 2;;
    --lm-eval-tasks)
      LM_EVAL_TASKS="${2:-}"; shift 2;;
    --skip-lm-eval)
      RUN_LM_EVAL=false; shift 1;;
    --skip-evalscope)
      RUN_EVALSCOPE=false; shift 1;;
    --no-apply-chat-template)
      APPLY_CHAT_TEMPLATE=false; shift 1;;
    --evalscope-output)
      EVALSCOPE_OUTPUT_DIR="${2:-}"; shift 2;;
    --lm-eval-output)
      LM_EVAL_OUTPUT_DIR="${2:-}"; shift 2;;
    --seed)
      SEED="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      die "Unknown argument: $1 (use --help)";;
  esac
done

[[ -n "${CKPT_PATH}" ]] || die "Missing required argument: --checkpoint-path (use --help)"

if $USE_BF16; then
  MIXED_PRECISION="bfloat16"
else
  MIXED_PRECISION="float32"
fi

if $USE_THINKING_MODE; then
  THINKING_MODE_FLAG="--thinking_mode"
else
  THINKING_MODE_FLAG=""
fi

if $USE_GREEDY; then
  GREEDY_FLAG="--greedy"
else
  GREEDY_FLAG=""
fi

if [ "${LIMIT_SAMPLES}" != "null" ]; then
  LIMIT_SAMPLES_FLAG="--limit ${LIMIT_SAMPLES}"
else
  LIMIT_SAMPLES_FLAG=""
fi

if $APPLY_CHAT_TEMPLATE; then
  APPLY_CHAT_TEMPLATE_FLAG="--apply_chat_template"
else
  APPLY_CHAT_TEMPLATE_FLAG=""
fi

if [ ${NUM_GPUS} -gt 1 ]; then
  EVAL_CMD_HEAD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --mixed_precision=${MIXED_PRECISION} -m gen_distill.evals.lm_harness_eval"
elif [ ${NUM_GPUS} -eq 1 ]; then
  EVAL_CMD_HEAD="python -m gen_distill.evals.lm_harness_eval"
else
  die "Invalid number of GPUs: ${NUM_GPUS}"
fi

mkdir -p "${EVALSCOPE_OUTPUT_DIR}"
mkdir -p "${LM_EVAL_OUTPUT_DIR}"

############################################
# Run the evals
############################################

# EvalScope evals
if $RUN_EVALSCOPE; then
  MAX_NEW_TOKENS_FLAG=""
  if [ -n "${MAX_NEW_TOKENS}" ]; then
    MAX_NEW_TOKENS_FLAG="--max_new_tokens ${MAX_NEW_TOKENS}"
  fi
  EVAL_CMD="python -m gen_distill.evals.evalscope_evals \
      --test_datasets ${EVALSCOPE_TASKS} \
      --batch-size ${BATCH_SIZE} \
      --precision ${MIXED_PRECISION} \
      --model-name-or-path ${CKPT_PATH} \
      --seed ${SEED} \
      --output_path ${EVALSCOPE_OUTPUT_DIR} \
      ${THINKING_MODE_FLAG} \
      ${GREEDY_FLAG} \
      ${MAX_NEW_TOKENS_FLAG} \
      ${LIMIT_SAMPLES_FLAG}"
  echo "Running evalscope evals: Command: ${EVAL_CMD}"
  eval "${EVAL_CMD}"
fi

# LM-Evaluation harness evals
if $RUN_LM_EVAL; then
  # EvalScope takes --max-new-tokens as a flag (above); lm-eval only accepts a
  # generation length inside gen_kwargs, so the same value is spliced in here.
  LM_EVAL_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-300}"
  if $USE_GREEDY; then
    GEN_KWARGS="{\"max_new_tokens\":${LM_EVAL_MAX_NEW_TOKENS},\"do_sample\":false}"
  else
    GEN_KWARGS="{\"temperature\":0.7,\"top_p\":0.8,\"top_k\":20,\"min_p\":0.0,\"max_new_tokens\":${LM_EVAL_MAX_NEW_TOKENS},\"do_sample\":true}"
  fi
  ATTN_IMPLEMENTATION=sdpa
  # No --cache_requests: lm-eval 0.4.9.2 crashes on a fresh install when asked to
  # delete a request cache that was never created. delete_cache() calls
  # os.listdir(PATH) with no existence check (lm_eval/caching/cache.py:54), while
  # only the save path creates the directory, so the very first run dies with
  # FileNotFoundError before evaluating anything. Omitting the flag disables
  # request caching entirely, which is the state the flag was reaching for anyway.
  EVAL_CMD="${EVAL_CMD_HEAD} \
    --model efficient_qwen \
    --model_args pretrained=${CKPT_PATH},attn_implementation=${ATTN_IMPLEMENTATION},dtype="auto",trust_remote_code=True,device=cuda:0,enable_thinking=False,max_length=327680 \
    --gen_kwargs '${GEN_KWARGS}' \
    --tasks ${LM_EVAL_TASKS} \
    --seed '${SEED},${SEED},${SEED},${SEED}' \
    --trust_remote_code \
    ${APPLY_CHAT_TEMPLATE_FLAG} \
    --confirm_run_unsafe_code \
    --show_config \
    --batch_size ${BATCH_SIZE} \
    --output_path ${LM_EVAL_OUTPUT_DIR} ${LIMIT_SAMPLES_FLAG}"
  echo "Running LM-Evaluation harness evals: Command: ${EVAL_CMD}"
  eval "${EVAL_CMD}"
fi
