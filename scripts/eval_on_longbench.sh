#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=true
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_ALLOW_CODE_EVAL=1

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

Evaluate a checkpoint on LongBench (real-world long-context QA and summarization,
roughly 8K-22K token contexts) through the LM Evaluation Harness.

The six default tasks are the ones reported in the paper: DuReader, HotpotQA,
MuSiQue, NarrativeQA, QMSum and TriviaQA. Each task is run as its own lm_eval
invocation so that one failing task does not discard the others' results.

Required arguments:
  --checkpoint-path PATH   Path to model checkpoint (local dir or HuggingFace Hub ID).

Optional arguments:
  --tasks TASKS            Comma-separated LongBench tasks (default: the 6 paper tasks).
  --all-tasks              Run the full 21-task LongBench suite instead of the 6 paper tasks.
  --n-gpus N               Number of GPUs to use (default: 1).
  --bsz B                  Batch size for evaluation (default: 1).
  --limit N                Limit number of documents per task (default: all).
  --bf16                   Use bfloat16 precision instead of float32 for accelerate.
  --thinking_mode          Enable thinking mode in the chat template.
  --greedy                 Use greedy decoding instead of the sampling config.
  --max-new-tokens N       Cap generated tokens per document (default: 300).
  --max-length N           Model max context length (default: 327680, i.e. no truncation).
  --seed N                 Random seed (default: 1).
  --gpu-id ID              Value for CUDA_VISIBLE_DEVICES (default: leave the environment alone).
  --no-apply-chat-template Disable --apply_chat_template (enabled by default).
  --output DIR             Output dir for results (default: ./eval_outputs/longbench).
  -h, --help               Show this help message and exit.

Example usage:
  bash scripts/eval_on_longbench.sh --checkpoint-path ./checkpoints/hybrid-kda --limit 2 --bsz 1
  bash scripts/eval_on_longbench.sh --checkpoint-path <hf-repo-id-or-local-path> --bf16 --greedy
EOF
}

############################################
# Arg parsing
############################################
CKPT_PATH=""
NUM_GPUS=1
BATCH_SIZE=1
LIMIT_SAMPLES=null
USE_BF16=false
USE_THINKING_MODE=false
USE_GREEDY=false
MAX_NEW_TOKENS=""
MAX_LENGTH=327680
SEED=1
GPU_ID=""
APPLY_CHAT_TEMPLATE=true
OUTPUT_DIR="./eval_outputs/longbench"
EVAL_ALL=false
OVERRIDE_TASKS=""

# The six LongBench tasks reported in the paper.
PAPER_TASKS="longbench_dureader,longbench_hotpotqa,longbench_musique,longbench_narrativeqa,longbench_qmsum,longbench_triviaqa"
ALL_TASKS="longbench_dureader,longbench_hotpotqa,longbench_musique,longbench_narrativeqa,longbench_qmsum,longbench_triviaqa,longbench_samsum,longbench_passage_count,longbench_passage_retrieval_zh,longbench_2wikimqa,longbench_vcsum,longbench_lcc,longbench_multifieldqa_en,longbench_qasper,longbench_passage_retrieval_en,longbench_trec,longbench_gov_report,longbench_lsht,longbench_multi_news,longbench_multifieldqa_zh,longbench_repobench-p"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-path)
      CKPT_PATH="${2:-}"; shift 2;;
    --max-new-tokens)
      MAX_NEW_TOKENS="${2:-}"; shift 2;;
    --tasks)
      OVERRIDE_TASKS="${2:-}"; shift 2;;
    --all-tasks)
      EVAL_ALL=true; shift 1;;
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
    --max-length)
      MAX_LENGTH="${2:-}"; shift 2;;
    --seed)
      SEED="${2:-}"; shift 2;;
    --gpu-id)
      GPU_ID="${2:-}"; shift 2;;
    --no-apply-chat-template)
      APPLY_CHAT_TEMPLATE=false; shift 1;;
    --output)
      OUTPUT_DIR="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      die "Unknown argument: $1 (use --help)";;
  esac
done

[[ -n "${CKPT_PATH}" ]] || die "Missing required argument: --checkpoint-path (use --help)"

if [ -n "${OVERRIDE_TASKS}" ]; then
  TASKS="${OVERRIDE_TASKS}"
elif $EVAL_ALL; then
  TASKS="${ALL_TASKS}"
else
  TASKS="${PAPER_TASKS}"
fi

if $USE_BF16; then
  MIXED_PRECISION="bfloat16"
else
  MIXED_PRECISION="float32"
fi

LONGBENCH_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-300}"
if $USE_GREEDY; then
  GEN_KWARGS="{\"max_new_tokens\":${LONGBENCH_MAX_NEW_TOKENS},\"do_sample\":false}"
else
  GEN_KWARGS="{\"temperature\":0.7,\"top_p\":0.8,\"top_k\":20,\"min_p\":0.0,\"max_new_tokens\":${LONGBENCH_MAX_NEW_TOKENS},\"do_sample\":true}"
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

if [ -n "${GPU_ID}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
  EVAL_CMD_HEAD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --mixed_precision=${MIXED_PRECISION} -m gen_distill.evals.lm_harness_eval"
elif [ "${NUM_GPUS}" -eq 1 ]; then
  EVAL_CMD_HEAD="python -m gen_distill.evals.lm_harness_eval"
else
  die "Invalid number of GPUs: ${NUM_GPUS}"
fi

mkdir -p "${OUTPUT_DIR}"

############################################
# Run the evals
############################################

echo ""
echo "================================================================"
echo " LongBench evaluation"
echo " Checkpoint:  ${CKPT_PATH}"
echo " Tasks:       ${TASKS}"
echo " Batch size:  ${BATCH_SIZE}   GPUs: ${NUM_GPUS}   Seed: ${SEED}"
echo " Max length:  ${MAX_LENGTH}"
echo " Output:      ${OUTPUT_DIR}"
echo "================================================================"
echo ""

# One lm_eval invocation per task: LongBench documents are long, and running the
# tasks separately keeps a single failure from discarding the whole sweep.
for task in ${TASKS//,/ }; do
  echo "Running ${task}..."
  EVAL_CMD="${EVAL_CMD_HEAD} \
    --model efficient_qwen \
    --model_args pretrained=${CKPT_PATH},attn_implementation=sdpa,dtype=auto,trust_remote_code=True,device=cuda:0,enable_thinking=False,max_length=${MAX_LENGTH} \
    --gen_kwargs '${GEN_KWARGS}' \
    --tasks '${task}' \
    --seed '${SEED},${SEED},${SEED},${SEED}' \
    --trust_remote_code \
    ${APPLY_CHAT_TEMPLATE_FLAG} \
    --show_config \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_DIR} ${LIMIT_SAMPLES_FLAG}"
  echo "Command: ${EVAL_CMD}"
  eval "${EVAL_CMD}"
done

echo ""
echo "LongBench evaluation complete. Results at: ${OUTPUT_DIR}"
