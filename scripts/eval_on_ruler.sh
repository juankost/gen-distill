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

Evaluate a checkpoint on the RULER needle-in-a-haystack tasks through the LM
Evaluation Harness. The defaults are the three NIAH-single subtasks the paper
reports: Single-1 (numeric needle in a repeated sentence), Single-2 (numeric
needle in natural prose) and Single-3 (UUID needle in natural prose).

RULER builds its haystacks on the fly with the model's own tokenizer, packing
text to each requested sequence length, so no dataset download is needed. It
generates 500 samples per task per sequence length.

Contexts beyond the model's native max_position_embeddings need YaRN RoPE
scaling: the paper uses --rope-scaling-factor 4.0 for the 64K and 128K columns
and no scaling for 4K-32K.

Required arguments:
  --checkpoint-path PATH             Path to model checkpoint (local dir or HuggingFace Hub ID).

Optional arguments:
  --seq-lengths "L1,L2,..."          Comma-separated context lengths (default: 4096,8192,16384,32768).
  --rope-scaling-factor F            YaRN RoPE scaling factor, e.g. 4.0, to reach 64K/128K (default: none).
  --rope-scaling-original-max-pos N  Original max position embeddings for YaRN (default: the model config value).
  --tasks TASKS                      Comma-separated RULER tasks (default: niah_single_1,niah_single_2,niah_single_3).
  --n-gpus N                         Number of GPUs to use (default: 1).
  --bsz B                            Batch size for evaluation (default: 1).
  --limit N                          Limit samples per task. Only allowed with a single --seq-lengths value.
  --bf16                             Use bfloat16 precision instead of float32 for accelerate.
  --max-gen-toks N                   Override the task YAMLs' max_gen_toks (default: 128). For
                                     plumbing checks only: a needle cannot be found in fewer
                                     tokens than it occupies, so scores become meaningless.
  --seed N                           Random seed (default: 1).
  --gpu-id ID                        Value for CUDA_VISIBLE_DEVICES (default: leave the environment alone).
  --output DIR                       Output dir for results (default: ./eval_outputs/ruler).
  -h, --help                         Show this help message and exit.

Example usage:
  # Native context (4K-32K), no YaRN needed
  bash scripts/eval_on_ruler.sh --checkpoint-path ./checkpoints/hybrid-kda --seq-lengths "4096,8192,16384,32768"

  # Extended context (64K-128K) with YaRN
  bash scripts/eval_on_ruler.sh --checkpoint-path ./checkpoints/hybrid-kda --seq-lengths "65536,131072" --rope-scaling-factor 4.0

  # Fast plumbing check: one length, a handful of samples
  bash scripts/eval_on_ruler.sh --checkpoint-path ./checkpoints/hybrid-kda --seq-lengths "4096" --limit 4
EOF
}

############################################
# Arg parsing
############################################
CKPT_PATH=""
SEQ_LENGTHS="4096,8192,16384,32768"
ROPE_SCALING_FACTOR=""
ROPE_SCALING_ORIGINAL_MAX_POS=""
NUM_GPUS=1
BATCH_SIZE=1
LIMIT_SAMPLES=null
USE_BF16=false
MAX_GEN_TOKS=""
SEED=1
GPU_ID=""
OUTPUT_DIR="./eval_outputs/ruler"

# NOTE: the subtasks are listed individually rather than via the "ruler" group
# name on purpose. lm-eval strips --metadata (which carries max_seq_lengths)
# from subtasks loaded through a group, because "metadata" is one of
# GROUP_ONLY_KEYS (lm_eval/tasks/__init__.py). Through the group name every
# task silently falls back to DEFAULT_SEQ_LENGTHS and the requested lengths are
# ignored.
TASKS="niah_single_1,niah_single_2,niah_single_3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-path)
      CKPT_PATH="${2:-}"; shift 2;;
    --max-gen-toks)
      MAX_GEN_TOKS="${2:-}"; shift 2;;
    --seq-lengths)
      SEQ_LENGTHS="${2:-}"; shift 2;;
    --rope-scaling-factor)
      ROPE_SCALING_FACTOR="${2:-}"; shift 2;;
    --rope-scaling-original-max-pos)
      ROPE_SCALING_ORIGINAL_MAX_POS="${2:-}"; shift 2;;
    --tasks)
      TASKS="${2:-}"; shift 2;;
    --n-gpus)
      NUM_GPUS="${2:-}"; shift 2;;
    --bsz)
      BATCH_SIZE="${2:-}"; shift 2;;
    --limit)
      LIMIT_SAMPLES="${2:-}"; shift 2;;
    --bf16)
      USE_BF16=true; shift 1;;
    --seed)
      SEED="${2:-}"; shift 2;;
    --gpu-id)
      GPU_ID="${2:-}"; shift 2;;
    --output)
      OUTPUT_DIR="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      die "Unknown argument: $1 (use --help)";;
  esac
done

[[ -n "${CKPT_PATH}" ]] || die "Missing required argument: --checkpoint-path (use --help)"

# Compute max_length from the largest requested length, plus headroom for the
# generated answer.
IFS=',' read -ra SEQ_LENGTH_ARR <<< "${SEQ_LENGTHS}"
[[ ${#SEQ_LENGTH_ARR[@]} -gt 0 ]] || die "--seq-lengths must not be empty"
MAX_SEQ_LENGTH=0
for len in "${SEQ_LENGTH_ARR[@]}"; do
  if (( len > MAX_SEQ_LENGTH )); then
    MAX_SEQ_LENGTH=$len
  fi
done
MAX_LENGTH=$(( MAX_SEQ_LENGTH + 256 ))

SEQ_LENGTHS_JSON="[${SEQ_LENGTHS}]"

# RULER emits its samples grouped by length: every 4K sample first, then every
# 8K sample, and so on. lm-eval's --limit truncates that concatenated list, so
# with more than one length it would only ever evaluate the shortest one. Allow
# it for the single-length case, which is the useful smoke-test shape.
LIMIT_SAMPLES_FLAG=""
if [ "${LIMIT_SAMPLES}" != "null" ]; then
  if [ ${#SEQ_LENGTH_ARR[@]} -gt 1 ]; then
    die "--limit is only supported with a single --seq-lengths value (RULER orders samples by length, so a limit would silently evaluate only the shortest). Requested lengths: ${SEQ_LENGTHS}"
  fi
  LIMIT_SAMPLES_FLAG="--limit ${LIMIT_SAMPLES}"
fi

if $USE_BF16; then
  MIXED_PRECISION="bfloat16"
else
  MIXED_PRECISION="float32"
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

MODEL_ARGS="pretrained=${CKPT_PATH},attn_implementation=sdpa,dtype=auto,trust_remote_code=True,device=cuda:0,enable_thinking=False,max_length=${MAX_LENGTH}"
if [ -n "${ROPE_SCALING_FACTOR}" ]; then
  MODEL_ARGS="${MODEL_ARGS},rope_scaling_factor=${ROPE_SCALING_FACTOR}"
  if [ -n "${ROPE_SCALING_ORIGINAL_MAX_POS}" ]; then
    MODEL_ARGS="${MODEL_ARGS},rope_scaling_original_max_pos=${ROPE_SCALING_ORIGINAL_MAX_POS}"
  fi
fi

mkdir -p "${OUTPUT_DIR}"

############################################
# Run the evals
############################################

echo ""
echo "================================================================"
echo " RULER needle-in-a-haystack evaluation"
echo " Checkpoint:       ${CKPT_PATH}"
echo " Tasks:            ${TASKS}"
echo " Sequence lengths: ${SEQ_LENGTHS}"
echo " Model max_length: ${MAX_LENGTH} (longest length + 256)"
echo " YaRN factor:      ${ROPE_SCALING_FACTOR:-none}"
echo " Batch size:       ${BATCH_SIZE}   GPUs: ${NUM_GPUS}   Seed: ${SEED}"
echo " Output:           ${OUTPUT_DIR}"
echo "================================================================"
echo ""

# RULER tasks use doc_to_text "{{input}}", i.e. raw text completion rather than
# a chat exchange, so --apply_chat_template is deliberately NOT passed here.
# Decoding is greedy with max_gen_toks=128, fixed in the task YAMLs. --max-gen-toks
# overrides that through gen_kwargs, the only generation-length knob lm-eval honours
# here.
GEN_KWARGS_FLAG=""
if [ -n "${MAX_GEN_TOKS}" ]; then
  GEN_KWARGS_FLAG="--gen_kwargs '{\"max_gen_toks\":${MAX_GEN_TOKS}}'"
fi
EVAL_CMD="${EVAL_CMD_HEAD} \
  --model efficient_qwen \
  --model_args ${MODEL_ARGS} \
  ${GEN_KWARGS_FLAG} \
  --tasks ${TASKS} \
  --metadata '{\"max_seq_lengths\":${SEQ_LENGTHS_JSON}}' \
  --seed '${SEED},${SEED},${SEED},${SEED}' \
  --trust_remote_code \
  --confirm_run_unsafe_code \
  --show_config \
  --batch_size ${BATCH_SIZE} \
  --output_path ${OUTPUT_DIR} ${LIMIT_SAMPLES_FLAG}"

echo "Command: ${EVAL_CMD}"
eval "${EVAL_CMD}"

echo ""
echo "RULER evaluation complete. Results at: ${OUTPUT_DIR}"
