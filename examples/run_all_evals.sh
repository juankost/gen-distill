#!/usr/bin/env bash
#
# Run one checkpoint over every evaluation suite the paper reports.
#
#   1. Generation-based short context  (EvalScope)
#   2. Log-likelihood short context    (LM Evaluation Harness)
#   3. LongBench                       (LM Evaluation Harness)
#   4. RULER NIAH-single, 4K-32K       (LM Evaluation Harness)
#   5. RULER NIAH-single, 64K-128K     (LM Evaluation Harness, YaRN)
#
# Edit CKPT_PATH below, or pass --checkpoint-path. Start with --smoke to check
# the plumbing in minutes before committing to the full run.
#
set -euo pipefail

############################################
# Edit me (or pass --checkpoint-path)
############################################
CKPT_PATH="/path/to/your/checkpoint"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

die() {
  echo -e "\nError: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $0 [--checkpoint-path PATH] [options]

Runs every evaluation suite reported in the paper against one checkpoint.

Optional arguments:
  --checkpoint-path PATH  Model checkpoint (local dir or HF Hub ID). Overrides
                          the CKPT_PATH value edited into this script.
  --smoke                 Tiny run to verify the plumbing: few samples per task,
                          RULER restricted to a single 4K length, no 64K/128K leg.
  --limit N               Samples per task. Implied by --smoke (default there: 2).
  --max-new-tokens N      Cap generated tokens per sample. Implied by --smoke (default
                          there: 4). A plumbing check asks whether every suite runs and
                          reports, not what it scores, and an untrained checkpoint never
                          emits EOS, so uncapped generation dominates the runtime.
  --bsz B                 Batch size (default: 4 normally, 1 under --smoke).
  --gpu-id ID             Value for CUDA_VISIBLE_DEVICES (default: environment default).
  --bf16                  Use bfloat16 instead of float32.
  --greedy                Greedy decoding instead of the paper's sampling config.
  --output-root DIR       Root for all results (default: ./eval_outputs).
  --skip-language         Skip the short-context suites (EvalScope + LM-Eval).
  --skip-longbench        Skip LongBench.
  --skip-ruler            Skip both RULER legs.
  --skip-ruler-extended   Skip only the 64K/128K RULER leg.
  --dry-run               Print the plan and the commands, run nothing.
  -h, --help              Show this help message and exit.

Example usage:
  bash examples/run_all_evals.sh --checkpoint-path ./checkpoints/hybrid-kda --smoke
  bash examples/run_all_evals.sh --checkpoint-path ./checkpoints/hybrid-kda --bf16 --greedy
EOF
}

############################################
# Arg parsing
############################################
SMOKE=false
LIMIT=""
MAX_NEW_TOKENS=""
BATCH_SIZE=""
GPU_ID=""
USE_BF16=false
USE_GREEDY=false
OUTPUT_ROOT="./eval_outputs"
RUN_LANGUAGE=true
RUN_LONGBENCH=true
RUN_RULER=true
RUN_RULER_EXTENDED=true
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-path)
      CKPT_PATH="${2:-}"; shift 2;;
    --smoke)
      SMOKE=true; shift 1;;
    --limit)
      LIMIT="${2:-}"; shift 2;;
    --max-new-tokens)
      MAX_NEW_TOKENS="${2:-}"; shift 2;;
    --bsz)
      BATCH_SIZE="${2:-}"; shift 2;;
    --gpu-id)
      GPU_ID="${2:-}"; shift 2;;
    --bf16)
      USE_BF16=true; shift 1;;
    --greedy)
      USE_GREEDY=true; shift 1;;
    --output-root)
      OUTPUT_ROOT="${2:-}"; shift 2;;
    --skip-language)
      RUN_LANGUAGE=false; shift 1;;
    --skip-longbench)
      RUN_LONGBENCH=false; shift 1;;
    --skip-ruler)
      RUN_RULER=false; shift 1;;
    --skip-ruler-extended)
      RUN_RULER_EXTENDED=false; shift 1;;
    --dry-run)
      DRY_RUN=true; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      die "Unknown argument: $1 (use --help)";;
  esac
done

if [ "${CKPT_PATH}" = "/path/to/your/checkpoint" ]; then
  die "No checkpoint set. Either edit CKPT_PATH at the top of this script or pass --checkpoint-path PATH."
fi

############################################
# Task lists, matching the paper
############################################

# Generation-based, EvalScope. Knowledge, common sense, reasoning, code and
# instruction following.
EVALSCOPE_TASKS="ceval,mmlu_redux,cmmlu,arc,hellaswag,winogrande,bbh,gsm8k,humaneval,ifeval"

# Log-likelihood ranking, LM Evaluation Harness. Note that MMLU (not MMLU-Redux)
# is used here, because MMLU-Redux only exists in a generative format in this
# harness, and that PIQA and LAMBADA appear only under this protocol.
LM_EVAL_TASKS="arc_easy,arc_challenge,hellaswag,winogrande,piqa,lambada_openai,mmlu,cmmlu"

LONGBENCH_TASKS="longbench_dureader,longbench_hotpotqa,longbench_musique,longbench_narrativeqa,longbench_qmsum,longbench_triviaqa"
NIAH_TASKS="niah_single_1,niah_single_2,niah_single_3"
NIAH_NATIVE_LENGTHS="4096,8192,16384,32768"
NIAH_EXTENDED_LENGTHS="65536,131072"
NIAH_EXTENDED_YARN="4.0"

############################################
# Smoke-mode overrides
############################################
if $SMOKE; then
  LIMIT="${LIMIT:-2}"
  BATCH_SIZE="${BATCH_SIZE:-1}"
  # Cap generation hard. An untrained or randomly initialised checkpoint never emits
  # EOS, so every sample otherwise decodes to the per-task maximum (2048 for IFEval),
  # which turns a plumbing check into a run of many hours.
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
  NIAH_NATIVE_LENGTHS="4096"
  RUN_RULER_EXTENDED=false
else
  BATCH_SIZE="${BATCH_SIZE:-4}"
fi

# Pin the GPU once here rather than per child script. eval_on_language.sh takes
# no --gpu-id flag, and every child process inherits this environment anyway.
if [ -n "${GPU_ID}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

# Flags understood by both eval_on_language.sh and eval_on_longbench.sh.
COMMON_FLAGS=""
if $USE_BF16; then
  COMMON_FLAGS="${COMMON_FLAGS} --bf16"
fi
if $USE_GREEDY; then
  COMMON_FLAGS="${COMMON_FLAGS} --greedy"
fi

# RULER is greedy by construction: do_sample false and max_gen_toks 128 are
# fixed in the task YAMLs, so eval_on_ruler.sh deliberately has no --greedy
# flag and must not be handed one.
RULER_FLAGS=""
if $USE_BF16; then
  RULER_FLAGS="${RULER_FLAGS} --bf16"
fi

LIMIT_FLAG=""
[ -n "${LIMIT}" ] && LIMIT_FLAG="--limit ${LIMIT}"

# One cap, two spellings: the EvalScope and LongBench scripts take
# --max-new-tokens, while RULER overrides its task YAMLs' max_gen_toks instead.
MAX_NEW_TOKENS_FLAG=""
RULER_MAX_GEN_FLAG=""
if [ -n "${MAX_NEW_TOKENS}" ]; then
  MAX_NEW_TOKENS_FLAG="--max-new-tokens ${MAX_NEW_TOKENS}"
  RULER_MAX_GEN_FLAG="--max-gen-toks ${MAX_NEW_TOKENS}"
fi

############################################
# Plan
############################################
echo ""
echo "================================================================"
echo " gen-distill: full evaluation sweep"
echo "================================================================"
echo " Checkpoint:   ${CKPT_PATH}"
echo " Output root:  ${OUTPUT_ROOT}"
echo " Batch size:   ${BATCH_SIZE}"
echo " Precision:    $($USE_BF16 && echo bfloat16 || echo float32)"
echo " Decoding:     $($USE_GREEDY && echo greedy || echo 'sampling (temp 0.7, top-p 0.8, top-k 20)')"
echo " Sample limit: ${LIMIT:-none (full benchmarks)}"
echo " Max new tok:  ${MAX_NEW_TOKENS:-per-task defaults}"
echo " Mode:         $($SMOKE && echo 'SMOKE, plumbing check only, numbers are meaningless' || echo 'FULL')"
echo ""
echo " About to run:"
$RUN_LANGUAGE  && echo "   [1] EvalScope, generation-based:  ${EVALSCOPE_TASKS}"
$RUN_LANGUAGE  && echo "   [2] LM-Eval, log-likelihood:      ${LM_EVAL_TASKS}"
$RUN_LONGBENCH && echo "   [3] LongBench:                    ${LONGBENCH_TASKS}"
$RUN_RULER     && echo "   [4] RULER NIAH ${NIAH_NATIVE_LENGTHS}: ${NIAH_TASKS}"
($RUN_RULER && $RUN_RULER_EXTENDED) && echo "   [5] RULER NIAH ${NIAH_EXTENDED_LENGTHS} (YaRN ${NIAH_EXTENDED_YARN}): ${NIAH_TASKS}"
echo ""
if ! $SMOKE; then
  cat <<'EOF'
 Cost warning. A full sweep is hours to days on a single GPU, not minutes:
   - EvalScope generation tasks decode up to 2048 new tokens per sample
     (IFEval), and HumanEval/GSM8K up to 1024.
   - RULER synthesises 500 samples per task per length. The native leg alone
     is 3 tasks x 4 lengths x 500 = 6000 long-context generations, and the
     32K/64K/128K samples are individually expensive.
   - LongBench documents run to roughly 22K tokens each.
 Run with --smoke first to confirm the plumbing end to end.
EOF
  echo ""
fi

run_step() {
  local label="$1"; shift
  echo ""
  echo "---------------- ${label} ----------------"
  echo "+ $*"
  if $DRY_RUN; then
    return 0
  fi
  "$@"
}

if $DRY_RUN; then
  echo " DRY RUN: commands are printed but not executed."
fi

############################################
# Run
############################################

if $RUN_LANGUAGE; then
  # shellcheck disable=SC2086
  run_step "Short context: EvalScope + LM-Eval" \
    bash "${REPO_ROOT}/scripts/eval_on_language.sh" \
      --checkpoint-path "${CKPT_PATH}" \
      --evalscope-tasks "${EVALSCOPE_TASKS}" \
      --lm-eval-tasks "${LM_EVAL_TASKS}" \
      --bsz "${BATCH_SIZE}" \
      --evalscope-output "${OUTPUT_ROOT}/evalscope" \
      --lm-eval-output "${OUTPUT_ROOT}/lm-eval" \
      ${COMMON_FLAGS} ${LIMIT_FLAG} ${MAX_NEW_TOKENS_FLAG}
fi

if $RUN_LONGBENCH; then
  # shellcheck disable=SC2086
  run_step "Long context: LongBench" \
    bash "${REPO_ROOT}/scripts/eval_on_longbench.sh" \
      --checkpoint-path "${CKPT_PATH}" \
      --tasks "${LONGBENCH_TASKS}" \
      --bsz "${BATCH_SIZE}" \
      --output "${OUTPUT_ROOT}/longbench" \
      ${COMMON_FLAGS} ${LIMIT_FLAG} ${MAX_NEW_TOKENS_FLAG}
fi

if $RUN_RULER; then
  # shellcheck disable=SC2086
  run_step "Long context: RULER NIAH-single (${NIAH_NATIVE_LENGTHS})" \
    bash "${REPO_ROOT}/scripts/eval_on_ruler.sh" \
      --checkpoint-path "${CKPT_PATH}" \
      --tasks "${NIAH_TASKS}" \
      --seq-lengths "${NIAH_NATIVE_LENGTHS}" \
      --bsz "${BATCH_SIZE}" \
      --output "${OUTPUT_ROOT}/ruler" \
      ${RULER_FLAGS} ${LIMIT_FLAG} ${RULER_MAX_GEN_FLAG}

  if $RUN_RULER_EXTENDED; then
    # The extended leg deliberately takes no --limit: it always spans two
    # lengths, and eval_on_ruler.sh rejects --limit in that case.
    # shellcheck disable=SC2086
    run_step "Long context: RULER NIAH-single (${NIAH_EXTENDED_LENGTHS}, YaRN ${NIAH_EXTENDED_YARN})" \
      bash "${REPO_ROOT}/scripts/eval_on_ruler.sh" \
        --checkpoint-path "${CKPT_PATH}" \
        --tasks "${NIAH_TASKS}" \
        --seq-lengths "${NIAH_EXTENDED_LENGTHS}" \
        --rope-scaling-factor "${NIAH_EXTENDED_YARN}" \
        --bsz "${BATCH_SIZE}" \
        --output "${OUTPUT_ROOT}/ruler_extended" \
        ${RULER_FLAGS} ${RULER_MAX_GEN_FLAG}
  fi
fi

echo ""
echo "================================================================"
if $DRY_RUN; then
  echo " Dry run finished. Nothing was executed."
else
  echo " All requested evaluation suites finished."
  echo " Results under: ${OUTPUT_ROOT}"
fi
echo "================================================================"
