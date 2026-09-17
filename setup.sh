#!/usr/bin/env bash
#
# setup.sh - build the inference + evaluation environment for gen-distill.
#
# This installs everything needed to load a released checkpoint and run the two
# evaluation harnesses. It does NOT install training-only packages (deepspeed,
# trl, vllm): none of them is imported by this repository.
#
# One script covers every supported GPU. The pinned torch build (cu128) carries
# SASS for sm_75/80/86/90/100/120, and the pinned mamba-ssm / causal-conv1d
# release wheels carry SASS for sm_53..sm_120, so one path serves them all.
#
# Run `bash setup.sh --help` for the flag list.

set -euo pipefail

# ---------------------------------------------------------------------------
# Pinned versions. Every pin here was resolved from a working environment; see
# README.md for the verified combination.
# ---------------------------------------------------------------------------
TORCH_VERSION="2.7.0"
CUDA_TAG="cu128"
TRITON_VERSION="3.4.0"
TRANSFORMERS_VERSION="4.54.1"
DATASETS_VERSION="3.6.0"
LM_EVAL_VERSION="0.4.9.2"
MAMBA_VERSION="2.3.0"
CAUSAL_CONV1D_VERSION="1.6.0"

# flash-linear-attention is installed from git, NOT from PyPI: the published
# PyPI artifacts for 0.3.x and newer ship fla/layers and fla/models but no
# fla/ops, so `from fla.ops.kda import chunk_kda` - which this package does at
# import time - fails against a PyPI install. This commit is the one the
# released checkpoints were evaluated with (reports itself as version 0.5.1).
FLA_REPO="https://github.com/fla-org/flash-linear-attention"
FLA_REF="e91af3c61de0ed8fd5604c9a1a40cd3c56ef7911"

# The conda environment this script creates. Fixed on purpose: the repository
# has exactly one environment and nothing here varies per user.
ENV_NAME="gen_distill"

# Pinned, not overridable. This script installs the one environment the released
# checkpoints were evaluated with; every version above is part of that set.
PYTHON_VERSION="3.12"

# Operational flags. Neither changes what gets installed.
RECREATE=0
DRY_RUN=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() {
    echo "" >&2
    echo "ERROR: $*" >&2
    exit 1
}

log() {
    echo ""
    echo "=== $* ==="
}

# Echo then run. Every install step goes through this, so a failure is visible
# and - because of `set -e` - stops the script instead of being carried past.
run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        return 0
    fi
    "$@"
}

usage() {
    cat <<EOF
Usage: bash setup.sh [options]

Creates the conda environment '${ENV_NAME}' with everything needed to run and
evaluate the released gen-distill checkpoints. Safe to re-run: an existing
environment is reused and the pinned packages are re-asserted.

Every version is pinned in the script and there is no flag to change one: this
builds the single environment the released checkpoints were evaluated with.

Options:
  --recreate          Delete and recreate the environment first.
  --dry-run           Print every command without running any of it.
  -h, --help          Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recreate)       RECREATE=1;               shift 1 ;;
        --dry-run)        DRY_RUN=1;                shift 1 ;;
        -h|--help)        usage; exit 0 ;;
        *)                die "Unknown argument: $1 (use --help)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolved plan. Printed before anything is touched, so --dry-run can be judged
# on what it resolved rather than on its exit code.
# ---------------------------------------------------------------------------
TORCH_MINOR="$(echo "${TORCH_VERSION}" | cut -d. -f1,2)"
CPYTHON_TAG="cp$(echo "${PYTHON_VERSION}" | tr -d '.')"
MACHINE="$(uname -m)"

log "gen-distill environment setup"
cat <<EOF
Repository:     ${REPO_ROOT}
Environment:    conda env '${ENV_NAME}'
Python:         ${PYTHON_VERSION} (${CPYTHON_TAG})
PyTorch:        ${TORCH_VERSION} from the ${CUDA_TAG} index
Triton:         ${TRITON_VERSION} (forced; see note below)
transformers:   ${TRANSFORMERS_VERSION}
mamba-ssm:      ${MAMBA_VERSION} (prebuilt wheel)
causal-conv1d:  ${CAUSAL_CONV1D_VERSION} (prebuilt wheel)
FLA:            ${FLA_REPO}@${FLA_REF:0:12}
Eval stack:     lm_eval ${LM_EVAL_VERSION}, evalscope
Machine:        ${MACHINE}
Dry run:        $([[ "${DRY_RUN}" -eq 1 ]] && echo "YES - nothing will be installed" || echo "no")
EOF

if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "Detected GPUs:"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | sed 's/^/  /'
else
    echo ""
    echo "WARNING: nvidia-smi not found. Installing anyway; both sequence mixers"
    echo "         need a CUDA GPU at run time, there is no CPU path."
fi

# ---------------------------------------------------------------------------
# Environment creation. Idempotent: an existing environment is reused unless
# --recreate was passed.
# ---------------------------------------------------------------------------
command -v conda >/dev/null 2>&1 \
    || die "conda not found. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) and re-run."

# Non-interactive Anaconda ToS acceptance. Not an install step; a failure
# here is harmless on conda builds that have no `tos` subcommand.
if [[ "${DRY_RUN}" -eq 0 ]]; then
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
fi

ENV_EXISTS=0
if [[ "${DRY_RUN}" -eq 0 ]] && conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    ENV_EXISTS=1
fi

if [[ "${ENV_EXISTS}" -eq 1 && "${RECREATE}" -eq 1 ]]; then
    log "Removing existing conda environment '${ENV_NAME}'"
    run conda env remove -n "${ENV_NAME}" -y
    ENV_EXISTS=0
fi

if [[ "${ENV_EXISTS}" -eq 1 ]]; then
    log "Reusing existing conda environment '${ENV_NAME}'"
else
    log "Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})"
    # `pip` must be requested explicitly: a bare `conda create python=X` env has no
    # pip module, and every install step below runs through `python -m pip`.
    run conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
fi

CONDA_BASE="$(conda info --base)"
PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    [[ -x "${PY}" ]] || die "Python interpreter not found at ${PY}"
fi

PIP=("${PY}" -m pip)

log "Upgrading pip tooling"
# ninja stays even though nothing here compiles: mamba-ssm, causal-conv1d and
# scipy all declare it as a requirement, and the kernel wheels are installed
# with --no-deps, so pip never pulls it in on their behalf. Without it the
# finished environment fails `pip check` for a reason unrelated to the one
# deliberate deviation (triton).
run "${PIP[@]}" install --upgrade pip setuptools wheel packaging ninja

# ---------------------------------------------------------------------------
# PyTorch. Installed first and on its own: the kernel wheels below are built
# against a specific torch minor and ABI, and are read back off the installed
# torch rather than guessed.
# ---------------------------------------------------------------------------
log "Installing PyTorch ${TORCH_VERSION} (${CUDA_TAG})"
run "${PIP[@]}" install "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${PY}" -c "import torch" \
        || die "torch failed to import after installation"
    CXX11_ABI="$("${PY}" -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")"
    echo "torch $("${PY}" -c 'import torch; print(torch.__version__)') / CXX11 ABI ${CXX11_ABI}"
else
    CXX11_ABI="TRUE"
fi

# ---------------------------------------------------------------------------
# mamba-ssm and causal-conv1d, from the upstream GitHub release wheels.
#
# Neither publishes a binary wheel on PyPI - PyPI has sdists only - so a plain
# `pip install mamba-ssm` always compiles, needs nvcc, and needs torch already
# importable. The upstream GitHub releases do publish binary wheels covering
# sm_53..sm_120, which makes a 20-45 minute source build unnecessary.
#
# causal-conv1d is NOT optional in practice. The short-convolution layer
# defaults to backend='cuda'; without the package it switches itself to
# backend='triton', which dispatches to fla.ops.triton.causal_conv1d - a module
# current FLA no longer ships - and raises TypeError at the first forward.
# ---------------------------------------------------------------------------
MAMBA_WHEEL="https://github.com/state-spaces/mamba/releases/download/v${MAMBA_VERSION}/mamba_ssm-${MAMBA_VERSION}%2Bcu12torch${TORCH_MINOR}cxx11abi${CXX11_ABI}-${CPYTHON_TAG}-${CPYTHON_TAG}-linux_${MACHINE}.whl"
CONV_WHEEL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_VERSION}/causal_conv1d-${CAUSAL_CONV1D_VERSION}%2Bcu12torch${TORCH_MINOR}cxx11abi${CXX11_ABI}-${CPYTHON_TAG}-${CPYTHON_TAG}-linux_${MACHINE}.whl"

check_wheel_exists() {
    local url="$1" name="$2"
    command -v curl >/dev/null 2>&1 || return 0
    if ! curl -fsSLI -o /dev/null "${url}"; then
        die "No prebuilt ${name} wheel for python ${PYTHON_VERSION} / torch ${TORCH_MINOR} / ${MACHINE}.
       Expected: ${url}
       This is the pinned combination the released checkpoints were evaluated
       with, so an absent wheel means the upstream release assets moved."
    fi
}

log "Installing prebuilt causal-conv1d ${CAUSAL_CONV1D_VERSION} and mamba-ssm ${MAMBA_VERSION}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    check_wheel_exists "${CONV_WHEEL}"  "causal-conv1d"
    check_wheel_exists "${MAMBA_WHEEL}" "mamba-ssm"
fi
# --no-deps: these declare a bare `torch` requirement, and we do not want the
# resolver to reconsider the +cu128 build we just pinned.
run "${PIP[@]}" install --no-deps "${CONV_WHEEL}"
run "${PIP[@]}" install --no-deps "${MAMBA_WHEEL}"

# ---------------------------------------------------------------------------
# Core runtime imports. transformers is pinned: the model uses a private
# transformers cache API whose signature changed after 4.54.1, and 4.54.1 is the
# version the released checkpoints were saved with.
# ---------------------------------------------------------------------------
log "Installing core runtime dependencies"
run "${PIP[@]}" install \
    "torch==${TORCH_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    safetensors \
    einops

# FLA with --no-deps: its declared requirements (torch, triton, transformers,
# einops) are all already installed at the versions we want, and letting the
# resolver see them risks pulling a different torch or reverting triton.
log "Installing flash-linear-attention from git (${FLA_REF:0:12})"
run "${PIP[@]}" install --no-deps --upgrade "git+${FLA_REPO}@${FLA_REF}"

# ---------------------------------------------------------------------------
# Evaluation stack. torch is re-pinned on this line because several of these
# packages depend on torch transitively and would otherwise be free to upgrade
# it, orphaning the kernel wheels built against torch ${TORCH_MINOR}.
# The ruler extra pulls nltk/wonderwords/scipy for scripts/eval_on_ruler.sh; the
# longbench extra pulls jieba/fuzzywuzzy/rouge for the metrics that
# scripts/eval_on_longbench.sh's longbench_* tasks score with.
# ---------------------------------------------------------------------------
log "Installing evaluation dependencies"
run "${PIP[@]}" install \
    "torch==${TORCH_VERSION}" \
    "lm_eval[ruler,longbench]==${LM_EVAL_VERSION}" \
    "datasets==${DATASETS_VERSION}" \
    "evalscope[ifeval]==1.7.1" \
    modelscope \
    accelerate \
    jinja2 \
    numpy

# The package itself. --no-deps because every dependency it declares has just
# been installed deliberately above, several of them in forms pip could not
# resolve on its own (GitHub release wheels, a git ref).
log "Installing gen-distill (editable)"
run "${PIP[@]}" install --no-deps -e "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Triton, forced last.
#
# torch ${TORCH_VERSION} hard-pins triton==3.3.0, but FLA's chunk kernels
# (chunk_kda, chunk_gated_delta_rule, ...) fail to compile under 3.3.0 on
# compute capability 12.0: the TritonGPUAccelerateMatmul MLIR pass raises
# "PassManager::run failed" the first time a KDA kernel is compiled.
#
# This has to be the LAST install in the script. Every later pip resolve that
# mentions torch re-reads that pin and silently puts triton back to 3.3.0, so
# forcing it any earlier is undone by the time setup finishes. Afterwards
# `pip check` reports the deviation: that report is the intended state here.
# ---------------------------------------------------------------------------
log "Forcing triton ${TRITON_VERSION} (FLA chunk kernels)"
run "${PIP[@]}" install --no-deps --force-reinstall "triton==${TRITON_VERSION}"

# ---------------------------------------------------------------------------
# Verification.
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "Dry run complete - nothing was installed"
    exit 0
fi

log "Verifying installation"
GD_EXPECTED_TRITON="${TRITON_VERSION}" "${PY}" - <<'PYEOF'
import os
import sys

failures = []


def check(label, fn):
    try:
        print(f"{label:<22} {fn()}")
    except Exception as exc:  # noqa: BLE001 - we want to report, then fail loudly
        print(f"{label:<22} FAILED: {type(exc).__name__}: {exc}")
        failures.append(label)


import torch  # noqa: E402

check("torch", lambda: torch.__version__)
check("torch CUDA build", lambda: torch.version.cuda)
check("triton", lambda: __import__("triton").__version__)

# Assert the pin rather than just printing it. A pip resolve that mentions torch
# reverts triton to torch's own pin, which fails only when a KDA kernel is first
# compiled - inside an eval, long after setup reported success. Catch it here.
_expected_triton = os.environ.get("GD_EXPECTED_TRITON")
_actual_triton = __import__("triton").__version__
if _expected_triton and _actual_triton != _expected_triton:
    print(f"{'triton pin':<22} FAILED: is {_actual_triton}, expected {_expected_triton}")
    failures.append("triton pin")
check("transformers", lambda: __import__("transformers").__version__)
check("safetensors", lambda: __import__("safetensors").__version__)
check("einops", lambda: __import__("einops").__version__)
check("fla", lambda: __import__("fla").__version__)
check("mamba_ssm", lambda: __import__("mamba_ssm").__version__)
check("causal_conv1d", lambda: __import__("causal_conv1d").__version__)

# The imports the package actually performs at load time.
check("fla.ops.kda", lambda: (__import__("fla.ops.kda", fromlist=["chunk_kda"]).chunk_kda) and "OK")
check("mamba_ssm ssd_combined", lambda: (__import__(
    "mamba_ssm.ops.triton.ssd_combined", fromlist=["mamba_chunk_scan_combined"]
).mamba_chunk_scan_combined) and "OK")
check("gen_distill", lambda: (__import__("gen_distill").EfficientQwenForCausalLM) and "OK")

# The evaluation stack. Running the evals is the point of this repository, so a
# missing harness is a failure, not a note.
check("lm_eval", lambda: getattr(__import__("lm_eval"), "__version__", "installed"))
check("evalscope", lambda: getattr(__import__("evalscope"), "__version__", "installed"))
check("accelerate", lambda: getattr(__import__("accelerate"), "__version__", "installed"))

print()
print(f"{'CUDA available':<22} {torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    print(f"{'GPU':<22} {torch.cuda.get_device_name(0)} ({sm})")
    print(f"{'torch arch list':<22} {' '.join(torch.cuda.get_arch_list())}")
    if sm not in torch.cuda.get_arch_list():
        print(f"\nERROR: this torch build has no {sm} kernels for your GPU.")
        failures.append("gpu-arch")
else:
    print("\nWARNING: no CUDA GPU visible. Both sequence mixers need one at run "
          "time; there is no CPU path.")

if failures:
    print("\nFAILED: " + ", ".join(failures))
    sys.exit(1)
print("\nAll import checks passed.")
PYEOF

log "Setup complete"
echo "Activate with:  conda activate ${ENV_NAME}"
