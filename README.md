# When Perplexity Lies: Generation-Focused Distillation of Hybrid Sequence Models

Inference and evaluation code for the distilled hybrid models from the paper of the same name, published in Transactions on Machine Learning Research (TMLR), 2026.

- Paper (OpenReview): https://openreview.net/forum?id=u4sfTcn6Tx
- Preprint (arXiv): https://arxiv.org/abs/2603.26556

## What this repository is

This repository holds everything needed to run and evaluate the released checkpoints:

- the `EfficientQwen` config and model, which build a Qwen3 backbone where a selected subset of layers keeps softmax attention and every other layer is replaced by a linear sequence mixer,
- all five sequence mixers the paper studies: Kimi Delta Attention (KDA), Mamba2, Gated DeltaNet, Gated Linear Attention, and Lightning Attention,
- the evaluation harnesses the paper reports: EvalScope for generation-based scoring, the LM Evaluation Harness for log-likelihood scoring, plus LongBench and RULER needle-in-a-haystack for long context,
- an installer that builds a working environment in one command.

## Installation

```bash
bash setup.sh                 # creates the conda env 'gen_distill'
conda activate gen_distill
```

Run `bash setup.sh --dry-run` first if you want to see the fully resolved plan, every pinned version and every wheel URL, without installing anything. Every version is pinned in the script; there are only two options:

| Option | Purpose |
|---|---|
| `--recreate` | Delete and rebuild the environment. |
| `--dry-run` | Print every command, install nothing. |

Notes:

- There is one install path, not a per-GPU matrix. torch 2.7.0+cu128 ships kernels for compute capabilities 7.5 through 12.0, so the same install covers that whole range.
- Nothing is compiled. Expect roughly 6 GB of downloads and about 15 GB on disk.
- `triton` is force-bumped to 3.4.0 after PyTorch is installed, because FLA's chunked kernels fail to compile under the 3.3.0 that torch 2.7.0 pins. The two cannot be resolved together in one pip invocation, so the bump has to be the last step of the install, and `pip check` afterwards reports the deviation. That report is the intended state.
- `transformers` is pinned to 4.54.1. The cache class here uses an API whose signature changed in later releases, and 4.54.1 is the version the released checkpoints were saved with.
- `causal-conv1d` is a hard requirement, not an optional accelerator. Without it the short-convolution layer falls through to a Triton path that current FLA no longer ships, and raises at the first forward.
- The installer ends by importing every dependency, printing the resolved versions, and checking that your GPU's compute capability is in the PyTorch build's architecture list. It exits non-zero if any of that fails.

## Running the model

Importing `gen_distill` registers the config and model with the `transformers` auto classes, so `AutoModelForCausalLM` resolves the architecture. Do it before `from_pretrained`.

```python
import torch
import gen_distill  # registers EfficientQwenConfig / EfficientQwenForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TODO"  # HuggingFace repo id, or a local checkpoint directory

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")
model.eval()

messages = [{"role": "user", "content": "Explain what a state space model is, in two sentences."}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )

print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

Pass `do_sample=False` for the deterministic setting used in the paper's evaluations. These students are distilled from an instruction-tuned teacher, so prompts should go through the chat template. Feeding raw text works but is out of distribution.

### Architectures

Which layers keep attention and which get a linear mixer is stored in the checkpoint's `layer_specs`, so you do not normally set it by hand. The released 0.6B models use the values below over the 28 layers of Qwen3-0.6B, indexed from 0. The seven retained attention layers were selected by the beam search described in the paper, and are the same `{0, 2, 6, 11, 13, 18, 21}` in every hybrid.

| Architecture | Mixer type string | `layer_specs` |
|---|---|---|
| Hybrid-KDA | `efficient_kda` | `attention:0,2,6,11,13,18,21;efficient_kda:1,3,4,5,7,8,9,10,12,14,15,16,17,19,20,22,23,24,25,26,27` |
| Hybrid Mamba2 | `discrete_mamba2` | `attention:0,2,6,11,13,18,21;discrete_mamba2:1,3,4,5,7,8,9,10,12,14,15,16,17,19,20,22,23,24,25,26,27` |
| Hybrid Gated DeltaNet | `gated_deltanet` | `attention:0,2,6,11,13,18,21;gated_deltanet:1,3,...,27` |
| Hybrid Gated Linear Attention | `efficient_gla` | `attention:0,2,6,11,13,18,21;efficient_gla:1,3,...,27` |
| Hybrid Lightning Attention | `lightning_attention` | `attention:0,2,6,11,13,18,21;lightning_attention:1,3,...,27` |
| Pure KDA | `efficient_kda` | `efficient_kda:0,1,2,...,27` |
| Pure Mamba2 | `discrete_mamba2` | `discrete_mamba2:0,1,2,...,27` |

An unknown or unconstructible mixer type raises. It does not silently fall back to attention, so a checkpoint can never load into the wrong architecture without telling you.

## Released checkpoints

All 0.6B students are distilled from Qwen3-0.6B and the 1.7B student from Qwen3-1.7B, using the recipe the paper selects under generation-based evaluation: knowledge distillation, completion-only loss masking, and frozen attention layers during instruction tuning.

| Model | Teacher | Architecture | HuggingFace |
|---|---|---|---|
| Hybrid-KDA 0.6B | Qwen3-0.6B | 7 of 28 attention layers retained, 21 KDA | `TODO` |
| Hybrid Mamba2 0.6B | Qwen3-0.6B | 7 of 28 attention layers retained, 21 Mamba2 | `TODO` |
| Hybrid Gated DeltaNet 0.6B | Qwen3-0.6B | 7 of 28 attention layers retained, 21 GDN | `TODO` |
| Hybrid GLA 0.6B | Qwen3-0.6B | 7 of 28 attention layers retained, 21 GLA | `TODO` |
| Hybrid Lightning 0.6B | Qwen3-0.6B | 7 of 28 attention layers retained, 21 Lightning | `TODO` |
| Pure KDA 0.6B | Qwen3-0.6B | no attention retained, 28 KDA | `TODO` |
| Pure Mamba2 0.6B | Qwen3-0.6B | no attention retained, 28 Mamba2 | `TODO` |
| Hybrid-KDA 1.7B | Qwen3-1.7B | 7 of 28 attention layers retained, 21 KDA | `TODO` |

Hybrid-KDA is the recommended model. The other mixers are reported in the paper as an architecture comparison.

## Evaluation

The paper scores every model under two protocols, because they disagree. Generation-based evaluation makes the model produce its answer autoregressively. Log-likelihood evaluation ranks fixed candidate answers by their score under the model. The central finding is that the second protocol understates the teacher-student gap and can reverse the ranking of design choices, so generation is the protocol to trust.

### Everything at once

```bash
bash examples/run_all_evals.sh --checkpoint-path TODO --bf16 --greedy
```

Check the plumbing first, which takes minutes instead of hours:

```bash
bash examples/run_all_evals.sh --checkpoint-path TODO --smoke --dry-run  # print the plan only
bash examples/run_all_evals.sh --checkpoint-path TODO --smoke            # few samples per task
```

`--skip-language`, `--skip-longbench`, `--skip-ruler` and `--skip-ruler-extended` select a subset, and `--gpu-id` pins the device.

### Short context, both protocols

```bash
bash scripts/eval_on_language.sh --checkpoint-path TODO --bf16 --greedy \
    --evalscope-tasks ceval,mmlu_redux,cmmlu,arc,hellaswag,winogrande,bbh,gsm8k,humaneval,ifeval \
    --lm-eval-tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,lambada_openai,mmlu,cmmlu
```

The task split is the paper's. MMLU-Redux is generation-only and plain MMLU is used for log-likelihood scoring, because MMLU-Redux exists only in generative form in that harness. PIQA and LAMBADA exist only under the log-likelihood protocol.

### Long context

```bash
bash scripts/eval_on_longbench.sh --checkpoint-path TODO \
    --tasks longbench_dureader,longbench_hotpotqa,longbench_musique,longbench_narrativeqa,longbench_qmsum,longbench_triviaqa

bash scripts/eval_on_ruler.sh --checkpoint-path TODO --seq-lengths "4096,8192,16384,32768"
bash scripts/eval_on_ruler.sh --checkpoint-path TODO --seq-lengths "65536,131072" --rope-scaling-factor 4.0
```

Those are the six LongBench tasks and the RULER NIAH-single lengths the paper reports. Beyond 32K the models need YaRN scaling, which is what the second RULER command applies.

### Reproducing the published numbers

The script defaults are tuned for a quick check, not for reproduction. The paper uses an evaluation batch size of 64 and averages over three seeds, so a faithful run means raising `--bsz` and looping `--seed 1 2 3`. Expect hours to days per checkpoint for the full suite.

## Citation

```bibtex
@article{kostelec2026perplexity,
  title   = {When Perplexity Lies: Generation-Focused Distillation of Hybrid Sequence Models},
  author  = {Kostelec, Juan Gabriel and Guo, Qinghai},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  volume  = {TODO},
  url     = {https://openreview.net/forum?id=u4sfTcn6Tx},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
