from evalscope import run_task, TaskConfig
from transformers import set_seed
import os
import argparse

os.environ["NUMEXPR_MAX_THREADS"] = "64"

OUTPUT_DIR = os.environ.get("GEN_DISTILL_EVAL_OUTPUT", "./eval_outputs/evalscope")

# Register the EfficientQwen model with HuggingFace AutoModel
import gen_distill  # noqa — triggers AutoConfig/AutoModel registration


# ---------------------------------------------------------------------------
# Monkey-patch MBPP adapter to support local (non-sandbox) code execution.
# The upstream MBPPAdapter.match_score raises RuntimeError when use_sandbox is
# False, but Docker is unavailable in this environment.  We add a local
# execution fallback that runs code in an isolated subprocess (same approach
# HumanEval uses for its local execution path).
# ---------------------------------------------------------------------------
def _mbpp_run_code_local(code: str, timeout: float) -> dict:
    """Execute MBPP code+tests in a subprocess with timeout, returning result dict."""
    import multiprocessing
    from evalscope.benchmarks.humaneval.utils import unsafe_execute

    # Build a problem dict that makes unsafe_execute run our code directly.
    # unsafe_execute builds: prompt + completion + '\n' + test + '\n' + 'check(entry_point)'
    # We put everything in 'prompt' and leave the rest empty, then define a no-op check().
    problem = {
        "prompt": code + "\ndef check(*a, **k): pass\n",
        "test": "",
        "entry_point": "",
    }

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=unsafe_execute, args=(problem, "", timeout, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        result.append("timed out")
    return {"task_id": "mbpp", "passed": result[0] == "passed", "result": result[0]}


def _mbpp_match_score_with_local_fallback(
    self, original_prediction, filtered_prediction, reference, task_state
):
    from evalscope.api.metric import Score

    score = Score(
        extracted_prediction=filtered_prediction, prediction=original_prediction
    )
    problem = task_state.metadata
    completion = filtered_prediction
    for test in problem["test_list"]:
        completion += "\n" + test + "\n"

    if not self.use_sandbox:
        res = _mbpp_run_code_local(completion, self.review_timeout)
        passed = res["passed"]
    else:
        res = self.execute_code_in_sandbox(
            code=completion, timeout=self.review_timeout, language="python"
        )
        passed = res.get("status") == "success"

    score.value = {"acc": passed}
    score.metadata = {
        "task_id": problem["task_id"],
        "timeout": self.review_timeout,
        "execution_result": res,
    }
    score.main_score_name = "acc"
    return score


try:
    from evalscope.benchmarks.mbpp.mbpp_adapter import MBPPAdapter

    MBPPAdapter.match_score = _mbpp_match_score_with_local_fallback
except ImportError:
    pass


DATASETS_ARGS = {
    "gpqa_diamond": {
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "ceval": {
        "few_shot_num": 5,
        "few_shot_random": False,
    },
    "mmlu_redux": {
        "few_shot_num": 5,
        "few_shot_random": False,
    },
    "mmlu": {
        "few_shot_num": 5,
        "few_shot_random": False,
    },
    "live_code_bench": {
        "subset_list": ["release_v5"],
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "math_500": {
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "ifeval": {
        "few_shot_num": 0,
        "few_shot_random": False,
        "filters": {"remove_until": "</think>"},
    },
    "winogrande": {
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "arc": {
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "hellaswag": {
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "needle_haystack": {
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "bbh": {
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "cmmlu": {
        "few_shot_num": 0,  # CMMLU has train_split=None, so few-shot is not supported
        "few_shot_random": False,
    },
    "gsm8k": {
        "few_shot_num": 4,  # should enable cot!
        "few_shot_random": False,
    },
    "humaneval": {  # evalplus? is zero shot - average of humaneval and humaneval+
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "truthful_qa": {
        "few_shot_num": 0,
        "few_shot_random": False,
    },
    "mbpp": {
        "few_shot_num": 0,
        "few_shot_random": False,
    },
}

# Per-task recommended max new tokens.  Multiple-choice tasks need very few;
# generation tasks (code, math CoT, instruction following) need more.
DEFAULT_MAX_NEW_TOKENS = {
    "arc": 64,
    "hellaswag": 64,
    "winogrande": 64,
    "ceval": 64,
    "cmmlu": 64,
    "piqa": 64,
    "truthful_qa": 64,
    "mmlu": 512,  # CoT template asks for step-by-step reasoning
    "mmlu_redux": 512,
    "bbh": 512,
    "gsm8k": 512,  # multi-step math reasoning + \boxed{answer}
    "math_500": 1024,
    "humaneval": 1024,  # full function implementation
    "mbpp": 1024,
    "live_code_bench": 1024,
    "gpqa_diamond": 1024,
    "ifeval": 2048,  # free-form, prompts may require specific word/paragraph counts
    "needle_haystack": 512,
}
FALLBACK_MAX_NEW_TOKENS = 512


# Automatically adjust the batch size if (OOM) error
def run_task_with_auto_adjust_bsz(task_config):
    bsz = task_config.eval_batch_size
    while bsz > 1:
        try:
            eval_results = run_task(task_cfg=task_config)
            break
        except Exception as e:
            if any(
                ["OOM" in str(e), "out of memory" in str(e), "MemoryError" in str(e)]
            ):
                print(f"Error: {e}")
                bsz /= 2
                task_config.eval_batch_size = bsz
            else:
                # Not OOM error, raise the error
                raise e
    if bsz == 1:
        eval_results = run_task(
            task_cfg=task_config
        )  # if it fails here, it will raise the error
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evalscope evaluations with dynamic config"
    )
    parser.add_argument(
        "--test_datasets",
        type=str,
        nargs="+",
        default=["ceval"],
        help="Comma-separated or space-separated subset of DATASETS to evaluate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run evaluation on all available test datasets",
    )
    parser.add_argument(
        "--thinking_mode",
        action="store_true",
        help="Enable thinking mode",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to evaluate per dataset (for testing purposes). If not specified, evaluates all samples.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate (overrides per-task defaults from DEFAULT_MAX_NEW_TOKENS)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Initial evaluation batch size; will auto-reduce on OOM",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Name ID of HF model, or path to checkpoint folder",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Run evaluation using greedy generation configs",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        help="Model precision (float16, float32, bfloat16, auto)",
    )
    parser.add_argument(
        "--output_path", type=str, default=OUTPUT_DIR, help="Results saving place"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda",
        choices=["cuda", "auto", "balanced", "balanced_low_0", "sequential"],
        help="Device map strategy for transformers engine: 'cuda' (single GPU), 'auto' (multi-GPU), etc.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--few-shot-num",
        type=int,
        default=None,
        help="If set, override the per-task few_shot_num in DATASETS_ARGS for every evaluated dataset.",
    )

    args = parser.parse_args()
    if args.few_shot_num is not None:
        for _ds in DATASETS_ARGS:
            DATASETS_ARGS[_ds]["few_shot_num"] = args.few_shot_num
    set_seed(args.seed)
    DATASETS = list(DATASETS_ARGS.keys())
    if args.all:
        args.test_datasets = DATASETS
    else:
        # Parse comma-separated strings and flatten the list
        test_datasets_list = []
        for dataset_arg in args.test_datasets:
            # Split by comma and strip whitespace
            datasets = [d.strip() for d in dataset_arg.split(",")]
            test_datasets_list.extend(datasets)

        # Validate all datasets
        invalid_datasets = [d for d in test_datasets_list if d not in DATASETS]
        if invalid_datasets:
            raise ValueError(
                f"Invalid dataset(s): {invalid_datasets}. "
                f"Valid choices are: {', '.join(DATASETS)}"
            )

        # Remove duplicates while preserving order
        test_datasets = []
        seen = set()
        for d in test_datasets_list:
            if d not in seen:
                test_datasets.append(d)
                seen.add(d)
        args.test_datasets = test_datasets

    model_path = args.model_name_or_path
    folder_name = model_path.split("/")[-1]
    test_datasets = args.test_datasets
    sub_folder_name = "thinking" if args.thinking_mode else "nonthinking"

    # Resolve max_new_tokens: CLI override > per-task defaults > fallback
    if args.max_new_tokens is not None:
        effective_max_new_tokens = args.max_new_tokens
    else:
        effective_max_new_tokens = max(
            DEFAULT_MAX_NEW_TOKENS.get(ds, FALLBACK_MAX_NEW_TOKENS)
            for ds in test_datasets
        )
    print(f"  max_new_tokens={effective_max_new_tokens} for tasks={test_datasets}")

    if args.greedy:
        generation_config = {
            "do_sample": False,
            "max_tokens": effective_max_new_tokens,
        }
    else:
        generation_config = {
            "do_sample": True,
            "temperature": 0.6 if args.thinking_mode else 0.7,
            "max_tokens": effective_max_new_tokens,
            "top_p": 0.95 if args.thinking_mode else 0.8,
            "top_k": 20,
            "seed": 42,
        }
    model_args = {
        "revision": "master",
        "precision": args.precision,
        "device_map": args.device_map,
        "enable_thinking": args.thinking_mode,
        "_attn_implementation": "sdpa",
    }
    task_config_kwargs = {
        "model": model_path,
        "datasets": test_datasets,
        "dataset_hub": "modelscope",
        "model_args": model_args,
        "generation_config": generation_config,
        "dataset_args": {name: DATASETS_ARGS[name] for name in test_datasets},
        "limit": args.limit,
        "work_dir": os.path.join(args.output_path, sub_folder_name),
        "eval_batch_size": args.batch_size,
        "seed": args.seed,
        "repeats": 10 if "gpqa_diamond" in test_datasets else 1,
    }
    task_config = TaskConfig(**task_config_kwargs)
    eval_results = run_task_with_auto_adjust_bsz(task_config=task_config)
