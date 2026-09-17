import torch
import numpy as np
import random
from lm_eval.__main__ import cli_evaluate  # noqa

# Importing the wrapper registers it with lm-eval under the name "efficient_qwen"
from gen_distill.models.qwen_lm_eval_wrapper import EfficientQwenLMWrapper  # noqa


if __name__ == "__main__":

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(20)
    cli_evaluate()
