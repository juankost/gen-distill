"""gen-distill: Inference and evaluation for distilled hybrid sequence models."""

from gen_distill.models.configuration_qwen import EfficientQwenConfig
from gen_distill.models.efficient_qwen import EfficientQwenForCausalLM

from transformers import AutoConfig, AutoModelForCausalLM

# Auto-register so that `import gen_distill` is all users need before from_pretrained()
AutoConfig.register("efficient_qwen", EfficientQwenConfig)
AutoModelForCausalLM.register(EfficientQwenConfig, EfficientQwenForCausalLM)
