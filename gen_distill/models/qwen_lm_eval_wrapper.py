"""
Wrapper that lets lm-evaluation-harness treat EfficientQwen as a standard
causal Hugging-Face model.
"""

import torch
from typing import Optional, Union, List, Dict
import jinja2
import lm_eval
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from transformers import GenerationConfig


@register_model("efficient_qwen")
class EfficientQwenLMWrapper(HFLM):
    """Wrapper for EfficientQwen for compatibility with lm-evaluation-harness."""

    def __init__(self, pretrained: str, **kwargs) -> None:
        if "backend" in kwargs:
            assert kwargs["backend"] == "causal"
        self.enable_thinking = kwargs.pop("enable_thinking", False)
        self._rope_scaling_factor = kwargs.pop("rope_scaling_factor", None)
        self._rope_scaling_original_max_pos = kwargs.pop("rope_scaling_original_max_pos", None)

        super().__init__(
            pretrained=pretrained,
            backend=kwargs.pop("backend", "causal"),
            tokenizer=kwargs.pop("tokenizer", pretrained),  # default: same checkpoint
            max_length=kwargs.pop("max_length", 4096),
            **kwargs,
        )

    def _get_config(self, pretrained: str, **_kwargs) -> None:
        from gen_distill.models.configuration_qwen import EfficientQwenConfig

        self._config = EfficientQwenConfig.from_pretrained(pretrained)
        self._generation_config = GenerationConfig.from_pretrained(pretrained)

        # Apply YaRN RoPE scaling to extend context window beyond native max_position_embeddings
        if self._rope_scaling_factor is not None:
            factor = float(self._rope_scaling_factor)
            original_max_pos = int(self._rope_scaling_original_max_pos or self._config.max_position_embeddings)
            self._config.rope_scaling = {
                "type": "yarn",
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": original_max_pos,
            }
            self._config.max_position_embeddings = int(original_max_pos * factor)

    def _create_model(
        self,
        pretrained: str,
        dtype: Optional[Union[str, torch.dtype]] = "float16",
        **_kwargs,
    ) -> None:
        from gen_distill.models.efficient_qwen import EfficientQwenForCausalLM

        dtype = torch.bfloat16 if dtype == "auto" else lm_eval.models.utils.get_dtype(dtype)
        self._model = EfficientQwenForCausalLM.from_pretrained(pretrained)

        # Move model to device after creation to avoid device serialization issues
        self._model = self._model.to(dtype=dtype, device=self._device)

        self._model.config._attn_implementation = _kwargs['attn_implementation']

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # Strip unsupported kwargs that the harness may pass
        generation_kwargs.pop("attention_mask", None)
        generation_kwargs.pop("max_new_tokens", None)

        # Stopping criteria currently not supported; we over-generate then trim
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            **generation_kwargs,
        )

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                enable_thinking=self.enable_thinking,
            )
        except jinja2.exceptions.TemplateError:
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                enable_thinking=self.enable_thinking,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated
