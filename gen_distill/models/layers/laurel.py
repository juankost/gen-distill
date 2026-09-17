import torch
import torch.nn as nn

from gen_distill.models.configuration_qwen import EfficientQwenConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


class LaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: EfficientQwenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.linear_left = nn.Linear(self.config.hidden_size, self.config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(self.config.laurel_rank, self.config.hidden_size, bias=False)
        self.post_laurel_norm = Qwen3RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
        return hidden_states + normed_laurel_hidden_states

    def _init_custom_weights(self) -> None:
        """Custom initialization for LaurelBlock."""
        # Initialize left like other Linear layers in this model
        self.linear_left.weight.data.normal_(mean=0.0, std=0.0005)
        if self.linear_left.bias is not None:
            self.linear_left.bias.data.zero_()

        # Right is explicitly zero-initialized
        self.linear_right.weight.data.normal_(mean=0.0, std=0.0005)
        if self.linear_right.bias is not None:
            self.linear_right.bias.data.zero_()

        # Initialize also the RMS norm
        self.post_laurel_norm.weight.data.fill_(1.0)
