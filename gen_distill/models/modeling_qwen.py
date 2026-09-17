import torch
import warnings
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging
from typing import Optional, Tuple, Dict

# Import Qwen3 components as base classes
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    Qwen3RMSNorm,
    Qwen3MLP,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

# Import custom mixers
from gen_distill.models.mixers import (
    Mamba2,
    DiscreteMamba2,
    EfficientKDA,
    EfficientGatedDeltaNet,
    EfficientGLA,
    EfficientLightningAttention,
)
from gen_distill.models.configuration_qwen import EfficientQwenConfig
from gen_distill.models.layers.laurel import LaurelBlock

logger = logging.get_logger(__name__)


class EfficientQwenRotaryEmbedding(Qwen3RotaryEmbedding):
    """
    EfficientQwen rotary embedding that extends Qwen3's rotary embedding.
    This is essentially a direct subclass with no modifications needed.
    """

    pass


class EfficientQwenRMSNorm(Qwen3RMSNorm):
    """
    EfficientQwen RMS normalization layer that extends Qwen3's RMS norm.
    This maintains compatibility with the Qwen3 implementation.
    """

    pass


class EfficientQwenMLP(Qwen3MLP):
    """
    EfficientQwen MLP that follows the Qwen3 gated MLP pattern.
    This is based on the Qwen3MLP but can be configured for different activation functions.
    """

    pass


class EfficientQwenAttention(nn.Module):
    """
    EfficientQwen attention module that extends Qwen3Attention with additional functionality.
    This maintains the Qwen3 attention implementation but adds mixer-specific outputs.
    """

    def __init__(self, config: EfficientQwenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Hardcoded to see if hte GVA setup works for the Hybrid model (the GVA setup should only
        # impact the SSM layers, but currently it impacts all the layers)
        self.n_q_heads = config.n_q_heads
        self.n_k_heads = config.n_k_heads
        self.n_v_heads = config.n_v_heads
        # self.n_q_heads = 16  # config.n_q_heads
        # self.n_k_heads = 8  # config.n_k_heads
        # self.n_v_heads = 8  # config.n_v_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_groups = self.n_q_heads // self.n_k_heads
        # num_key_value_groups is used to automatically do repeat_kv by the attention_interfaces!
        assert self.n_k_heads == self.n_v_heads, (
            "GQA requires n_k_heads == n_v_heads, but got " f"{self.n_k_heads} and {self.n_v_heads}"
        )
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.n_q_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.n_k_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.n_v_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_q_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_mixer_matrix: Optional[bool] = False,
        document_boundary: Optional[torch.LongTensor] = None,
        head_mask_binary: Optional[torch.Tensor] = None,
        head_sliding_window_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that returns a dictionary compatible with the CustomMixer interface.

        Args:
            head_mask_binary: Optional tensor of shape (num_heads,) with 1s for heads to keep
                and 0s for heads to mask (zero out their output). Used for head importance analysis.
            head_sliding_window_mask: Optional tensor of shape (1, num_heads, seq_len, seq_len)
                with per-head attention masks. This allows specific heads to use sliding window
                attention while others use full attention. When provided, overrides attention_mask
                for this layer and forces eager attention implementation.
        """
        input_shape = hidden_states.shape[:-1]  # [B, L]
        hidden_shape = (*input_shape, -1, self.head_dim)  # [B, L, H, E]

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B, H, L, E]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update_kv_cache(
                key_states, value_states, self.layer_idx, **cache_kwargs
            )

        # NOTE: Before I was using the eager attention implementation on eval,since I was getting
        # NaN values on the eval loop otherwise --> would need to debug this again
        if output_mixer_matrix and self.config._attn_implementation != "eager":
            raise ValueError(f"Output mixer matrix is only supported for eager attention "
                             f"implementation, got {self.config._attn_implementation}")

        # Determine the effective attention mask and implementation to use
        effective_attention_mask = attention_mask
        use_per_head_mask = head_sliding_window_mask is not None

        if use_per_head_mask:
            # Per-head sliding window masks require eager attention since we need to apply
            # different masks to different heads. Flash attention doesn't support this.
            if self.config._attn_implementation != "eager":
                logger.warning_once(
                    f"Per-head sliding window mask requires eager attention, but "
                    f"config uses {self.config._attn_implementation}. Falling back to eager."
                )
            attention_interface = eager_attention_forward
            # Use the per-head mask instead of the regular attention mask
            effective_attention_mask = head_sliding_window_mask
        else:
            # Select attention implementation strictly based on configuration
            # this is important in the case of document_boundary, since the attention mask is created
            # based on the _attn_implementation setting, and it needs to match the actual attention implementation
            # used in the forward pass
            if self.config._attn_implementation == "eager":
                attention_interface = eager_attention_forward
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2" and not use_per_head_mask:
            repeat_factor = self.n_q_heads // max(self.n_k_heads, self.n_v_heads)
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)
            value_states = value_states.repeat_interleave(repeat_factor, dim=1)

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            effective_attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window if not use_per_head_mask else None,
            **kwargs,
        )  # [B, L, H, E]

        # Apply head masking if provided (for head importance analysis)
        # head_mask_binary shape: (num_heads,) -> broadcast to [1, 1, H, 1]
        if head_mask_binary is not None:
            attn_output = attn_output * head_mask_binary[None, None, :, None].to(attn_output.device)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        self.attn_output = attn_output
        self.position_embeddings = position_embeddings
        self.attention_mask = attention_mask
        self.document_boundary = document_boundary
        self.cache_position = cache_position

        # # ONLY FOR DEBUGGING -> REMOVE AFTERWARDS!
        # if torch.isnan(attn_output).any():
        #     print("Using the attention implementation: ", self.config._attn_implementation)
        #     print(f"NaN values found in attn_outputs of layer {self.layer_idx}")
        #     nan_in_inputs = (
        #         torch.isnan(query_states).any()
        #         or torch.isnan(key_states).any()
        #         or torch.isnan(value_states).any()
        #     )
        #     print("nan_in_inputs: ", nan_in_inputs)
        #     print("position_embeddings: ", position_embeddings)
        #     print("attention_mask: ", attention_mask)
        #     print("document_boundary: ", document_boundary)
        #     print("cache_position: ", cache_position)
        #     print("Dtypes of inputs: ", query_states.dtype, key_states.dtype, value_states.dtype)
        #     print("Attention dropout: ", self.attention_dropout)
        #     print("Sliding window: ", self.sliding_window)
        #     print("Kwargs keys: ", kwargs.keys())
        #     print("Input ids that are problematic: ", kwargs["input_ids"][:, :10])
        #     raise ValueError("NaN values found in attn_outputs")

        attn_output = self.o_proj(attn_output)

        if torch.isnan(attn_output).any():
            print(f"NaN values detected in attn_output of layer {self.layer_idx}")

        outputs = {"output_states": attn_output}
        if output_mixer_matrix:
            outputs["mixer_matrix"] = attn_weights
        return outputs


class EfficientSequenceMixer(nn.Module):
    """
    CustomMixer that can use different sequence mixing strategies including attention, Mamba
    This is adapted from the HybridPythia implementation to work with EfficientQwen.
    """

    def __init__(self, config, layer_idx=None, mixer_type_override: Optional[str] = None):
        super().__init__()
        self.config = config

        current_mixer_type = (
            mixer_type_override if mixer_type_override is not None else config.mixer_type
        )
        self.current_mixer_type = current_mixer_type
        if current_mixer_type == "discrete_mamba2":
            self.mixer = DiscreteMamba2(
                d_model=self.config.hidden_size,
                layer_idx=layer_idx,
                **self.config.ssm_cfg,
            )
        elif self.current_mixer_type == "mamba2":
            self.mixer = Mamba2(
                d_model=self.config.hidden_size,
                layer_idx=layer_idx,
                **self.config.ssm_cfg,
            )
        elif self.current_mixer_type == "efficient_kda":
            # Kimi Delta Attention mixer wrapped in EfficientKDA adapter
            self.mixer = EfficientKDA(
                d_model=self.config.hidden_size,
                layer_idx=layer_idx,
                **self.config.ssm_cfg,
            )
        elif self.current_mixer_type == "gated_deltanet":
            # GDN uses the same EfficientKDA-style interface; see
            # gen_distill/models/mixers/gated_deltanet.py for the relation to KDA.
            self.mixer = EfficientGatedDeltaNet(
                d_model=self.config.hidden_size,
                layer_idx=layer_idx,
                **self.config.ssm_cfg,
            )
        elif self.current_mixer_type == "efficient_gla":
            # Gated Linear Attention (Yang et al. 2024) - K-space low-rank
            # forget gate shared across heads-in-a-group, no delta rule.
            self.mixer = EfficientGLA(
                d_model=self.config.hidden_size,
                layer_idx=layer_idx,
                **self.config.ssm_cfg,
            )
        elif self.current_mixer_type == "lightning_attention":
            # Lightning Attention (Qin et al. 2024a) - data-independent per-head
            # log-decay driven by simple_gla. num_layers is required to compute
            # the closed-form decay slope.
            self.mixer = EfficientLightningAttention(
                d_model=self.config.hidden_size,
                num_layers=self.config.num_hidden_layers,
                layer_idx=layer_idx,
                **self.config.ssm_cfg,
            )
        else:
            raise ValueError(f"Invalid mixer type: {current_mixer_type}")

    def forward(
        self,
        x,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        head_mask=None,
        use_cache=False,
        output_mixer_matrix=False,
        cache_position=None,
        position_embeddings=None,
        document_boundary=None,
        head_mask_binary=None,
        head_sliding_window_mask=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with the following keys:
        - "output_states": the output states of the mixer
        - "mixer_matrix": the mixer matrix (if output_mixer_matrix=True)

        Note: head_mask_binary and head_sliding_window_mask are accepted for interface
        compatibility but ignored by non-attention mixers (Mamba, etc.).
        """
        if self.current_mixer_type in [
            "mamba2",
            "discrete_mamba2",
            "efficient_kda",
            "gated_deltanet",
            "efficient_gla",
            "lightning_attention",
        ]:
            return self.mixer(
                x,
                output_mixer_matrix=output_mixer_matrix,
                cache_params=past_key_value,
                cache_position=cache_position,
                document_boundary=document_boundary,
                attention_mask=attention_mask,
            )
        else:
            raise ValueError(f"Invalid mixer type: {self.current_mixer_type}")


class EfficientQwenLayer(nn.Module):
    """
    EfficientQwen decoder layer that can use different mixers (attention, Mamba, etc.).
    This is based on HybridPythiaLayer but adapted for the Qwen3 architecture.
    """

    def __init__(self, config: EfficientQwenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.input_layernorm = EfficientQwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = EfficientQwenRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = EfficientQwenMLP(config)
        self.run_mlp_component = config.run_mlp_component
        self.attention_type = "full_attention"  # options [full_attention, sliding_attention]

        # Determine mixer type for this layer
        self.effective_mixer_type = "attention"  # Default
        # 1. Check layer_mixer_map (populated from --layer_specs)
        if config.layer_mixer_map and self.layer_idx in config.layer_mixer_map:
            self.effective_mixer_type = config.layer_mixer_map[self.layer_idx]
        # 2. Fallback to global custom_mixer_layers and mixer_type if not in layer_mixer_map
        elif config.custom_mixer_layers and self.layer_idx in config.custom_mixer_layers:
            self.effective_mixer_type = config.mixer_type

        if self.effective_mixer_type == "attention":
            self.self_attn = EfficientQwenAttention(config, layer_idx)
        else:
            # NOTE: deliberate divergence from the research repo, which catches this
            # ValueError and falls back to EfficientQwenAttention with only a logged
            # error. In an inference release that fallback silently returns a randomly
            # initialized attention layer whenever a checkpoint config names a mixer
            # this package cannot build, so the user gets a wrong model and no
            # exception. Fail loudly instead.
            try:
                self.self_attn = EfficientSequenceMixer(
                    config, layer_idx, mixer_type_override=self.effective_mixer_type
                )
            except ValueError as e:
                raise ValueError(
                    f"Failed to initialize the sequence mixer for layer {layer_idx} with "
                    f"mixer type '{self.effective_mixer_type}': {e}. Supported mixer types "
                    f"are: attention, mamba2, discrete_mamba2, efficient_kda, "
                    f"gated_deltanet, efficient_gla, lightning_attention."
                ) from e

        self.do_laurel = config.do_laurel
        if config.do_laurel:
            self.laurel = LaurelBlock(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Cache] = None,
        output_mixer_matrix: Optional[bool] = False,
        output_mixer_states: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        document_boundary: Optional[torch.LongTensor] = None,
        head_mask_binary: Optional[torch.Tensor] = None,
        head_sliding_window_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        # Self-attention/mixer block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # NOTE: In Gemma3n, they put the Laurel block after the first input layernorm
        # Reason is that the lauren is doing y = x + RMS(W_r W_l x) -->  since the second terms
        # is normed, we woudl ideally also like the first term to already be normed
        # TODO (juan): Gemma 3n seems to put the lauren only on the residual around the
        # attenbtion block. Why not also after the MLP block?
        if self.do_laurel:
            # residual = self.laurel(hidden_states)
            residual = self.laurel(residual)

        # if torch.isnan(hidden_states).any():
        #     warnings.warn(
        #         f"NaN detected in Hidden States at EfficientQwenLayer layer {self.layer_idx}"
        #     )

        mixer_outputs = self.self_attn(
            hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            head_mask=head_mask,
            use_cache=use_cache,
            output_mixer_matrix=output_mixer_matrix,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            document_boundary=document_boundary,
            head_mask_binary=head_mask_binary,
            head_sliding_window_mask=head_sliding_window_mask,
            **kwargs,
        )

        mixer_output_states = mixer_outputs["output_states"]
        mixer_matrix = mixer_outputs.get("mixer_matrix", None)

        # if torch.isnan(mixer_output_states).any():
        #     warnings.warn(
        #         f"NaN detected in Mixer Output States at EfficientQwenLayer layer {self.layer_idx}"
        #     )

        # Residual connection
        hidden_states = residual + mixer_output_states

        # MLP block
        if self.run_mlp_component:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(hidden_states)
            hidden_states = residual + mlp_output

        # if torch.isnan(hidden_states).any():
        #     warnings.warn(f"NaN detected after MLP in EfficientQwenLayer layer {self.layer_idx}")

        # outputs = {"output_states": hidden_states}
        # if output_mixer_states:
        #     outputs["mixer_states"] = mixer_output_states
        # if output_mixer_matrix:
        #     outputs["mixer_matrix"] = mixer_matrix

        # return outputs
        return hidden_states, mixer_output_states, mixer_matrix
