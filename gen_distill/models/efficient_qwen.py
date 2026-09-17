import os
import json
import warnings
import shutil
from typing import Optional, Tuple, Union, List, Dict
from functools import partial
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache, DynamicLayer
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils.generic import TransformersKwargs
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers import GenerationConfig
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file
from safetensors import safe_open

# Import our configuration and modeling components
from gen_distill.models.configuration_qwen import (
    EfficientQwenConfig,
    EfficientQwenModelOutput,
    EfficientQwenCausalLMOutput,
)
from gen_distill.models.modeling_qwen import (
    EfficientQwenRotaryEmbedding,
    EfficientQwenRMSNorm,
    EfficientQwenLayer,
)

logger = logging.get_logger(__name__)

IGNORE_INDEX = -100


def reset_position_ids_between_docs(position_ids: torch.LongTensor, document_boundary: torch.LongTensor) -> torch.LongTensor:
    """
    Reset the position ids to 0 between each document.

    Inputs:
        position_ids (torch.LongTensor): The position ids of shape (B, L)
        document_boundary (torch.LongTensor): The document boundary of shape (B,L)

    Outputs:
        position_ids (torch.LongTensor): The updated position ids of shape (B, L)
    """

    # Ensure boolean boundary mask and always treat index 0 as a boundary
    boundary = (document_boundary == 1)
    boundary[:, 0] = True

    # Compute positions as distance from the last boundary index
    batch_size, seq_len = position_ids.shape
    device = position_ids.device
    dtype = position_ids.dtype

    arange_idx = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, seq_len)
    reset_indices = torch.where(boundary, arange_idx, torch.full_like(arange_idx, -1))
    last_reset_indices, _ = reset_indices.cummax(dim=1)
    new_position_ids = arange_idx - last_reset_indices
    return new_position_ids


def apply_cross_document_eager_mask(
    causal_mask: torch.Tensor,
    document_boundary: torch.LongTensor,
) -> torch.Tensor:
    """
    Apply cross-document blocking to an existing additive causal mask for the eager backend.

    Args:
        causal_mask (torch.Tensor):
            A 4D additive attention mask of shape (B, 1, Q, KV) with values 0 where tokens
            should attend and -inf where they should be blocked.
        document_boundary (torch.LongTensor):
            A 2D tensor of shape (B, S) with value 1 at the start of each document (including
            index 0) and 0 otherwise. The cumulative sum over dim=1 defines per-token document ids.

    Returns:
        torch.Tensor: The modified 4D additive mask of shape (B, 1, Q, KV) where cross-document
        attention is additionally blocked (set to -inf) while preserving the original causal mask.

    Notes:
        - Uses values 0 (compute) and -inf (ignore), following eager mask conventions in transformers.
        - `document_boundary` is sliced from the right to align with the (Q, KV) lengths of `causal_mask`.
    """

    # If no document boundaries provided, return original mask
    if document_boundary is None:
        return causal_mask

    # Shapes
    batch_size, _, q_length, kv_length = causal_mask.shape
    device = causal_mask.device
    dtype = causal_mask.dtype

    # Compute per-token document ids from boundaries; slice to align with Q and KV
    doc_flags = document_boundary.to(dtype=torch.long, device=device)
    doc_ids = torch.cumsum(doc_flags, dim=1)
    doc_ids_q = doc_ids[:, -q_length:]
    doc_ids_k = doc_ids[:, -kv_length:]

    # Same-document boolean mask [B,1,Q,KV]
    same_doc = (doc_ids_q[:, :, None] == doc_ids_k[:, None, :])[:, None, :, :]

    # Build cross-document additive mask: 0 where allowed, -inf otherwise
    min_dtype = torch.finfo(dtype).min
    doc_mask = torch.full((batch_size, 1, q_length, kv_length), fill_value=min_dtype, device=device, dtype=dtype)
    doc_mask = doc_mask.masked_fill(same_doc, 0.0)

    # Combine with existing causal additive mask: take the stricter (more -inf)
    combined = torch.minimum(causal_mask, doc_mask)
    return combined.contiguous()


def apply_cross_document_sdpa_mask(
    causal_mask: Optional[torch.Tensor] = None,
    document_boundary: torch.LongTensor = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Apply cross-document blocking to an existing boolean causal mask for the SDPA backend.

    Args:
        causal_mask (Optional[torch.Tensor]):
            A 4D boolean attention mask of shape (B, 1, Q, KV) where True means compute and
            False means ignore. If None, the function returns None.
        document_boundary (torch.LongTensor):
            A 2D tensor of shape (B, S) with value 1 at the start of each document (including
            index 0) and 0 otherwise. The cumulative sum over dim=1 defines per-token document ids.
        **kwargs: Ignored. Present for call-site compatibility.

    Returns:
        Optional[torch.Tensor]: The modified 4D boolean mask (B, 1, Q, KV) with cross-document
        attention blocked (logical AND with same-document mask). If `document_boundary` is None,
        returns `causal_mask` unchanged.

    Notes:
        - `document_boundary` is sliced from the right to align with the (Q, KV) lengths of `causal_mask`.
    """
    if document_boundary is None:
        return causal_mask

    # Ensure boolean mask
    causal_mask_bool = causal_mask.to(dtype=torch.bool)

    # Shapes
    batch_size, _, q_length, kv_length = causal_mask_bool.shape
    device = causal_mask_bool.device

    # Compute per-token document ids from boundaries; slice to align with Q and KV
    doc_flags = document_boundary.to(dtype=torch.long, device=device)
    doc_ids = torch.cumsum(doc_flags, dim=1)
    doc_ids_q = doc_ids[:, -q_length:]
    doc_ids_k = doc_ids[:, -kv_length:]

    # Same-document boolean mask [B,1,Q,KV]
    same_doc = (doc_ids_q[:, :, None] == doc_ids_k[:, None, :])[:, None, :, :]

    # Combine with existing causal mask: attend only where both are True
    combined = causal_mask_bool & same_doc
    return combined.contiguous()


class EfficientQwenCache(Cache):
    """
    A cache class that supports heterogeneous layers found in EfficientQwen models.
    It can store standard Key-Value pairs for attention layers and specialized states
    (e.g., conv_state, ssm_state) for Mamba layers within the same cache object.

    The cache stores data in separate lists for different layer types.
    - For Attention layers: kv_cache stores tuples of (key_tensor, value_tensor)
    - For Mamba layers: mamba_cache stores dictionaries with {'conv': conv_state, 'ssm': ssm_state}
    """

    def __init__(self) -> None:
        super().__init__(layer_classes=DynamicLayer)
        self._seen_tokens = 0
        self._current_sequence_length = 0  # Track the current sequence length being processed

        self.kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        self.mamba_cache: List[Optional[Dict[str, torch.Tensor]]] = []
        self.kda_cache: List[Optional[Dict[str, torch.Tensor]]] = []
        self.gla_cache: List[Optional[Dict[str, torch.Tensor]]] = []

    def reset(self):
        """Reset all cache data."""
        self._seen_tokens = 0
        self._current_sequence_length = 0

        self.kv_cache = []
        self.mamba_cache = []

    def __len__(self):
        """Returns the number of layers currently present in the cache."""
        return max(len(self.kv_cache), len(self.mamba_cache))

    def _ensure_layer_capacity(self, layer_idx: int):
        """Ensures the cache lists are long enough to store data for layer_idx."""
        while len(self.kv_cache) <= layer_idx:
            self.kv_cache.append(None)
        while len(self.mamba_cache) <= layer_idx:
            self.mamba_cache.append(None)
        while len(self.kda_cache) <= layer_idx:
            self.kda_cache.append(None)
        while len(self.gla_cache) <= layer_idx:
            self.gla_cache.append(None)

    def get_seq_length(self, layer_idx: Optional[int] = None, use_orig_seq: bool = False) -> int:
        """Returns the total sequence length accommodated by the cache so far."""
        return self._seen_tokens

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> Optional[int]:
        raise NotImplementedError(
            "EfficientQwenCache grows dynamically, so it does not have a fixed maximum shape unless pre-allocated."
        )

    def update_seq_length(self, new_tokens: int):
        """Update the sequence length when new tokens are processed."""
        if self._seen_tokens == 0:
            # Initial processing
            self._seen_tokens = new_tokens
        else:
            # Generation step - add new tokens
            self._seen_tokens += new_tokens
        self._current_sequence_length = new_tokens

    def update_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the KV cache with the new Key-Value states for the layer `layer_idx`.
        This method is primarily intended for Attention layers.
        """
        self._ensure_layer_capacity(layer_idx)
        current_cache = self.kv_cache[layer_idx]
        if current_cache is None:
            self.kv_cache[layer_idx] = (key_states, value_states)
        elif isinstance(current_cache, tuple):
            cached_key, cached_value = current_cache
            new_key = torch.cat([cached_key, key_states], dim=-2)
            new_value = torch.cat([cached_value, value_states], dim=-2)
            self.kv_cache[layer_idx] = (new_key, new_value)
        else:
            raise TypeError(
                f"Layer {layer_idx} has incompatible cache type {type(current_cache)} for K/V update."
            )
        return self.kv_cache[layer_idx]

    def get_mamba_state(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieves the Mamba state dictionary for a given layer."""
        self._ensure_layer_capacity(layer_idx)
        return self.mamba_cache[layer_idx]

    def update_mamba_state(
        self,
        layer_idx: int,
        ssm_state: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> None:
        """Updates the Mamba state for a given layer."""
        self._ensure_layer_capacity(layer_idx)

        current_cache = self.mamba_cache[layer_idx]
        if current_cache is None:
            self.mamba_cache[layer_idx] = {
                "ssm_state": ssm_state.detach(),
                "conv_state": conv_state.detach(),
            }
        else:
            current_cache["ssm_state"].copy_(ssm_state.detach())
            current_cache["conv_state"].copy_(conv_state.detach())

    def get_kda_state(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieves the KDA state dictionary for a given layer."""
        self._ensure_layer_capacity(layer_idx)
        return self.kda_cache[layer_idx]

    def update_kda_state(
        self,
        layer_idx: int,
        recurrent_state: Optional[torch.Tensor] = None,
        conv_state_q: Optional[torch.Tensor] = None,
        conv_state_k: Optional[torch.Tensor] = None,
        conv_state_v: Optional[torch.Tensor] = None,
        offset: Optional[int] = None,
    ) -> None:
        """Updates the KDA state for a given layer."""
        self._ensure_layer_capacity(layer_idx)
        self.kda_cache[layer_idx] = {
            "recurrent_state": recurrent_state,
            "conv_state_q": conv_state_q,
            "conv_state_k": conv_state_k,
            "conv_state_v": conv_state_v,
            "layer_idx": layer_idx,
            "offset": offset,
        }

    def get_gla_state(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieves the GLA state dictionary for a given layer."""
        self._ensure_layer_capacity(layer_idx)
        return self.gla_cache[layer_idx]

    def update_gla_state(
        self,
        layer_idx: int,
        recurrent_state: Optional[torch.Tensor] = None,
        conv_state_q: Optional[torch.Tensor] = None,
        conv_state_k: Optional[torch.Tensor] = None,
        conv_state_v: Optional[torch.Tensor] = None,
        offset: Optional[int] = None,
    ) -> None:
        """Updates the GLA state for a given layer."""
        self._ensure_layer_capacity(layer_idx)
        self.gla_cache[layer_idx] = {
            "recurrent_state": recurrent_state,
            "conv_state_q": conv_state_q,
            "conv_state_k": conv_state_k,
            "conv_state_v": conv_state_v,
            "layer_idx": layer_idx,
            "offset": offset,
        }


class EfficientQwenPreTrainedModel(Qwen3PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models for EfficientQwen.
    """

    base_model_prefix = "model"
    _no_split_modules = ["EfficientQwenLayer"]
    _keys_to_ignore_on_load_unexpected = [
        r"self_attn.bias",
        r"self_attn.masked_bias",
        r"\.mixer\.gate_proj\.",
    ]
    # _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    config_class = EfficientQwenConfig

    @staticmethod
    def _convert_old_mamba_checkpoint(state_dict):
        """
        Convert old Mamba2 checkpoint format to new format.

        Old format: separate gate_proj and in_proj projections
        New format: combined in_proj = [gate_proj, old_in_proj]
        """
        gate_proj_keys = [k for k in state_dict.keys() if ".mixer.gate_proj." in k]
        if not gate_proj_keys:
            return state_dict

        # Get unique layer prefixes (e.g., "model.layers.1.self_attn.mixer.")
        layer_prefixes = {key.rsplit("gate_proj", 1)[0] for key in gate_proj_keys}

        new_state_dict = dict(state_dict)
        for prefix in layer_prefixes:
            gate_weight_key = f"{prefix}gate_proj.weight"
            in_weight_key = f"{prefix}in_proj.weight"

            # Merge weights: new in_proj = [gate_proj, old in_proj]
            new_state_dict[in_weight_key] = torch.cat(
                [state_dict[gate_weight_key], state_dict[in_weight_key]], dim=0
            )
            del new_state_dict[gate_weight_key]

            # Handle biases if present
            gate_bias_key = f"{prefix}gate_proj.bias"
            in_bias_key = f"{prefix}in_proj.bias"
            if gate_bias_key in state_dict:
                if in_bias_key in state_dict:
                    new_state_dict[in_bias_key] = torch.cat(
                        [state_dict[gate_bias_key], state_dict[in_bias_key]], dim=0
                    )
                del new_state_dict[gate_bias_key]

        return new_state_dict

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Override from_pretrained to handle old Mamba checkpoint format conversion.

        Detects old format (separate gate_proj and in_proj) and converts to new format
        (combined in_proj) before loading. Backs up original files with _old suffix.
        """
        checkpoint_path = pretrained_model_name_or_path

        if not (isinstance(checkpoint_path, str) and os.path.isdir(checkpoint_path)):
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")

        # Detect format and check if conversion needed (prefer single file over sharded)
        shard_files = set()
        if os.path.exists(safetensors_path):
            is_sharded = False
            with safe_open(safetensors_path, framework="pt") as f:
                keys = list(f.keys())
        elif os.path.exists(index_path):
            is_sharded = True
            with open(index_path, "r") as f:
                index_data = json.load(f)
            keys = list(index_data.get("weight_map", {}).keys())
            shard_files = set(index_data.get("weight_map", {}).values())
        else:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if not any(".mixer.gate_proj." in k for k in keys):
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        logger.info("Old Mamba checkpoint format detected. Converting...")

        # Load state dict
        if is_sharded:
            state_dict = {}
            for shard_file in shard_files:
                state_dict.update(safetensors_load_file(os.path.join(checkpoint_path, shard_file)))
        else:
            state_dict = safetensors_load_file(safetensors_path)

        # Convert and pass to parent
        converted_state_dict = cls._convert_old_mamba_checkpoint(state_dict)

        # Backup original files and save converted checkpoint
        if is_sharded:
            shutil.copy2(
                index_path, os.path.join(checkpoint_path, "model_old.safetensors.index.json")
            )
            for shard_file in shard_files:
                src = os.path.join(checkpoint_path, shard_file)
                shutil.copy2(src, src.replace(".safetensors", "_old.safetensors"))
            # Remove sharded files (will be replaced by single file)
            os.remove(index_path)
            for shard_file in shard_files:
                shard_path = os.path.join(checkpoint_path, shard_file)
                if os.path.exists(shard_path):
                    os.remove(shard_path)
            logger.info("Backed up and removed sharded checkpoint files")
        else:
            shutil.copy2(safetensors_path, os.path.join(checkpoint_path, "model_old.safetensors"))

        # Save converted checkpoint as single file
        safetensors_save_file(converted_state_dict, safetensors_path)
        logger.info(f"Saved converted checkpoint to {safetensors_path}")

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights"""
        if hasattr(module, "_init_custom_weights"):
            module._init_custom_weights()  # Init the Mamba2 parameters
        else:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, (nn.LayerNorm, EfficientQwenRMSNorm)):
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.zero_()
                module.weight.data.fill_(1.0)


class EfficientQwenModel(EfficientQwenPreTrainedModel):
    """
    EfficientQwen model that supports hybrid sequence mixers.
    This is the core model class that handles the embedding, layers, and final normalization.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [EfficientQwenLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = EfficientQwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = EfficientQwenRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        document_boundary: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_mixer_matrix: Optional[bool] = False,
        output_mixer_states: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layer_indices_to_process: Optional[List[int]] = None,
        head_mask_binary: Optional[Dict[int, torch.Tensor]] = None,
        head_sliding_window_mask: Optional[Dict[int, torch.Tensor]] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, EfficientQwenModelOutput]:
        """
        Args:
            head_mask_binary: Optional dict mapping layer_idx -> tensor of shape (num_heads,) with
                1s for heads to keep and 0s for heads to mask. Used for head importance analysis.
            head_sliding_window_mask: Optional dict mapping layer_idx -> tensor of shape
                (1, num_heads, seq_len, seq_len) with per-head attention masks. Heads can have
                sliding window masks while others have full causal masks.
        """

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = EfficientQwenCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if document_boundary is not None:
            # Need to update the position_ids - reset the position ids for each document!
            position_ids = reset_position_ids_between_docs(position_ids, document_boundary)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs
                )
            # Ensure mask tensors are contiguous for SDPA compatibility
            for k, v in list(causal_mask_mapping.items()):
                if isinstance(v, torch.Tensor):
                    causal_mask_mapping[k] = v.contiguous()

        # Ensure mask tensors are contiguous even if they were prepared externally (e.g., by `generate`)
        if isinstance(causal_mask_mapping, dict):
            for k, v in list(causal_mask_mapping.items()):
                if isinstance(v, torch.Tensor):
                    causal_mask_mapping[k] = v.contiguous()

        # If provided, add a cross-document blocking mask derived from `document_boundary`
        # NOTE: It is ok to use the attention type of Layer 0, since we are hardcoding
        # the attention type for all layers to be "full_attention"
        if document_boundary is not None:
            if self.config._attn_implementation == "eager":
                attention_mask = apply_cross_document_eager_mask(
                    causal_mask=causal_mask_mapping[self.layers[0].attention_type],
                    document_boundary=document_boundary,
                )
            elif self.config._attn_implementation == "sdpa":
                # HACK: we need to materialize the causal mask for the SDPA since we will modify
                # it due to document boundaries --> we leverage the eager_causal_mask creation
                # and convert the eager mask 0/-inf values to sdpa mask: boolean 1/0
                self.config._attn_implementation = "eager"
                eager_causal_mask = create_causal_mask(**mask_kwargs)
                self.config._attn_implementation = "sdpa"
                causal_mask = (eager_causal_mask == 0).bool().contiguous()
                attention_mask = apply_cross_document_sdpa_mask(
                    causal_mask=causal_mask,
                    document_boundary=document_boundary,
                )
            elif self.config._attn_implementation == "flash_attention_2":
                # For flash-attention, packed seq contaminiation is avoided by position ids and
                # attention_mask needs to be None
                attention_mask = None
            else:
                raise ValueError(f"We cannot have a custom attention mask for the implementation: {self.config._attn_implementation}")
        else:
            attention_mask = causal_mask_mapping[self.layers[0].attention_type]

        # Prepare head mask if needed
        converted_head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if head_mask is not None:
            head_mask = converted_head_mask.to(dtype=self.dtype, device=self.device)
        else:
            head_mask = converted_head_mask

        # Set the current sequence length in the cache for proper tracking
        if use_cache and past_key_values is not None:
            if hasattr(past_key_values, "update_seq_length"):
                current_tokens = inputs_embeds.shape[1]
                past_key_values.update_seq_length(current_tokens)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_mixer_matrices = () if output_mixer_matrix else None
        all_hidden_states = () if output_hidden_states else None
        all_mixer_states = () if output_mixer_states else None
        self.logged_sparsity = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # Get head_mask_binary for this layer if provided
                layer_head_mask_binary = None
                if head_mask_binary is not None and i in head_mask_binary:
                    layer_head_mask_binary = head_mask_binary[i]

                # Get head_sliding_window_mask for this layer if provided
                layer_head_sw_mask = None
                if head_sliding_window_mask is not None and i in head_sliding_window_mask:
                    layer_head_sw_mask = head_sliding_window_mask[i]

                hidden_states, mixer_states, mixer_matrix = self._gradient_checkpointing_func(
                    partial(layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    input_ids,
                    attention_mask,
                    position_ids,
                    head_mask[i] if head_mask is not None else None,
                    use_cache,
                    past_key_values,  # Full model Cache passed to each layer
                    output_mixer_matrix,
                    output_mixer_states,
                    cache_position,
                    position_embeddings,
                    document_boundary,
                    layer_head_mask_binary,
                    layer_head_sw_mask,
                )
            else:
                # Get head_mask_binary for this layer if provided
                layer_head_mask_binary = None
                if head_mask_binary is not None and i in head_mask_binary:
                    layer_head_mask_binary = head_mask_binary[i]

                # Get head_sliding_window_mask for this layer if provided
                layer_head_sw_mask = None
                if head_sliding_window_mask is not None and i in head_sliding_window_mask:
                    layer_head_sw_mask = head_sliding_window_mask[i]

                hidden_states, mixer_states, mixer_matrix = layer(
                    hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    past_key_value=past_key_values,  # Full model Cache passed to each layer
                    use_cache=use_cache,
                    output_mixer_matrix=output_mixer_matrix,
                    output_mixer_states=output_mixer_states,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    document_boundary=document_boundary,
                    head_mask_binary=layer_head_mask_binary,
                    head_sliding_window_mask=layer_head_sw_mask,
                    **flash_attn_kwargs,
                )
            # hidden_states = outputs["output_states"]

            if output_mixer_matrix:
                if layer_indices_to_process is not None and i in layer_indices_to_process:
                    # all_mixer_matrices = all_mixer_matrices + (outputs.get("mixer_matrix"),)
                    all_mixer_matrices = all_mixer_matrices + (mixer_matrix,)
                else:
                    all_mixer_matrices = all_mixer_matrices + (None,)
            if output_mixer_states:
                if layer_indices_to_process is not None and i in layer_indices_to_process:
                    # all_mixer_states = all_mixer_states + (outputs.get("mixer_states"),)
                    all_mixer_states = all_mixer_states + (mixer_states,)
                else:
                    all_mixer_states = all_mixer_states + (None,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            if layer_indices_to_process is not None and i in layer_indices_to_process:
                all_hidden_states = all_hidden_states + (hidden_states,)
            else:
                all_hidden_states = all_hidden_states + (None,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_mixer_matrices,
                    all_mixer_states,
                ]
                if v is not None
            )

        return EfficientQwenModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            mixer_matrices=all_mixer_matrices,
            mixer_states=all_mixer_states,
        )


class EfficientQwenForCausalLM(EfficientQwenPreTrainedModel, GenerationMixin):
    """
    EfficientQwen Model with a causal language modeling head.
    This model inherits from EfficientQwenPreTrainedModel and adds the language modeling head
    along with memory block functionality.
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, device=None, dtype=None, **kwargs) -> None:
        super().__init__(config, device, dtype, **kwargs)

        self.model = EfficientQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # [ANALYSIS ONLY] Persistent head mask for layer importance analysis.
        # When set via set_persistent_head_mask(), this mask is automatically applied
        # during ALL forward passes including generation. This allows layer masking
        # to work with model.generate() which doesn't support passing head_mask_binary.
        # Format: Dict[int, torch.Tensor] mapping layer_idx -> tensor of shape (num_heads,)
        # with 1s for heads to keep and 0s for heads to mask (zero out attention output).
        self._persistent_head_mask_binary: Optional[Dict[int, torch.Tensor]] = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _supports_default_dynamic_cache(self) -> bool:
        """Return False to avoid using DynamicCache during generation."""
        return False

    # =========================================================================
    # [ANALYSIS ONLY] Persistent head mask for layer importance analysis
    # =========================================================================

    def set_persistent_head_mask(
        self,
        head_mask_binary: Optional[Dict[int, torch.Tensor]]
    ) -> None:
        """
        [ANALYSIS ONLY] Set a persistent head mask that will be applied during ALL forward passes.

        This is useful for layer importance analysis where we want to mask certain layers
        during generation (model.generate()) which doesn't support passing head_mask_binary
        as a parameter.

        Args:
            head_mask_binary: Dict mapping layer_idx -> tensor of shape (num_heads,) with
                1s for heads to keep and 0s for heads to mask. Pass None to clear the mask.

        Example:
            # Mask all heads in layers 0, 5, and 10
            mask = {
                0: torch.zeros(num_heads, device=model.device),
                5: torch.zeros(num_heads, device=model.device),
                10: torch.zeros(num_heads, device=model.device),
            }
            model.set_persistent_head_mask(mask)

            # Now generate() will automatically apply the mask
            outputs = model.generate(input_ids, max_new_tokens=10)

            # Clear the mask
            model.set_persistent_head_mask(None)
        """
        self._persistent_head_mask_binary = head_mask_binary

    def get_persistent_head_mask(self) -> Optional[Dict[int, torch.Tensor]]:
        """[ANALYSIS ONLY] Get the current persistent head mask."""
        return self._persistent_head_mask_binary

    def clear_persistent_head_mask(self) -> None:
        """[ANALYSIS ONLY] Clear the persistent head mask."""
        self._persistent_head_mask_binary = None

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is None and kwargs.get("use_cache", False):
            past_key_values = EfficientQwenCache()

        # Call the parent implementation
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_mixer_matrix: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_mixer_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        layer_indices_to_process: Optional[List[int]] = None,
        document_boundary: Optional[torch.LongTensor] = None,
        head_mask_binary: Optional[Dict[int, torch.Tensor]] = None,
        head_sliding_window_mask: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, EfficientQwenCausalLMOutput]:
        """
        Args:
            head_mask_binary: Optional dict mapping layer_idx -> tensor of shape (num_heads,) with
                1s for heads to keep and 0s for heads to mask. Used for head importance analysis.
                If not provided but a persistent mask is set via set_persistent_head_mask(),
                the persistent mask will be used instead.
            head_sliding_window_mask: Optional dict mapping layer_idx -> tensor of shape
                (1, num_heads, seq_len, seq_len) with per-head attention masks. Heads can have
                sliding window masks while others have full causal masks.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # [ANALYSIS ONLY] Apply persistent head mask if no explicit mask is provided
        if head_mask_binary is None and self._persistent_head_mask_binary is not None:
            head_mask_binary = self._persistent_head_mask_binary

        if layer_indices_to_process is None:
            # We need to process all the layers
            layer_indices_to_process = list(range(len(self.model.layers)))

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            document_boundary=document_boundary,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_mixer_matrix=output_mixer_matrix,
            output_mixer_states=output_mixer_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            layer_indices_to_process=layer_indices_to_process,
            head_mask_binary=head_mask_binary,
            head_sliding_window_mask=head_sliding_window_mask,
            **kwargs,
        )
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return EfficientQwenCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else outputs.get(2),
            mixer_matrices=outputs.mixer_matrices if return_dict else outputs.get(3),
            mixer_states=outputs.mixer_states if return_dict else outputs.get(4),
        )


__all__ = [
    "EfficientQwenForCausalLM",
    "EfficientQwenLayer",
    "EfficientQwenModel",
    "EfficientQwenPreTrainedModel",
    "EfficientQwenCache",
]
