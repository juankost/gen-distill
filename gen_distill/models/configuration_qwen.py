from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.mamba.modeling_mamba import MambaConfig
from transformers.modeling_rope_utils import rope_config_validation
import torch

logger = logging.get_logger(__name__)


class EfficientQwenConfig(MambaConfig, Qwen3Config):
    """
    Configuration class for EfficientQwen models.

    This configuration class extends Qwen2Config to add support for hybrid sequence mixers,
    including attention, Mamba, and n-gram based mixers.

    It needs to support the arguments that appear in the Qwen checkpoints, but then it should
    convert them to the new notation that is used by the updatd EfficientKDA, Mamba,
    EfficientQwenAttention layers.

    Qwen checkpoint uses: head_dim, num_attention_heads, num_key_value_heads
    --> need to convert this to n_q_heads, n_k_heads, n_v_heads, d_state, head_dim?

    """

    model_type = "efficient_qwen"

    def __init__(
        self,
        mixer_type="attention",
        custom_mixer_layers=[],
        ssm_cfg=None,
        run_mlp_component=True,
        layer_specs=None,
        # SSM specific configs
        ssm_activation=None,
        use_qk_norm=False,
        use_post_ssm_norm=False,
        do_discretization=False,
        # Low Rank Projection layer specific configs
        do_laurel=False,
        laurel_rank=16,
        # Grouped Latent Attention specific configs
        qk_head_dim=None,
        latent_head_dim=None,
        num_query_heads=None,
        num_key_heads=None,
        num_value_heads=None,
        num_latent_heads=None,
        layer_types=None,
        use_gqa_setup=True,
        use_gva_setup=False,
        d_state=None,
        d_model=None,
        expand_q=1,
        expand_k=1,
        expand_v=1,
        allow_neg_eigval=False,
        out_gate_activation="sigmoid",
        use_short_conv=True,
        bla_num_modes=4,
        bla_period=128.0,
        **kwargs,
    ):
        # Call parent classes initialization explicitly
        MambaConfig.__init__(self, **kwargs)
        Qwen3Config.__init__(self, **kwargs)
        # this initialized the values: head_dim, num_attention_heads, num_key_value_heads, expand,
        # hidden_size

        # Now initalize the parameters specific to EfficientQwen
        self.mixer_type = mixer_type
        self.custom_mixer_layers = custom_mixer_layers
        self.ssm_cfg = ssm_cfg
        self.ssm_activation = ssm_activation
        self.do_discretization = do_discretization
        self.run_mlp_component = run_mlp_component
        self.layer_specs = layer_specs if layer_specs is not None else []
        self.layer_mixer_map: Dict[int, str] = self.parse_layer_specs(layer_specs)
        self.use_qk_norm = use_qk_norm
        self.use_post_ssm_norm = use_post_ssm_norm
        self.use_gqa_setup = use_gqa_setup
        self.use_gva_setup = use_gva_setup
        self.do_laurel = do_laurel
        self.laurel_rank = laurel_rank
        self.out_gate_activation = out_gate_activation  # Kimi specific
        self.use_short_conv = use_short_conv
        self.bla_num_modes = bla_num_modes
        self.bla_period = bla_period
        # Adapt them to the new notation that is used by the updated EfficientKDA, Mamba,
        # EfficientQwenAttention layers.
        self.n_q_heads = (
            num_query_heads if num_query_heads is not None else self.num_attention_heads
        )
        self.n_k_heads = (
            num_key_heads if num_key_heads is not None else self.num_key_value_heads
        )
        self.n_v_heads = (
            num_value_heads if num_value_heads is not None else self.num_key_value_heads
        )

        # New notation for the head dims!
        self.d_state = (
            d_state if d_state is not None else self.head_dim
        )  # SSM <-> Transformer dims
        self.d_model = (
            d_model if d_model is not None else self.hidden_size
        )  # SSM <-> Transformer
        self.expand_q = expand_q
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.head_k_dim = int(self.d_state * self.expand_k)  # = head_dim * expand_k
        self.head_q_dim = int(self.d_state * self.expand_q)  # = head_dim * expand_q
        self.head_v_dim = int(self.d_state * self.expand_v)  # = head_dim * expand_v
        self.allow_neg_eigval = allow_neg_eigval

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        # Grouped Latent Attention parameters expected by GroupedLatentAttention
        self.qk_head_dim = (
            qk_head_dim
            if qk_head_dim is not None
            else getattr(self, "head_dim", self.hidden_size // self.num_attention_heads)
        )
        self.latent_head_dim = latent_head_dim
        self.num_latent_heads = num_latent_heads

        # Ensure pad_token_id is set, defaulting to eos_token_id if not present
        if self.pad_token_id is None and self.eos_token_id is not None:
            logger.info(
                f"Setting pad_token_id to eos_token_id: {self.eos_token_id} in config"
            )
            self.pad_token_id = self.eos_token_id

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # Populate the mixer sub-configuration so a config built by hand is usable, not
        # only one loaded from a checkpoint. This is a no-op when ssm_cfg is already set
        # (a checkpoint's own ssm_cfg wins), since set_ssm_configs only fills a missing one.
        self.set_ssm_configs()

    def set_ssm_configs(self):
        """
        We can only set the SSM configs once we know the layer specs
        """
        # Default SSM configuration for standard Mamba2 layers
        self.ssm_cfg = self.ssm_cfg or {
            "d_state": self.d_state,
            # "d_model": self.d_model,  # needs to be passed as a positional argument to the mixer!
            "d_conv": 4,
            "n_q_heads": self.n_q_heads,
            "n_k_heads": self.n_k_heads,
            "n_v_heads": self.n_v_heads,
            "expand_q": self.expand_q,
            "expand_k": self.expand_k,
            "expand_v": self.expand_v,
            "activation": self.ssm_activation or "identity",
            "chunk_size": 128,
            "bias": False,
            "conv_bias": True,
            "rms_norm_eps": self.rms_norm_eps,
            "allow_neg_eigval": self.allow_neg_eigval,
            "use_qk_norm": self.use_qk_norm,
            "use_post_ssm_norm": self.use_post_ssm_norm,
            "use_short_conv": self.use_short_conv,
            "out_gate_activation": self.out_gate_activation,
            "use_gqa_setup": self.use_gqa_setup,
            "use_gva_setup": self.use_gva_setup,
        }

    def parse_layer_specs(self, layer_specs):
        """Parse layer specifications to determine which mixer to use for each layer."""
        layer_mixer_map = {}
        if layer_specs:
            layer_specs = layer_specs.split(";")
            for spec_str in layer_specs:
                try:
                    mixer_type_str, indices_str = spec_str.split(":", 1)
                    mixer_type_str = mixer_type_str.strip()
                    indices = [int(idx.strip()) for idx in indices_str.split(",")]
                    for idx in indices:
                        if idx in layer_mixer_map:
                            logger.warning(
                                f"Layer index {idx} specified multiple times in layer_specs. "
                                f"Using last encountered type: '{mixer_type_str}'. "
                                f"Previous: '{layer_mixer_map[idx]}'."
                            )
                        layer_mixer_map[idx] = mixer_type_str
                except ValueError:
                    logger.warning(
                        f"Malformed layer_spec string: '{spec_str}'. Expected format 'type:idx1,idx2,...'. Skipping."
                    )
                except Exception as e:
                    logger.warning(
                        f"Error parsing layer_spec string '{spec_str}': {e}. Skipping."
                    )
        else:
            logger.warning_once(
                "No layer specs specified. Are you still using the old format?"
            )
            logger.warning_once("Using the old format to create the layer mixer map")
            for idx in range(self.num_hidden_layers):
                if (
                    self.custom_mixer_layers is not None
                    and idx in self.custom_mixer_layers
                ):
                    layer_mixer_map[idx] = self.mixer_type
                else:
                    layer_mixer_map[idx] = "attention"

        return layer_mixer_map


@dataclass
class EfficientQwenModelOutput(BaseModelOutputWithPast):
    """
    Output class for EfficientQwen model that includes additional mixer-specific outputs.
    """

    mixer_matrices: Optional[Tuple[torch.FloatTensor, ...]] = None
    mixer_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class EfficientQwenCausalLMOutput(CausalLMOutputWithPast):
    """
    Output class for EfficientQwen causal language model that includes additional mixer-specific outputs
    and memory block adaptations.
    """

    mixer_matrices: Optional[Tuple[torch.FloatTensor, ...]] = None
    mixer_states: Optional[Tuple[torch.FloatTensor, ...]] = None
