#!/usr/bin/env python
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any

import os
import warnings
import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F
from fla.modules import FusedRMSNormGated
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate
from safetensors import safe_open

from gen_distill.models.layers.short_convolution import BoundaryAwareShortConvolution

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class EfficientKDA(nn.Module):
    """
    Kimi Delta Attention (KDA) mixer with an EfficientQwen-style interface.

    Args:
        d_model (int):
            Hidden size of the input.
        d_state (int):
            Per-head key dimension (head_dim).
        n_q_heads (int):
            Number of query/key heads.
        n_k_heads (int):
            Number of key heads (for GVA/GQA-style layouts).
        n_v_heads (int):
            Number of value heads (for GVA/GQA-style layouts).
        d_conv (int):
            Kernel size of the short convolution.
        expand_q (float):
            Expansion ratio for the query dimension (per-head).
        expand_k (float):
            Expansion ratio for the key dimension (per-head).
        expand_v (float):
            Expansion ratio for the value dimension (per-head).
        mode (str):
            KDA kernel mode, `"chunk"` or `"fused_recurrent"`.
        use_short_conv (bool):
            Whether to use short convolutions for Q, K, V.
        allow_neg_eigval (bool):
            Whether to allow negative eigenvalues in the KDA gate.
        conv_bias (bool):
            Whether to use bias in the short convolution.
        layer_idx (int):
            Layer index used for cache addressing.
        rms_norm_eps (float):
            Epsilon for the output gated RMSNorm.
        activation (str):
            Activation for the short convolution branches.
        use_qk_norm (bool):
            Whether to apply RMSNorm to Q and K projections.
        use_gqa_setup (bool):
            Whether to use Grouped Query Attention setup (Q > K=V).
        use_gva_setup (bool):
            Whether to use Grouped Value Attention setup (V > Q=K).
        use_post_ssm_norm (bool):
            Whether to use post-SSM normalization.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        n_q_heads: int = 16,
        n_k_heads: int = 8,
        n_v_heads: int = 8,
        d_conv: int = 4,
        expand_q: float = 1.0,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_bias: bool = True,
        layer_idx: Optional[int] = None,
        rms_norm_eps: float = 1e-5,
        activation: str = "identity",
        out_gate_activation: str = "sigmoid",
        use_qk_norm: bool = False,
        use_gqa_setup: bool = True,
        use_gva_setup: bool = False,
        use_post_ssm_norm: bool = True,
        **kwargs,
    ) -> "EfficientKDA":
        super().__init__()

        self.layer_idx = layer_idx
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.use_short_conv = use_short_conv
        self.conv_size = d_conv
        self.conv_bias = conv_bias
        self.use_gqa_setup = use_gqa_setup
        self.use_gva_setup = use_gva_setup
        self.use_qk_norm = use_qk_norm
        self.n_q_heads = n_q_heads
        self.n_k_heads = n_k_heads
        self.n_v_heads = n_v_heads
        self.expand_v = expand_v
        self.expand_k = expand_k
        self.expand_q = expand_q
        self.head_k_dim = int(self.d_state * self.expand_k)
        self.head_q_dim = int(self.d_state * self.expand_q)
        self.head_v_dim = int(self.d_state * self.expand_v)
        self.key_dim = int(self.n_k_heads * self.head_k_dim)
        self.query_dim = int(self.n_q_heads * self.head_q_dim)
        self.value_dim = int(self.n_v_heads * self.head_v_dim)

        self.activation = activation
        self.out_gate_activation = out_gate_activation
        self.use_post_ssm_norm = use_post_ssm_norm
        if self.out_gate_activation not in ["sigmoid", "silu"]:
            raise ValueError(f"Unsupported output gate activation: {self.out_gate_activation}")

        if self.use_gqa_setup:
            assert self.n_q_heads % self.n_k_heads == 0
            assert self.n_q_heads % self.n_v_heads == 0
            self.out_dim = self.query_dim
        elif self.use_gva_setup:
            assert self.n_v_heads % self.n_q_heads == 0
            assert self.n_v_heads % self.n_k_heads == 0
            self.out_dim = self.value_dim

        # Internal KDA config
        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = d_model
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.query_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # Optional short convolutions
        if self.use_short_conv:
            self.q_conv1d = BoundaryAwareShortConvolution(
                hidden_size=self.query_dim,
                kernel_size=self.conv_size,
                bias=self.conv_bias,
                activation=activation,
            )
            self.k_conv1d = BoundaryAwareShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_size,
                bias=self.conv_bias,
                activation=activation,
            )
            self.v_conv1d = BoundaryAwareShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=self.conv_size,
                bias=self.conv_bias,
                activation=activation,
            )

        # KDA gate parameters (similar role to dt/A_log in Mamba)
        # Gate is in Q space (controls updates for the state, which is driven by QK^T)
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.n_q_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.A_log._no_weight_decay = True

        # dt_bias maps to Q space
        self.dt_bias = nn.Parameter(torch.zeros(self.query_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.query_dim, bias=False),
        )
        self.b_proj = nn.Linear(self.hidden_size, self.n_q_heads, bias=False)

        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.out_dim, bias=True),
        )
        if self.use_post_ssm_norm:
            self.o_norm = FusedRMSNormGated(
                self.head_v_dim, activation=self.out_gate_activation, eps=rms_norm_eps
            )
        self.out_proj = nn.Linear(self.out_dim, self.hidden_size, bias=False)

        # Track one-time attention mask warning for non-2D masks
        self._attention_mask_warned_shape = False
        self._mixer_matrix_not_supported_warned = False
        if not self.use_qk_norm:
            warnings.warn(
                "use_qk_norm is set to False, this will likely lead to NaNs in the KDA kernel based on our experiments.",
                RuntimeWarning,
            )

    def _init_custom_weights(self):
        """
        Re-initializes the weights of the Mamba2 module. This is particularly
        useful when the model is loaded with `device_map="auto"`, as the default
        initialization in `__init__` is skipped for modules on meta device.
        """
        device = self.out_proj.weight.device

        # Identity init of conv1d
        if self.use_short_conv:
            self.q_conv1d.weight.data.zero_()
            if self.conv_bias:
                self.q_conv1d.bias.data.zero_()
            self.q_conv1d.weight.data[:, :, -1] = 1.0
            self.k_conv1d.weight.data.zero_()
            if self.conv_bias:
                self.k_conv1d.bias.data.zero_()
            self.k_conv1d.weight.data[:, :, -1] = 1.0
            self.v_conv1d.weight.data.zero_()
            if self.conv_bias:
                self.v_conv1d.bias.data.zero_()
            self.v_conv1d.weight.data[:, :, -1] = 1.0

        # # Re-run the dt_bias and the A_log initialization logic
        # # NOTE (juan): On the GGZ server, the OG init (commented out) performed much better than
        # # the custom init
        # dt_bias_init = torch.zeros(self.query_dim, dtype=torch.float32, device=device)
        # self.dt_bias.data.copy_(dt_bias_init)

        # A_log_init = torch.log(
        #     torch.empty(self.n_q_heads, dtype=torch.float32, device=device).uniform_(1, 16)
        # )  # A_log is initialized between[0, ~2.77]
        # self.A_log.data.copy_(A_log_init)

        # NOTE (juan): The custom init below performed much better on hte YZ server, but much worse
        # on the GZ server
        # Use the same init as Mamba for the dt_init and the A_log!
        dt_max = 1e-1
        dt_min = 1e-3
        dt_init_floor = 1e-4
        A_init_range = [1, 16]
        dt = torch.exp(
            torch.rand(self.query_dim, dtype=torch.float32, device=device)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)

        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter: Q heads, because it is called after the GQA expansion
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.n_q_heads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=torch.float32)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # # Identity init of g_proj[1] --> zeros on projection, and 1.278 on bias
        if self.out_gate_activation == "silu":
            # We need to set the z = gate_proj(hidden_states) to be 1.278 since F.silu(1.278) = 1
            self.g_proj[1].weight.data.zero_()
            self.g_proj[1].bias.data.fill_(1.278)
        elif self.out_gate_activation == "sigmoid":
            # We need to set the z = gate_proj(hidden_states) to be ??, since F.sigmoid(0) = 1
            # not possible, only z -> inf satisfied the condition. Instead we will set it to
            # a constant value --> linearly rescales the input, which is fine I guess -->
            # this can be compensated by the output projection
            self.g_proj[1].weight.data.zero_()
            self.g_proj[1].bias.data.zero_()  # this gives 1/2 for the gate
        else:
            raise ValueError(f"Unsupported output gate projection: {self.out_gate_activation}")

        # Initialize the rms_norm weights if present
        if self.use_post_ssm_norm:
            self.o_norm.weight.data.fill_(1.0)

        # NOTE: Out projection will use the default initialization (will be called by the
        # _init_weights_) from the parent model. Same for the post SSM normalization

    def init_mixer_proj_from_qkvo(self, checkpoint_file_dir: str):
        """
        Initialize KDA projections from a teacher Transformer's Q, K, V, and O projections.

        Teacher tensors follow the pattern:
            `model.layers.{layer_idx}.self_attn.{q,k,v,o}_proj.{weight|bias}`
        """
        if self.layer_idx is None:
            raise ValueError("EfficientKDA.init_mixer_from_qkvo requires a valid layer_idx.")

        model_path = os.path.join(checkpoint_file_dir, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model.safetensors file found in {checkpoint_file_dir}")

        checkpoint = safe_open(model_path, framework="pt")

        # Load and copy Q, K, V projections
        for param_name in ["q_proj", "k_proj", "v_proj"]:
            weight_name = f"model.layers.{self.layer_idx}.self_attn.{param_name}.weight"
            bias_name = f"model.layers.{self.layer_idx}.self_attn.{param_name}.bias"

            if weight_name not in checkpoint.keys():
                raise KeyError(f"Missing tensor `{weight_name}` in teacher checkpoint.")

            weight = checkpoint.get_tensor(weight_name)
            bias = checkpoint.get_tensor(bias_name) if bias_name in checkpoint.keys() else None

            proj = getattr(self, param_name)
            proj.weight.data.copy_(weight)
            if bias is not None and proj.bias is not None:
                proj.bias.data.copy_(bias)

        # Load and copy O projection
        o_weight_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.weight"
        o_bias_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.bias"

        if o_weight_name not in checkpoint.keys():
            raise KeyError(f"Missing tensor `{o_weight_name}` in teacher checkpoint.")

        o_weight = checkpoint.get_tensor(o_weight_name)
        self.out_proj.weight.data.copy_(o_weight)
        if o_bias_name in checkpoint.keys() and self.out_proj.bias is not None:
            o_bias = checkpoint.get_tensor(o_bias_name)
            self.out_proj.bias.data.copy_(o_bias)

    def init_mixer_proj_from_vo(self, checkpoint_file_dir: str):
        """
        Initialize KDA projections from a teacher Transformer's Value and Output projections only.
        """
        if self.layer_idx is None:
            raise ValueError("EfficientKDA.init_mixer_from_vo requires a valid layer_idx.")

        model_path = os.path.join(checkpoint_file_dir, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model.safetensors file found in {checkpoint_file_dir}")

        checkpoint = safe_open(model_path, framework="pt")

        # V projection
        v_weight_name = f"model.layers.{self.layer_idx}.self_attn.v_proj.weight"
        v_bias_name = f"model.layers.{self.layer_idx}.self_attn.v_proj.bias"
        if v_weight_name not in checkpoint.keys():
            raise KeyError(f"Missing tensor `{v_weight_name}` in teacher checkpoint.")

        v_weight = checkpoint.get_tensor(v_weight_name)
        v_bias = checkpoint.get_tensor(v_bias_name) if v_bias_name in checkpoint.keys() else None

        self.v_proj.weight.data.copy_(v_weight)
        if v_bias is not None and self.v_proj.bias is not None:
            self.v_proj.bias.data.copy_(v_bias)

        # O projection
        o_weight_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.weight"
        o_bias_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.bias"

        if o_weight_name not in checkpoint.keys():
            raise KeyError(f"Missing tensor `{o_weight_name}` in teacher checkpoint.")

        o_weight = checkpoint.get_tensor(o_weight_name)
        self.out_proj.weight.data.copy_(o_weight)
        if o_bias_name in checkpoint.keys() and self.out_proj.bias is not None:
            o_bias = checkpoint.get_tensor(o_bias_name)
            self.out_proj.bias.data.copy_(o_bias)

    def forward(
        self,
        u: torch.Tensor,
        output_mixer_matrix: bool = False,
        cache_params: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_boundary: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        layer_state: Optional[Dict[str, Any]] = None,
        **kwargs: "Unpack[dict]",
    ) -> Dict[str, torch.Tensor]:
        """
        EfficientQwen mixer-style forward.

        Args:
            u: Input tensor of shape (B, L, D).
            output_mixer_matrix: Not supported for KDA (raises if True).
            cache_params: Optional `EfficientQwenCache` instance.
            attention_mask: Optional 2D padding mask (B, L). Non-2D masks are
                ignored (same behavior as `Mamba2`).
            document_boundary: Optional (B, L) mask with 1 at document starts.
            layer_state: Optional dictionary containing KDA state (for explicit state passing).
        """
        if output_mixer_matrix and not self._mixer_matrix_not_supported_warned:
            warnings.warn(
                "EfficientKDA does not support mixer_matrix materialization. We ignore the output_mixer_matrix argument.",
                RuntimeWarning,
            )
            self._mixer_matrix_not_supported_warned = True

        hidden_states = u
        batch_size, seq_len, _ = hidden_states.shape

        # if torch.isnan(hidden_states).any():
        #     warnings.warn(f"NaN detected in Hidden States at layer {self.layer_idx}")

        # Change to inference mode: prefer fused_recurrent for short sequences at eval
        mode = "fused_recurrent" if (not self.training and seq_len == 1) else "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        # Attention-mask handling
        # NOTE: During generation (seq_len == 1), we should NOT derive document_boundary from
        # attention_mask because attention_mask has the full sequence length while we're only
        # processing a single token. The document boundary logic is only relevant during prefill.
        if attention_mask is not None and seq_len > 1:
            if attention_mask.dim() != 2:
                if not self._attention_mask_warned_shape:
                    warnings.warn(
                        "EfficientKDA: received attention_mask not of shape (B, L); "
                        "ignoring it for KDA computation.",
                        RuntimeWarning,
                    )
                    self._attention_mask_warned_shape = True
            else:
                attn_bool = attention_mask.to(dtype=torch.bool)
                prev = F.pad(attn_bool[:, :-1], (1, 0), value=False)
                boundary_from_mask = attn_bool & (~prev)
                if document_boundary is None:
                    document_boundary = boundary_from_mask
                else:
                    document_boundary = document_boundary.to(dtype=torch.bool) | boundary_from_mask

        # During generation steps, ignore document_boundary if it doesn't match the input sequence length
        if document_boundary is not None and document_boundary.shape[-1] != seq_len:
            document_boundary = None

        # Determine cache usage
        use_cache_flag = (
            use_cache
            if use_cache is not None
            else (cache_params is not None or layer_state is not None)
        )
        last_state = None
        if cache_params is not None:
            last_state = cache_params.get_kda_state(self.layer_idx)
        elif layer_state is not None:
            last_state = layer_state

        # Short convolution branches
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q = last_state.get("conv_state_q", None)
                conv_state_k = last_state.get("conv_state_k", None)
                conv_state_v = last_state.get("conv_state_v", None)

            q_proj_out = self.q_proj(hidden_states)
            q, conv_state_q = self.q_conv1d(
                x=q_proj_out,
                cache=conv_state_q,
                output_final_state=use_cache_flag,
                # cu_seqlens=cu_seqlens,
                document_boundary=document_boundary,
            )

            k_proj_out = self.k_proj(hidden_states)
            k, conv_state_k = self.k_conv1d(
                x=k_proj_out,
                cache=conv_state_k,
                output_final_state=use_cache_flag,
                document_boundary=document_boundary,
            )

            v_proj_out = self.v_proj(hidden_states)
            v, conv_state_v = self.v_conv1d(
                x=v_proj_out,
                cache=conv_state_v,
                output_final_state=use_cache_flag,
                document_boundary=document_boundary,
            )
        else:
            q = self.q_proj(hidden_states)  # we remove the SiLU activation to align with Mamba
            k = self.k_proj(hidden_states)  # we remove the SiLU activation to align with Mamba
            v = self.v_proj(hidden_states)  # we remove the SiLU activation to align with Mamba
            conv_state_q = conv_state_k = conv_state_v = None

        # g is the decay params, beta is the rewriting params
        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states).float().sigmoid()

        # Rearrange QKV values
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_q_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_q_dim)

        # Handle GQA/GVA setups
        if self.use_gva_setup:
            if self.n_v_heads > self.n_q_heads:
                q, k, g = (
                    repeat(x, "... h d -> ... (h g) d", g=self.n_v_heads // self.n_q_heads)
                    for x in (q, k, g)
                )
                beta = repeat(beta, "... h -> ... (h g)", g=self.n_v_heads // self.n_q_heads)
        elif self.use_gqa_setup:
            if self.n_q_heads > self.n_k_heads:
                k = repeat(k, "... h d -> ... (h g) d", g=self.n_q_heads // self.n_k_heads)
                v = repeat(v, "... h d -> ... (h g) d", g=self.n_q_heads // self.n_v_heads)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Reset the hidden state at document boundaries --> forget gate should be very large
        # negative value (i.e sicne it is in log space)
        if document_boundary is not None:
            boundary_mask = document_boundary.to(dtype=torch.bool).unsqueeze(-1).unsqueeze(-1)
            g = g.masked_fill(
                boundary_mask, g.new_tensor(0)
            )  # TODO: For the GZ updated version, this should be 0!

        # if self.training:
        #     if torch.isnan(q).any():
        #         warnings.warn(f"NaN detected in Q at layer {self.layer_idx}")
        #     if torch.isnan(k).any():
        #         warnings.warn(f"NaN detected in K at layer {self.layer_idx}")
        #     if torch.isnan(v).any():
        #         warnings.warn(f"NaN detected in V at layer {self.layer_idx}")
        #     if torch.isnan(g).any():
        #         warnings.warn(f"NaN detected in G at layer {self.layer_idx}")
        #     if torch.isnan(beta).any():
        #         warnings.warn(f"NaN detected in Beta at layer {self.layer_idx}")

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache_flag,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                # cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            g = fused_kda_gate(g=g, A_log=self.A_log, dt_bias=self.dt_bias)
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache_flag,
                use_qk_l2norm_in_kernel=True,
                # cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Update EfficientQwenCache in-place
        if cache_params is not None and hasattr(cache_params, "update_kda_state"):
            cache_params.update_kda_state(
                layer_idx=self.layer_idx,
                recurrent_state=recurrent_state,
                conv_state_q=conv_state_q if self.use_short_conv else None,
                conv_state_k=conv_state_k if self.use_short_conv else None,
                conv_state_v=conv_state_v if self.use_short_conv else None,
                offset=seq_len,
            )
        elif layer_state is not None:
            layer_state["recurrent_state"] = recurrent_state
            if self.use_short_conv:
                layer_state["conv_state_q"] = conv_state_q
                layer_state["conv_state_k"] = conv_state_k
                layer_state["conv_state_v"] = conv_state_v

        # Output projection with gated RMSNorm
        # NOTE: Mamba has gating --> norm --> output proj (gate and norm are over the full features)
        # KDA has norm --> gating --> output_proj (also the gate and norm are per head)
        gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        if self.use_post_ssm_norm:
            o = self.o_norm(o.contiguous(), gate.contiguous())
        else:
            if self.out_gate_activation == "sigmoid":
                o = o * torch.sigmoid(gate)
            elif self.out_gate_activation == "silu":
                o = o * F.silu(gate)
            else:
                raise ValueError(f"Unsupported output gate activation: {self.out_gate_activation}")

        # if torch.isnan(o).any():
        #     warnings.warn(f"NaN detected in O at layer {self.layer_idx}")

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.out_proj(o)
        return {"output_states": o[:, :seq_len, :], "mixer_matrix": None}

    def step(
        self,
        u: torch.Tensor,
        state: Optional[Dict[str, Any]] = None,
        cache_params: Optional[Any] = None,
        document_boundary: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Single-token generation step.

        This mirrors the `Mamba2.step` signature but internally reuses the
        full-sequence KDA forward on a length-1 sequence while relying on
        `EfficientQwenCache` for state persistence.
        """
        if u.dim() != 2:
            raise ValueError(f"EfficientKDA.step expected input of shape (B, D), got {u.shape}")

        token_seq = u.unsqueeze(1)  # (B, 1, D)
        outputs = self.forward(
            token_seq,
            output_mixer_matrix=False,
            cache_params=cache_params,
            attention_mask=None,
            document_boundary=document_boundary,
            use_cache=True,
            layer_state=state,
        )
        return outputs["output_states"].squeeze(1), state

    def allocate_inference_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Allocate a minimal cache structure compatible with EfficientKDA.step.

        For KDA, the underlying kernels can initialize their own recurrent state
        from zeros when no state is provided, so we only allocate convolution
        caches matching the BoundaryAwareShortConvolution layout.
        """
        device = self.q_proj.weight.device
        conv_dtype = self.q_proj.weight.dtype if dtype is None else dtype

        conv_state_q = torch.zeros(
            batch_size,
            self.query_dim,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        conv_state_k = torch.zeros(
            batch_size,
            self.key_dim,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        conv_state_v = torch.zeros(
            batch_size,
            self.value_dim,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )

        # KDA recurrent state (analogous to Mamba2.ssm_state), required to be float32.
        # Shape follows `chunk_kda` docs: [N, H, K, V], where
        #   N = batch_size (number of sequences),
        #   H = num_v_heads,
        #   K = head_k_dim,
        #   V = head_v_dim.

        # If GQA, we effectively run n_q_heads parallel systems.
        # If GVA, we effectively run n_v_heads parallel systems (with shared Q/K params).
        # So H should be max(n_q, n_v)?
        num_effective_heads = self.n_q_heads  # GQA setup
        if self.use_gva_setup:
            num_effective_heads = self.n_v_heads

        recurrent_state = torch.zeros(
            batch_size,
            num_effective_heads,
            self.head_k_dim,
            self.head_v_dim,
            device=device,
            dtype=torch.float32,
        )

        return {
            "recurrent_state": recurrent_state,
            "conv_state_q": conv_state_q,
            "conv_state_k": conv_state_k,
            "conv_state_v": conv_state_v,
        }
