#!/usr/bin/env python
# Lightning Attention mixer with the EfficientQwen interface.
#
# Lightning Attention (Qin et al., 2024a, "Various Lengths, Constant Speed")
# is a linear-attention variant with a *data-independent* per-head log-decay
# `gamma_h = -(8 / H) * (1 - layer_idx / num_layers) * h` for head index h.
# The recurrence is otherwise identical to simple GLA:
#     S[t] = exp(g[t]) * S[t-1] + K[t]^T V[t]
# We use `fla.ops.simple_gla` (which `chunk_lightning_attn` delegates to) so
# that we can pass a per-token `g` tensor and force document-boundary state
# resets in the same way as EfficientKDA / EfficientGatedDeltaNet.

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any

import os
import warnings
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F
from fla.modules import FusedRMSNormGated, RMSNorm
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
from safetensors import safe_open


from gen_distill.models.layers.short_convolution import BoundaryAwareShortConvolution

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


# Boundary reset: log-space decay value at document boundaries.
# exp(-80) ~= 1.8e-35, effectively zero — full state reset on the boundary token.
_BOUNDARY_LOG_DECAY = -80.0


class EfficientLightningAttention(nn.Module):
    """
    Lightning Attention mixer.

    Closed-form per-head log-decay slope:
        gamma_h = -(8 / H) * (1 - layer_idx / num_layers) * h

    Args mirror EfficientKDA where possible so that hybrids that swap mixer
    types do not need to re-tune the head/dim setup.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
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
    ) -> "EfficientLightningAttention":
        super().__init__()

        if layer_idx is None:
            raise ValueError("EfficientLightningAttention requires a valid layer_idx.")
        if num_layers is None or num_layers <= 0:
            raise ValueError(
                f"EfficientLightningAttention requires num_layers > 0, got {num_layers}."
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.d_model = d_model
        self.hidden_size = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_size = d_conv
        self.use_short_conv = use_short_conv
        self.conv_bias = conv_bias
        self.use_qk_norm = use_qk_norm
        self.use_gqa_setup = use_gqa_setup
        self.use_gva_setup = use_gva_setup
        self.use_post_ssm_norm = use_post_ssm_norm

        self.n_q_heads = n_q_heads
        self.n_k_heads = n_k_heads
        self.n_v_heads = n_v_heads
        self.expand_q = expand_q
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.head_q_dim = int(self.d_state * self.expand_q)
        self.head_k_dim = int(self.d_state * self.expand_k)
        self.head_v_dim = int(self.d_state * self.expand_v)
        self.query_dim = int(self.n_q_heads * self.head_q_dim)
        self.key_dim = int(self.n_k_heads * self.head_k_dim)
        self.value_dim = int(self.n_v_heads * self.head_v_dim)

        self.activation = activation
        self.out_gate_activation = out_gate_activation
        if self.out_gate_activation not in ["sigmoid", "silu"]:
            raise ValueError(
                f"Unsupported output gate activation: {self.out_gate_activation}"
            )

        if self.use_gqa_setup:
            assert self.n_q_heads % self.n_k_heads == 0
            assert self.n_q_heads % self.n_v_heads == 0
            self.out_dim = self.query_dim
            self.n_effective_heads = self.n_q_heads
        elif self.use_gva_setup:
            assert self.n_v_heads % self.n_q_heads == 0
            assert self.n_v_heads % self.n_k_heads == 0
            self.out_dim = self.value_dim
            self.n_effective_heads = self.n_v_heads
        else:
            self.out_dim = self.query_dim
            self.n_effective_heads = self.n_q_heads

        self.mode = mode

        # Projections (no f_proj / b_proj / A_log / dt_bias — Lightning has no learnable gates).
        self.q_proj = nn.Linear(self.hidden_size, self.query_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

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

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_q_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_k_dim, eps=rms_norm_eps)

        # Output gate (kept structurally identical to EfficientKDA).
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.out_dim, bias=True),
        )
        if self.use_post_ssm_norm:
            self.o_norm = FusedRMSNormGated(
                self.head_v_dim, activation=self.out_gate_activation, eps=rms_norm_eps
            )
        self.out_proj = nn.Linear(self.out_dim, self.hidden_size, bias=False)

        # Closed-form per-head log-decay slope (data-independent, fixed at init).
        # Shape [n_effective_heads]; we broadcast to [B, T, H] at forward time.
        slopes = (
            -(8.0 / self.n_effective_heads)
            * (1.0 - self.layer_idx / self.num_layers)
            * torch.arange(self.n_effective_heads, dtype=torch.float32)
        )
        self.register_buffer("lightning_slopes", slopes, persistent=False)

        self._attention_mask_warned_shape = False
        self._mixer_matrix_not_supported_warned = False

    def _init_custom_weights(self):
        """
        Re-init weights for modules created on meta device.
        Mirrors EfficientKDA._init_custom_weights for short-conv / norms / output gate.
        """
        if self.use_short_conv:
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
                conv.weight.data.zero_()
                if self.conv_bias:
                    conv.bias.data.zero_()
                conv.weight.data[:, :, -1] = 1.0  # identity init

        if self.use_qk_norm:
            self.q_norm.weight.data.fill_(1.0)
            self.k_norm.weight.data.fill_(1.0)

        # Output gate: match KDA's identity-ish init so the gate ~= 1 at start.
        if self.out_gate_activation == "silu":
            self.g_proj[1].weight.data.zero_()
            self.g_proj[1].bias.data.fill_(1.278)  # F.silu(1.278) ~= 1
        else:  # sigmoid
            self.g_proj[1].weight.data.zero_()
            self.g_proj[1].bias.data.zero_()  # sigmoid(0) = 0.5

        if self.use_post_ssm_norm:
            self.o_norm.weight.data.fill_(1.0)

    def init_mixer_proj_from_qkvo(self, checkpoint_file_dir: str):
        """Initialize Q/K/V/O projections from the teacher transformer."""
        model_path = os.path.join(checkpoint_file_dir, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model.safetensors file found in {checkpoint_file_dir}")

        checkpoint = safe_open(model_path, framework="pt")

        for param_name in ["q_proj", "k_proj", "v_proj"]:
            weight_name = f"model.layers.{self.layer_idx}.self_attn.{param_name}.weight"
            bias_name = f"model.layers.{self.layer_idx}.self_attn.{param_name}.bias"
            if weight_name not in checkpoint.keys():
                raise KeyError(f"Missing tensor `{weight_name}` in teacher checkpoint.")
            weight = checkpoint.get_tensor(weight_name)
            bias = (
                checkpoint.get_tensor(bias_name)
                if bias_name in checkpoint.keys()
                else None
            )
            proj = getattr(self, param_name)
            proj.weight.data.copy_(weight)
            if bias is not None and proj.bias is not None:
                proj.bias.data.copy_(bias)

        o_weight_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.weight"
        o_bias_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.bias"
        if o_weight_name not in checkpoint.keys():
            raise KeyError(f"Missing tensor `{o_weight_name}` in teacher checkpoint.")
        self.out_proj.weight.data.copy_(checkpoint.get_tensor(o_weight_name))
        if o_bias_name in checkpoint.keys() and self.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(checkpoint.get_tensor(o_bias_name))

        if self.use_qk_norm:
            q_norm_name = f"model.layers.{self.layer_idx}.self_attn.q_norm.weight"
            k_norm_name = f"model.layers.{self.layer_idx}.self_attn.k_norm.weight"
            if q_norm_name in checkpoint.keys():
                self.q_norm.weight.data.copy_(checkpoint.get_tensor(q_norm_name))
            if k_norm_name in checkpoint.keys():
                self.k_norm.weight.data.copy_(checkpoint.get_tensor(k_norm_name))

    def init_mixer_proj_from_vo(self, checkpoint_file_dir: str):
        """Initialize V and O projections only from the teacher transformer."""
        model_path = os.path.join(checkpoint_file_dir, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model.safetensors file found in {checkpoint_file_dir}")

        checkpoint = safe_open(model_path, framework="pt")

        v_weight_name = f"model.layers.{self.layer_idx}.self_attn.v_proj.weight"
        v_bias_name = f"model.layers.{self.layer_idx}.self_attn.v_proj.bias"
        if v_weight_name not in checkpoint.keys():
            raise KeyError(f"Missing tensor `{v_weight_name}` in teacher checkpoint.")
        self.v_proj.weight.data.copy_(checkpoint.get_tensor(v_weight_name))
        if v_bias_name in checkpoint.keys() and self.v_proj.bias is not None:
            self.v_proj.bias.data.copy_(checkpoint.get_tensor(v_bias_name))

        o_weight_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.weight"
        o_bias_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.bias"
        if o_weight_name not in checkpoint.keys():
            raise KeyError(f"Missing tensor `{o_weight_name}` in teacher checkpoint.")
        self.out_proj.weight.data.copy_(checkpoint.get_tensor(o_weight_name))
        if o_bias_name in checkpoint.keys() and self.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(checkpoint.get_tensor(o_bias_name))

    def _build_g(
        self,
        batch_size: int,
        seq_len: int,
        document_boundary: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build the per-token log-decay tensor `g` of shape (B, T, H) by
        broadcasting the closed-form Lightning slopes and forcing a state
        reset (g = _BOUNDARY_LOG_DECAY) at every document boundary position.
        """
        slopes = self.lightning_slopes.to(device=device, dtype=dtype)
        g = slopes.view(1, 1, -1).expand(batch_size, seq_len, -1).contiguous()
        if document_boundary is not None:
            boundary = document_boundary.to(device=device, dtype=torch.bool)
            g = g.masked_fill(boundary.unsqueeze(-1), _BOUNDARY_LOG_DECAY)
        return g

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
        if output_mixer_matrix and not self._mixer_matrix_not_supported_warned:
            warnings.warn(
                "EfficientLightningAttention does not support mixer_matrix materialization. "
                "Ignoring output_mixer_matrix.",
                RuntimeWarning,
            )
            self._mixer_matrix_not_supported_warned = True

        hidden_states = u
        batch_size, seq_len, _ = hidden_states.shape

        # Use fused_recurrent for single-token generation; chunk otherwise.
        mode = "fused_recurrent" if (not self.training and seq_len == 1) else "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        # Derive doc-boundary from a 2D padding mask, mirroring EfficientKDA.
        if attention_mask is not None and seq_len > 1:
            if attention_mask.dim() != 2:
                if not self._attention_mask_warned_shape:
                    warnings.warn(
                        "EfficientLightningAttention: received attention_mask not of shape (B, L); "
                        "ignoring it.",
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
                    document_boundary = (
                        document_boundary.to(dtype=torch.bool) | boundary_from_mask
                    )

        if document_boundary is not None and document_boundary.shape[-1] != seq_len:
            document_boundary = None  # length mismatch during step-by-step gen

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

        # Projections + (optional) short-conv branches.
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None:
                conv_state_q = last_state.get("conv_state_q", None)
                conv_state_k = last_state.get("conv_state_k", None)
                conv_state_v = last_state.get("conv_state_v", None)

            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache_flag,
                document_boundary=document_boundary,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache_flag,
                document_boundary=document_boundary,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache_flag,
                document_boundary=document_boundary,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            conv_state_q = conv_state_k = conv_state_v = None

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_q_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # GQA/GVA expansion to a common head count, matching EfficientKDA.
        if self.use_gva_setup:
            if self.n_v_heads > self.n_q_heads:
                q, k = (
                    repeat(
                        x, "... h d -> ... (h g) d", g=self.n_v_heads // self.n_q_heads
                    )
                    for x in (q, k)
                )
        elif self.use_gqa_setup:
            if self.n_q_heads > self.n_k_heads:
                k = repeat(
                    k, "... h d -> ... (h g) d", g=self.n_q_heads // self.n_k_heads
                )
                v = repeat(
                    v, "... h d -> ... (h g) d", g=self.n_q_heads // self.n_v_heads
                )

        # Per-token log-decay with document-boundary reset (head dim = n_effective_heads).
        g = self._build_g(
            batch_size=batch_size,
            seq_len=seq_len,
            document_boundary=document_boundary,
            device=q.device,
            dtype=torch.float32,
        )

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        if mode == "chunk":
            o, recurrent_state = chunk_simple_gla(
                q=q,
                k=k,
                v=v,
                g=g,
                initial_state=recurrent_state,
                output_final_state=use_cache_flag,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_simple_gla(
                q=q,
                k=k,
                v=v,
                g=g,
                initial_state=recurrent_state,
                output_final_state=use_cache_flag,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Persist state.
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

        # Output gate + norm + out projection.
        gate = rearrange(
            self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
        )
        if self.use_post_ssm_norm:
            o = self.o_norm(o.contiguous(), gate.contiguous())
        else:
            if self.out_gate_activation == "sigmoid":
                o = o * torch.sigmoid(gate)
            else:
                o = o * F.silu(gate)

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
        """Single-token generation step. Same contract as EfficientKDA.step."""
        if u.dim() != 2:
            raise ValueError(
                f"EfficientLightningAttention.step expected input of shape (B, D), got {u.shape}"
            )
        token_seq = u.unsqueeze(1)
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
        """Allocate a cache dict compatible with EfficientLightningAttention.step."""
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

        recurrent_state = torch.zeros(
            batch_size,
            self.n_effective_heads,
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
