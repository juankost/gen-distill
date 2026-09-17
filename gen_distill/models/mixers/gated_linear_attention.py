#!/usr/bin/env python
# GLA mixer with EfficientQwen-style interface.
# Follows the published GLA design (Yang et al. 2024): K-space low-rank forget
# gate shared across heads-in-a-group, no delta rule. Structure mirrors
# EfficientKDA so that the rest of the hybrid plumbing (cache, projections,
# teacher init, gated output norm) stays identical.

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any

import os
import warnings

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F
from fla.modules import FusedRMSNormGated
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from safetensors import safe_open

from gen_distill.models.layers.short_convolution import BoundaryAwareShortConvolution

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class EfficientGLA(nn.Module):
    """
    Gated Linear Attention (GLA) mixer with an EfficientQwen-style interface.

    Mirrors the structure of EfficientKDA, but replaces the delta-rule update
    and the KDA-specific gate (A_log, dt_bias, beta, f_proj) with the standard
    GLA forget-gate parameterization: a low-rank K-space projection
    `gk_proj: hidden -> n_k_heads * head_k_dim`, followed by
    `gk = logsigmoid(gk) / gate_logit_normalizer`. The gate is shared across
    heads-in-a-group (repeated alongside `k` during GQA expansion), matching
    the design in Yang et al. 2024 and the reference `fla.layers.GatedLinearAttention`.
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
        conv_bias: bool = True,
        layer_idx: Optional[int] = None,
        rms_norm_eps: float = 1e-5,
        activation: str = "identity",
        out_gate_activation: str = "sigmoid",
        use_qk_norm: bool = False,
        use_gqa_setup: bool = True,
        use_gva_setup: bool = False,
        use_post_ssm_norm: bool = True,
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        **kwargs,
    ):
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
            raise ValueError(
                f"Unsupported output gate activation: {self.out_gate_activation}"
            )

        if self.use_gqa_setup:
            assert self.n_q_heads % self.n_k_heads == 0
            assert self.n_q_heads % self.n_v_heads == 0
            self.out_dim = self.query_dim
        elif self.use_gva_setup:
            assert self.n_v_heads % self.n_q_heads == 0
            assert self.n_v_heads % self.n_k_heads == 0
            self.out_dim = self.value_dim

        self.mode = mode
        self.hidden_size = d_model
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        # Q/K/V projections (sized to match the teacher Qwen3 attention)
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

        # Low-rank K-space forget-gate projection. Sized to n_k_heads * head_k_dim;
        # repeated alongside k during the GQA expansion so the gate is shared
        # across heads-in-a-group (standard GLA design).
        self.gate_low_rank_dim = gate_low_rank_dim
        self.gate_logit_normalizer = gate_logit_normalizer
        self.gk_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.gate_low_rank_dim, bias=False),
            nn.Linear(self.gate_low_rank_dim, self.key_dim, bias=True),
        )

        # Output gate (same parameterization as KDA: low-rank through head_v_dim)
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.out_dim, bias=True),
        )
        if self.use_post_ssm_norm:
            self.o_norm = FusedRMSNormGated(
                self.head_v_dim, activation=self.out_gate_activation, eps=rms_norm_eps
            )
        self.out_proj = nn.Linear(self.out_dim, self.hidden_size, bias=False)

        self._attention_mask_warned_shape = False
        self._mixer_matrix_not_supported_warned = False

    def _init_custom_weights(self):
        """Re-initialize after `device_map='auto'` materializes the module."""
        if self.use_short_conv:
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
                conv.weight.data.zero_()
                if self.conv_bias:
                    conv.bias.data.zero_()
                conv.weight.data[:, :, -1] = 1.0

        # Forget gate: zero-init weight and bias so initial
        #   gk = logsigmoid(0) / normalizer = -log(2) / normalizer
        # which is ~ -0.043 for normalizer=16 -> exp(gk) ~ 0.957: the state is
        # preserved almost perfectly at init, which is what we want for KD warm-up.
        self.gk_proj[0].weight.data.normal_(mean=0.0, std=0.02)
        self.gk_proj[1].weight.data.zero_()
        self.gk_proj[1].bias.data.zero_()

        # Output gate: same init as KDA (zero weight, bias chosen so that the
        # post-norm gating multiplier is ~1 at init).
        if self.out_gate_activation == "silu":
            self.g_proj[1].weight.data.zero_()
            self.g_proj[1].bias.data.fill_(1.278)  # silu(1.278) ~= 1
        elif self.out_gate_activation == "sigmoid":
            self.g_proj[1].weight.data.zero_()
            self.g_proj[1].bias.data.zero_()  # sigmoid(0) = 0.5
        else:
            raise ValueError(
                f"Unsupported output gate projection: {self.out_gate_activation}"
            )

        if self.use_post_ssm_norm:
            self.o_norm.weight.data.fill_(1.0)

    def init_mixer_proj_from_qkvo(self, checkpoint_file_dir: str):
        """Initialize GLA projections from a teacher Transformer's Q, K, V, O."""
        if self.layer_idx is None:
            raise ValueError(
                "EfficientGLA.init_mixer_proj_from_qkvo requires a valid layer_idx."
            )

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

    def init_mixer_proj_from_vo(self, checkpoint_file_dir: str):
        """Initialize GLA projections from a teacher Transformer's V and O only."""
        if self.layer_idx is None:
            raise ValueError(
                "EfficientGLA.init_mixer_proj_from_vo requires a valid layer_idx."
            )

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
                "EfficientGLA does not support mixer_matrix materialization. Ignoring.",
                RuntimeWarning,
            )
            self._mixer_matrix_not_supported_warned = True

        hidden_states = u
        batch_size, seq_len, _ = hidden_states.shape

        mode = "fused_recurrent" if (not self.training and seq_len == 1) else "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        # Derive document_boundary from 2D attention_mask if provided (prefill only)
        if attention_mask is not None and seq_len > 1:
            if attention_mask.dim() != 2:
                if not self._attention_mask_warned_shape:
                    warnings.warn(
                        "EfficientGLA: received attention_mask not of shape (B, L); ignoring it.",
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
            document_boundary = None

        use_cache_flag = (
            use_cache
            if use_cache is not None
            else (cache_params is not None or layer_state is not None)
        )
        last_state = None
        if cache_params is not None:
            last_state = cache_params.get_gla_state(self.layer_idx)
        elif layer_state is not None:
            last_state = layer_state

        # Short conv on Q/K/V (gate is not conv'd, matching fla.layers.GatedLinearAttention)
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
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

        # K-space forget gate (data-dependent, low-rank, shared across heads-in-a-group)
        gk = self.gk_proj(hidden_states)  # (B, L, key_dim)

        # Rearrange to per-head layout
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_q_dim)  # (B, L, n_q, K)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)  # (B, L, n_k, K)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)  # (B, L, n_v, V)
        gk = rearrange(gk, "... (h d) -> ... h d", d=self.head_k_dim)  # (B, L, n_k, K)

        # GQA / GVA expansion: bring q, k, v, gk to a common head count.
        # gk follows k (both are K-space and start at n_k_heads).
        if self.use_gqa_setup:
            if self.n_q_heads > self.n_k_heads:
                k = repeat(
                    k, "... h d -> ... (h g) d", g=self.n_q_heads // self.n_k_heads
                )
                gk = repeat(
                    gk, "... h d -> ... (h g) d", g=self.n_q_heads // self.n_k_heads
                )
            if self.n_q_heads > self.n_v_heads:
                v = repeat(
                    v, "... h d -> ... (h g) d", g=self.n_q_heads // self.n_v_heads
                )
        elif self.use_gva_setup:
            if self.n_v_heads > self.n_q_heads:
                q = repeat(
                    q, "... h d -> ... (h g) d", g=self.n_v_heads // self.n_q_heads
                )
            if self.n_v_heads > self.n_k_heads:
                k = repeat(
                    k, "... h d -> ... (h g) d", g=self.n_v_heads // self.n_k_heads
                )
                gk = repeat(
                    gk, "... h d -> ... (h g) d", g=self.n_v_heads // self.n_k_heads
                )

        # Standard GLA gate transform: logsigmoid -> normalize. Values are in
        # (-inf, 0]; exp(gk) in (0, 1] sets the per-key decay.
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        # Document-boundary state reset: push gk very negative so exp(gk) ~ 0
        # at the first token of each new document, wiping the carried state.
        if document_boundary is not None:
            boundary_mask = (
                document_boundary.to(dtype=torch.bool).unsqueeze(-1).unsqueeze(-1)
            )
            gk = gk.masked_fill(boundary_mask, gk.new_tensor(-20.0))

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        if mode == "chunk":
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache_flag,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache_flag,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if cache_params is not None and hasattr(cache_params, "update_gla_state"):
            cache_params.update_gla_state(
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

        # Output gating (same shape contract as KDA: norm -> gate -> out_proj)
        gate = rearrange(
            self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
        )
        if self.use_post_ssm_norm:
            o = self.o_norm(o.contiguous(), gate.contiguous())
        else:
            if self.out_gate_activation == "sigmoid":
                o = o * torch.sigmoid(gate)
            elif self.out_gate_activation == "silu":
                o = o * F.silu(gate)
            else:
                raise ValueError(
                    f"Unsupported output gate activation: {self.out_gate_activation}"
                )

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
        if u.dim() != 2:
            raise ValueError(
                f"EfficientGLA.step expected input of shape (B, D), got {u.shape}"
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

        # GLA recurrent state shape (default state_v_first=False layout): [N, H, K, V]
        num_effective_heads = self.n_q_heads
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
