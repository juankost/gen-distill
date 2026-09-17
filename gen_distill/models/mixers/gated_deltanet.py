# -*- coding: utf-8 -*-
# Gated DeltaNet (GDN) mixer adapted to the EfficientQwen interface.
#
# Theoretical relation to KDA: GDN and KDA share the same delta-rule recurrence;
# the only difference is the decay matrix. KDA uses a diagonal decay (one scalar
# per channel per head, computed from f_proj/A_log/dt_bias). GDN uses a scalar
# decay per head (scalar * identity), computed from a_proj/A_log/dt_bias.
#
# This file mirrors `EfficientKDA` line-for-line so that the two architectures
# can be compared apples-to-apples (same projections, same GQA layout, same
# short-conv with document-boundary handling, same cache shape, same init from
# teacher QKVO). The kernel call swaps `chunk_kda` / `fused_recurrent_kda` for
# `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`.

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any

import os
import warnings
import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F
from fla.modules import FusedRMSNormGated, RMSNorm
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from safetensors import safe_open

from gen_distill.models.layers.short_convolution import BoundaryAwareShortConvolution

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class EfficientGatedDeltaNet(nn.Module):
    """
    Gated DeltaNet (GDN) mixer with an EfficientQwen-style interface.

    GDN's recurrence is `S_t = (g_t * I) S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T`,
    i.e. a scalar (per head) decay times the identity. KDA generalizes this to a
    diagonal decay (per channel per head). Otherwise the algorithms are identical;
    we use the same projections, the same GQA layout, the same short-conv with
    document-boundary handling, and the same gated RMSNorm + output projection.

    Args:
        d_model (int): Hidden size.
        d_state (int): Per-head key dimension (a.k.a. head_dim).
        n_q_heads (int): Number of query heads.
        n_k_heads (int): Number of key heads (GQA: < n_q_heads).
        n_v_heads (int): Number of value heads (GQA: < n_q_heads, == n_k_heads).
        d_conv (int): Short convolution kernel size.
        expand_q / expand_k / expand_v (float): Per-head expansion ratios.
            For apples-to-apples comparison with KDA/Mamba we default expand_v=1
            (matching teacher value head dim). The published GDN recipe uses 2.
        mode (str): "chunk" or "fused_recurrent" — the underlying FLA kernel.
        use_short_conv (bool): Whether to use short convolutions on q/k/v.
        allow_neg_eigval (bool): If True, scale beta by 2 (negative-eigenvalue trick).
        conv_bias (bool): Whether the short convolution has a bias.
        layer_idx (int): Layer index, used for cache addressing.
        rms_norm_eps (float): Epsilon for the output gated RMSNorm.
        activation (str): Activation for the short-conv branches. Defaults to
            "identity" (matching EfficientKDA).
        out_gate_activation (str): "sigmoid" or "silu" — passed to FusedRMSNormGated
            when use_post_ssm_norm=True, otherwise used to gate manually.
        use_gqa_setup (bool): GQA mode (n_q > n_k = n_v); K/V are repeated to Q.
        use_gva_setup (bool): GVA mode (n_v > n_q = n_k); Q/K are repeated to V.
        use_post_ssm_norm (bool): Whether to apply FusedRMSNormGated post-SSM
            (KDA's preferred path). When False, gate is applied without RMSNorm.
        use_gate (bool): Whether to use the output gate at all. If False, falls
            back to a plain RMSNorm (matches the FLA reference).
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
        use_gqa_setup: bool = True,
        use_gva_setup: bool = False,
        use_post_ssm_norm: bool = True,
        use_gate: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_size = d_conv
        self.use_short_conv = use_short_conv
        self.conv_bias = conv_bias
        self.use_gqa_setup = use_gqa_setup
        self.use_gva_setup = use_gva_setup
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
        self.use_post_ssm_norm = use_post_ssm_norm
        self.use_gate = use_gate
        if self.out_gate_activation not in ["sigmoid", "silu"]:
            raise ValueError(
                f"Unsupported output gate activation: {self.out_gate_activation}"
            )

        # Determine the head count the kernel sees after GQA/GVA expansion.
        # The output projection input dim is num_effective_heads * head_v_dim.
        if self.use_gqa_setup:
            assert self.n_q_heads % self.n_k_heads == 0, (
                f"n_q_heads={self.n_q_heads} must be divisible by n_k_heads={self.n_k_heads}"
            )
            assert self.n_q_heads % self.n_v_heads == 0, (
                f"n_q_heads={self.n_q_heads} must be divisible by n_v_heads={self.n_v_heads}"
            )
            self.num_effective_heads = self.n_q_heads
        elif self.use_gva_setup:
            assert self.n_v_heads % self.n_q_heads == 0
            assert self.n_v_heads % self.n_k_heads == 0
            self.num_effective_heads = self.n_v_heads
        else:
            assert self.n_q_heads == self.n_k_heads == self.n_v_heads
            self.num_effective_heads = self.n_q_heads
        self.out_dim = self.num_effective_heads * self.head_v_dim

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = d_model
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        # Projections — matched 1:1 to the Qwen3 teacher attention layout in GQA
        # mode (q_proj, k_proj, v_proj, o_proj have identical shapes when expand_*=1).
        self.q_proj = nn.Linear(self.hidden_size, self.query_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # Optional short convolutions on q/k/v with document-boundary support.
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
        else:
            warnings.warn(
                "ShortConvolution is crucial to GDN performance; setting "
                "use_short_conv=False is only intended for ablations.",
                RuntimeWarning,
            )

        # Decay parameters. Per-head (scalar, contra KDA's per-channel).
        # In log-decay space `g_t = -exp(A_log) * softplus(a_proj(x) + dt_bias)`.
        self.a_proj = nn.Linear(self.hidden_size, self.n_v_heads, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.n_v_heads, bias=False)

        A = torch.empty(self.n_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # dt_bias initialized via softplus-inverse of dt ~ exp(U(log(dt_min), log(dt_max)))
        dt_min, dt_max, dt_init_floor = 1e-3, 1e-1, 1e-4
        dt = torch.exp(
            torch.rand(self.n_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Output gate. The FLA reference uses a single Linear(hidden, value_dim).
        # We size it to the post-GQA-repeat head count so that the gate matches
        # the SSM output shape `(B, L, num_effective_heads, head_v_dim)`.
        if self.use_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.out_dim, bias=False)
            if self.use_post_ssm_norm:
                self.o_norm = FusedRMSNormGated(
                    self.head_v_dim,
                    activation=self.out_gate_activation,
                    eps=rms_norm_eps,
                )
            else:
                # Manual gating, no extra norm (matches the FLA "use_gate=True, no post norm" path
                # exists only conceptually — we expose it for ablations).
                self.o_norm = None
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=rms_norm_eps)

        self.out_proj = nn.Linear(self.out_dim, self.hidden_size, bias=False)

        # One-time-warning state, matching EfficientKDA.
        self._attention_mask_warned_shape = False
        self._mixer_matrix_not_supported_warned = False

    def _init_custom_weights(self):
        """Re-initialize after meta-device load (DeepSpeed / device_map='auto')."""
        # Identity init of conv1d.
        if self.use_short_conv:
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
                conv.weight.data.zero_()
                if self.conv_bias and conv.bias is not None:
                    conv.bias.data.zero_()
                conv.weight.data[:, :, -1] = 1.0

        # Re-run dt_bias and A_log initialization on the correct device.
        device = self.out_proj.weight.device
        dt_min, dt_max, dt_init_floor = 1e-3, 1e-1, 1e-4
        dt = torch.exp(
            torch.rand(self.n_v_heads, dtype=torch.float32, device=device)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.data.copy_(inv_dt)

        A = torch.empty(self.n_v_heads, dtype=torch.float32, device=device).uniform_(
            1, 16
        )
        self.A_log.data.copy_(torch.log(A))

        # Output norm weight init.
        if self.o_norm is not None:
            self.o_norm.weight.data.fill_(1.0)

        # NOTE: q/k/v/o/g/a/b projections fall through to the parent _init_weights.

    def init_mixer_proj_from_qkvo(self, checkpoint_file_dir: str):
        """Seed q/k/v/o projections from the teacher's attention QKVO."""
        if self.layer_idx is None:
            raise ValueError("init_mixer_proj_from_qkvo requires a valid layer_idx.")

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

        # Output projection
        o_weight_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.weight"
        o_bias_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.bias"
        if o_weight_name not in checkpoint.keys():
            raise KeyError(f"Missing tensor `{o_weight_name}` in teacher checkpoint.")
        self.out_proj.weight.data.copy_(checkpoint.get_tensor(o_weight_name))
        if o_bias_name in checkpoint.keys() and self.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(checkpoint.get_tensor(o_bias_name))

    def init_mixer_proj_from_vo(self, checkpoint_file_dir: str):
        """Seed only v_proj and o_proj from the teacher (cheapest VO transfer)."""
        if self.layer_idx is None:
            raise ValueError("init_mixer_proj_from_vo requires a valid layer_idx.")

        model_path = os.path.join(checkpoint_file_dir, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model.safetensors file found in {checkpoint_file_dir}")

        checkpoint = safe_open(model_path, framework="pt")

        v_weight_name = f"model.layers.{self.layer_idx}.self_attn.v_proj.weight"
        v_bias_name = f"model.layers.{self.layer_idx}.self_attn.v_proj.bias"
        if v_weight_name not in checkpoint.keys():
            raise KeyError(f"Missing tensor `{v_weight_name}` in teacher checkpoint.")
        v_weight = checkpoint.get_tensor(v_weight_name)
        v_bias = (
            checkpoint.get_tensor(v_bias_name)
            if v_bias_name in checkpoint.keys()
            else None
        )
        self.v_proj.weight.data.copy_(v_weight)
        if v_bias is not None and self.v_proj.bias is not None:
            self.v_proj.bias.data.copy_(v_bias)

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
        """
        EfficientQwen mixer-style forward (matches EfficientKDA.forward).

        Args:
            u: Input tensor (B, L, D).
            output_mixer_matrix: Not supported (GDN has no materialized mixer matrix).
            cache_params: Optional `EfficientQwenCache`. GDN reuses the kda_cache slots
                (same dict layout: recurrent_state, conv_state_q/k/v).
            attention_mask: Optional 2-D padding mask (B, L). Used to derive a
                document_boundary if none was supplied (matches KDA behaviour).
            document_boundary: Optional (B, L) tensor with 1 at document starts.
            layer_state: Optional explicit state dict (alternative to cache_params).
        """
        if output_mixer_matrix and not self._mixer_matrix_not_supported_warned:
            warnings.warn(
                "EfficientGatedDeltaNet does not support mixer_matrix materialization; ignoring.",
                RuntimeWarning,
            )
            self._mixer_matrix_not_supported_warned = True

        hidden_states = u
        batch_size, seq_len, _ = hidden_states.shape

        # Mode selection: fused_recurrent for single-token decode, chunk otherwise.
        mode = "fused_recurrent" if (not self.training and seq_len == 1) else "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        # Attention mask -> document_boundary derivation (prefill only).
        if attention_mask is not None and seq_len > 1:
            if attention_mask.dim() != 2:
                if not self._attention_mask_warned_shape:
                    warnings.warn(
                        "EfficientGatedDeltaNet: attention_mask is not (B, L); ignoring it.",
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

        # During single-token decode, drop a stale document_boundary (length-mismatched).
        if document_boundary is not None and document_boundary.shape[-1] != seq_len:
            document_boundary = None

        # Cache lookup — reuse kda_cache slots (same dict shape).
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

        # q/k/v projections + optional short convolution with document-boundary handling.
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

        # Reshape Q/K/V into head-major form.
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_q_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # Scalar-per-head decay + beta.
        # g_raw has shape (B, L, n_v_heads); we'll expand to num_effective_heads below.
        g_raw = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )
        beta = self.b_proj(hidden_states).float().sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # GQA / GVA expansion to match kernel head count.
        if self.use_gqa_setup and self.n_q_heads > self.n_k_heads:
            k = repeat(k, "... h d -> ... (h g) d", g=self.n_q_heads // self.n_k_heads)
            v = repeat(v, "... h d -> ... (h g) d", g=self.n_q_heads // self.n_v_heads)
            g_raw = repeat(
                g_raw, "... h -> ... (h g)", g=self.n_q_heads // self.n_v_heads
            )
            beta = repeat(
                beta, "... h -> ... (h g)", g=self.n_q_heads // self.n_v_heads
            )
        elif self.use_gva_setup and self.n_v_heads > self.n_q_heads:
            q = repeat(q, "... h d -> ... (h g) d", g=self.n_v_heads // self.n_q_heads)
            k = repeat(k, "... h d -> ... (h g) d", g=self.n_v_heads // self.n_k_heads)

        # Reset decay at document boundaries. `g_raw` is already in log-decay space
        # (negative values); 0 = no decay, large negative = full forget. We use a
        # large negative constant so the previous-document state is zeroed out.
        if document_boundary is not None:
            boundary_mask = document_boundary.to(dtype=torch.bool).unsqueeze(-1)
            g_raw = g_raw.masked_fill(boundary_mask, -80.0)

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        if mode == "chunk":
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g_raw,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache_flag,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g_raw,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache_flag,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Cache update — write back into kda_cache slots.
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

        # Output: gate + (optional) RMSNorm, then projection.
        if self.use_gate:
            gate = rearrange(
                self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
            )
            if self.use_post_ssm_norm:
                assert self.o_norm is not None
                o = self.o_norm(o.contiguous(), gate.contiguous())
            else:
                if self.out_gate_activation == "sigmoid":
                    o = o * torch.sigmoid(gate)
                else:
                    o = o * F.silu(gate)
        else:
            assert self.o_norm is not None
            o = self.o_norm(o)

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
        """Single-token decode step (mirrors EfficientKDA.step)."""
        if u.dim() != 2:
            raise ValueError(
                f"EfficientGatedDeltaNet.step expected (B, D), got {u.shape}"
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
        """Pre-allocate the recurrent + conv caches for generation."""
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

        # Recurrent state: (B, num_effective_heads, head_k_dim, head_v_dim) in fp32,
        # same shape as KDA's recurrent state.
        recurrent_state = torch.zeros(
            batch_size,
            self.num_effective_heads,
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
