# Copied from https://github.com/cartesia-ai/edge/blob/main/cartesia-pytorch/cartesia_pytorch/Llamba/mixers/discrete_mamba2.py  # noqa
import os
import contextlib
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from safetensors import safe_open
from transformers.integrations import use_kernel_forward_from_hub
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from fla.modules import FusedRMSNormGated

from gen_distill.models.layers.short_convolution import BoundaryAwareShortConvolution

def segsum(x):
    """More stable segment sum calculation."""
    # [1, 2, 3]
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    # [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    # [[0, 0, 0], [2, 0, 0], [3, 3, 0]]
    x_segsum = torch.cumsum(x, dim=-2)
    # [[0, 0, 0], [2, 0, 0], [5, 3, 0]]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def materialize_mixer(A_log, B, C, D):
    """
    Since the transfer matrix will be equated to the attention matrix,
    we need to support the form: torch.matmul(attn_weights, value_states).
    Thus, y = torch.matmul(T, X)
    Arguments:
        A_log: (bsz, seq_len, n_q_heads)
        B: (bsz, seq_len, n_q_heads, d_state/head_dim)
        C: (bsz, seq_len, n_q_heads, d_state/head_dim)
    Return:
        T: (bsz, n_q_heads, seq_len, seq_len)
    """
    bsz, seq_len, n_q_heads, d_state = B.shape
    assert A_log.shape == (bsz, seq_len, n_q_heads)
    assert B.shape == C.shape == (bsz, seq_len, n_q_heads, d_state)

    # Compute:
    A_log = rearrange(-F.softplus(A_log), "b l h -> b h l")
    powers = torch.exp(segsum(A_log))
    T = torch.einsum("blhn,bshn,bhls->bhsl", C, B, powers)

    # Add D:
    if D is not None:
        T[:, :, torch.arange(seq_len), torch.arange(seq_len)] += D.view(1, n_q_heads, 1)

    T = rearrange(T, "b h z l -> b h l z")
    return T


@use_kernel_forward_from_hub("RMSNorm")
class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_state):
        input_dtype = hidden_state.dtype
        hidden_state = hidden_state.to(torch.float32)
        variance = hidden_state.pow(2).mean(-1, keepdim=True)
        hidden_state = hidden_state * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_state.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Mamba2(nn.Module):
    """Unified Mamba2 implementation supporting both standard and discrete modes."""

    def __init__(
        self,
        d_model,
        d_state=128,
        n_q_heads=16,
        n_k_heads=8,
        n_v_heads=8,
        d_conv=4,
        expand_q=1,
        expand_k=1,
        expand_v=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        activation="identity",
        bias=False,
        conv_bias=True,
        chunk_size=128,
        layer_idx=None,
        device=None,
        dtype=None,
        rms_norm_eps=1e-6,
        use_post_ssm_norm=False,
        use_discrete_mode=False,
        use_qk_norm=False,
        use_gqa_setup=True,
        use_gva_setup=False,
        **kwargs,
    ):
        """
        Unified Mamba2 implementation.

        Args:
            use_discrete_mode: If True, uses discrete parameterization (A_log as dt, A fixed to -1).
                              If False, uses standard parameterization (separate dt_bias and A_log).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # ARGUMENTS
        # General layer args
        self.layer_idx = layer_idx
        self.d_model = d_model  # transformer equivalent: hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.use_discrete_mode = use_discrete_mode
        self.use_qk_norm = use_qk_norm
        self.use_gqa_setup = use_gqa_setup
        self.use_gva_setup = use_gva_setup
        self.kwargs = kwargs
        self.expand_q = expand_q
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.n_q_heads = n_q_heads
        self.n_k_heads = n_k_heads
        self.n_v_heads = n_v_heads
        self.head_k_dim = int(self.d_state * self.expand_k)
        self.head_q_dim = int(self.d_state * self.expand_q)
        self.head_v_dim = int(self.d_state * self.expand_v)
        self.key_dim = int(self.n_k_heads * self.head_k_dim)
        self.query_dim = int(self.n_q_heads * self.head_q_dim)
        self.value_dim = int(self.n_v_heads * self.head_v_dim)

        if self.use_gqa_setup:
            assert self.n_q_heads % self.n_k_heads == 0
            assert self.n_q_heads % self.n_v_heads == 0
            self.out_dim = self.query_dim
            self.z_bias = (
                nn.Parameter(1.278 * torch.ones(self.query_dim, **factory_kwargs))
                if not bias
                else 1.278 * torch.ones(self.query_dim, **factory_kwargs)
            )
        elif self.use_gva_setup:
            assert self.n_v_heads % self.n_q_heads == 0
            assert self.n_v_heads % self.n_k_heads == 0
            self.out_dim = self.value_dim
            self.z_bias = (
                nn.Parameter(1.278 * torch.ones(self.value_dim, **factory_kwargs))
                if not bias
                else 1.278 * torch.ones(self.value_dim, **factory_kwargs)
            )
        # SSM specific args
        self.activation = activation
        self.chunk_size = chunk_size
        self.bias = bias
        self.dt_limit = dt_limit if not use_discrete_mode else (0.0, float("inf"))
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.dt_init_floor = dt_init_floor
        self.A_init_range = A_init_range

        # LAYERS
        # Projections: xBCA_log --> xBC, A_log
        self.in_proj = nn.Linear(
            self.d_model,
            self.out_dim + self.key_dim + self.query_dim + self.value_dim + self.n_q_heads,
            bias=bias,
            **factory_kwargs,
        )

        # QK norm that are compatible with Qwen3
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.d_state, eps=rms_norm_eps)  # norm over the head_dim
            self.k_norm = RMSNorm(self.d_state, eps=rms_norm_eps)  # norm over the head_dim

        # Convolutional layer
        self.conv_bias = conv_bias
        conv_dim = self.key_dim + self.query_dim + self.value_dim
        self.short_conv = BoundaryAwareShortConvolution(
            hidden_size=conv_dim,
            kernel_size=d_conv,
            bias=conv_bias,
            activation=self.activation,
            backend='cuda',
            device=device,
            dtype=dtype,
        )

        # Mode-specific parameter initialization
        if not use_discrete_mode:
            # Standard mode: Initialize dt bias and A_log separately
            # dt: q heads, because it is used after the GQA expansion
            dt = torch.exp(
                torch.rand(self.n_q_heads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True

            # A parameter: Q heads, because it is called after the GQA expansion
            assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
            A = torch.empty(self.n_q_heads, dtype=torch.float32, device=device).uniform_(
                *A_init_range
            )
            A_log = torch.log(A).to(dtype=dtype)
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
        # In discrete mode, we don't need dt_bias or separate A_log initialization

        # D "skip" parameter - we handle both GQA and GVA cases directly
        self.D = nn.Parameter(torch.zeros(max(self.n_q_heads, self.n_v_heads), **factory_kwargs))
        self.D._no_weight_decay = True

        self.use_post_ssm_norm = use_post_ssm_norm
        if self.use_post_ssm_norm:
            self.o_norm = FusedRMSNormGated(
                self.head_v_dim, eps=rms_norm_eps, activation="swish", **factory_kwargs
            )

        # Out projection
        self.out_proj = nn.Linear(self.out_dim, self.d_model, bias=bias, **factory_kwargs)

        # Track one-time document boundary warning
        self._document_boundary_warned = False

        # Track one-time attention mask warnings
        self._attention_mask_warned_shape = False

    def _init_custom_weights(self):
        """
        Re-initializes the weights of the Mamba2 module. This is particularly
        useful when the model is loaded with `device_map="auto"`, as the default
        initialization in `__init__` is skipped for modules on meta device.
        """
        factory_kwargs = {"device": self.in_proj.weight.device, "dtype": self.in_proj.weight.dtype}

        # Identity init of conv1d
        self.short_conv.weight.data.zero_()
        if self.conv_bias:
            self.short_conv.bias.data.zero_()
        self.short_conv.weight.data[:, :, -1] = 1.0

        # Identity init of gate projection --> zeros on projection, and 1.278 on bias
        self.in_proj.weight.data[: self.out_dim].zero_()
        if not self.bias:
            self.z_bias.fill_(1.278)

        if not self.use_discrete_mode:
            # Re-run dt_bias initialization logic
            dt = torch.exp(
                torch.rand(self.n_q_heads, **factory_kwargs)
                * (math.log(self.dt_max) - math.log(self.dt_min))
                + math.log(self.dt_min)
            )
            dt = torch.clamp(dt, min=self.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.data.copy_(inv_dt)

            # Re-run A_log initialization logic
            A = torch.empty(self.n_q_heads, device=factory_kwargs["device"]).uniform_(
                *self.A_init_range
            )
            A_log = torch.log(A).to(dtype=factory_kwargs["dtype"])
            self.A_log.data.copy_(A_log)

        # Identity init of D
        self.D.fill_(0.0)

        # Initialize the rms_norm weights if present
        if self.use_post_ssm_norm:
            self.o_norm.weight.data.fill_(1.0)

        # Initialize qk norm if used
        if self.use_qk_norm:
            self.q_norm.weight.data.fill_(1.0)
            self.k_norm.weight.data.fill_(1.0)

        # NOTE: Out projection will use the default initialization (will be called by the
        # _init_weights_) from the parent model. Same for the post SSM normalization

    def init_mixer_proj_from_qkvo(self, checkpoint_file_dir):
        """Load student projections from a teacher Transformer's QKV/O projections.

        The teacher model (`EfficientQwenAttention`) exposes individual `q_proj`, `k_proj`,
        `v_proj`, and `o_proj` layers. After reshaping the hidden states (`[B, L, D]` with
        `D = H * E`) to `[B, H, L, E]`, these projections produce the query, key, and value
        states consumed by the attention mechanism. The output projection (`o_proj`) maps the
        attended representation back to `[B, L, D]`.

        The student model (`Mamba2`) uses a single `in_proj` that yields the `x`, `B`, `C`,
        and `A_log` components. The `x`, `B`, and `C` slices correspond directly to the
        teacher's V, K, and Q projections, so their weight and bias segments can be copied
        without additional transposition. Similarly, the student's `out_proj` aligns with the
        teacher's `o_proj`.

        Teacher checkpoint tensors follow the pattern
        `model.layers.{layer_idx}.self_attn.{qkvo}_proj.{weight|bias}`.
        """

        if not os.path.exists(os.path.join(checkpoint_file_dir, "model.safetensors")):
            raise FileNotFoundError(
                f"No model.safetensors or pytorch_model.bin file found in {checkpoint_file_dir}"
            )
        if not self.use_gqa_setup:
            raise ValueError("Only GQA setup is supported for now for loading from the teacher")
        checkpoint = safe_open(
            os.path.join(checkpoint_file_dir, "model.safetensors"), framework="pt"
        )

        # Load and copy the Q,K,V projections from teacher
        for param_name in ["q_proj", "k_proj", "v_proj"]:
            name = f"model.layers.{self.layer_idx}.self_attn.{param_name}.weight"
            bias_name = f"model.layers.{self.layer_idx}.self_attn.{param_name}.bias"
            param = checkpoint.get_tensor(name)
            if bias_name in checkpoint.keys():
                bias = checkpoint.get_tensor(bias_name)
            else:
                bias = None

            # in_proj -> x, B, C --> equivalent to V, K, Q projections
            if param_name == "v_proj":
                self.in_proj.weight.data[self.out_dim : self.out_dim + self.value_dim].copy_(param)
                if bias is not None:
                    self.in_proj.bias.data[self.out_dim : self.out_dim + self.value_dim].copy_(bias)
            elif param_name == "k_proj":
                self.in_proj.weight.data[
                    self.out_dim + self.value_dim : self.out_dim + self.value_dim + self.key_dim
                ].copy_(param)
                if bias is not None:
                    self.in_proj.bias.data[
                        self.out_dim + self.value_dim : self.out_dim + self.value_dim + self.key_dim
                    ].copy_(bias)
            elif param_name == "q_proj":
                self.in_proj.weight.data[
                    self.out_dim
                    + self.value_dim
                    + self.key_dim : self.out_dim
                    + self.value_dim
                    + self.key_dim
                    + self.query_dim
                ].copy_(param)
                if bias is not None:
                    self.in_proj.bias.data[
                        self.out_dim
                        + self.value_dim
                        + self.key_dim : self.out_dim
                        + self.value_dim
                        + self.key_dim
                        + self.query_dim
                    ].copy_(bias)

        # Optionally load QK RMSNorm weights from teacher
        if self.use_qk_norm:
            q_norm_name = f"model.layers.{self.layer_idx}.self_attn.q_norm.weight"
            k_norm_name = f"model.layers.{self.layer_idx}.self_attn.k_norm.weight"
            if q_norm_name in checkpoint.keys():
                q_norm_weight = checkpoint.get_tensor(q_norm_name)
                self.q_norm.weight.data.copy_(q_norm_weight)
            if k_norm_name in checkpoint.keys():
                k_norm_weight = checkpoint.get_tensor(k_norm_name)
                self.k_norm.weight.data.copy_(k_norm_weight)

        # Load and copy the O projection
        name = f"model.layers.{self.layer_idx}.self_attn.o_proj.weight"
        bias_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.bias"
        param = checkpoint.get_tensor(name)
        self.out_proj.weight.data.copy_(param)
        if bias_name in checkpoint.keys():
            bias = checkpoint.get_tensor(bias_name)
            self.out_proj.bias.data.copy_(bias)

    def init_mixer_proj_from_vo(self, checkpoint_file_dir):
        """Load student projections from a teacher Transformer's Value and Output projections only.

        We follow the exact same approach as init_mamba_proj_from_qkvo, but we only load the
        Value and Output projections.

        """

        if not os.path.exists(os.path.join(checkpoint_file_dir, "model.safetensors")):
            raise FileNotFoundError(
                f"No model.safetensors or pytorch_model.bin file found in {checkpoint_file_dir}"
            )
        checkpoint = safe_open(
            os.path.join(checkpoint_file_dir, "model.safetensors"), framework="pt"
        )

        # Load and copy the V projections from teacher
        name = f"model.layers.{self.layer_idx}.self_attn.v_proj.weight"
        bias_name = f"model.layers.{self.layer_idx}.self_attn.v_proj.bias"
        param = checkpoint.get_tensor(name)
        if bias_name in checkpoint.keys():
            bias = checkpoint.get_tensor(bias_name)
        else:
            bias = None
        self.in_proj.weight.data[self.out_dim : self.out_dim + self.value_dim].copy_(param)
        if bias is not None:
            self.in_proj.bias.data[self.out_dim : self.out_dim + self.value_dim].copy_(bias)

        # Load and copy the O projection
        name = f"model.layers.{self.layer_idx}.self_attn.o_proj.weight"
        bias_name = f"model.layers.{self.layer_idx}.self_attn.o_proj.bias"
        param = checkpoint.get_tensor(name)
        self.out_proj.weight.data.copy_(param)
        if bias_name in checkpoint.keys():
            bias = checkpoint.get_tensor(bias_name)
            self.out_proj.bias.data.copy_(bias)

    def forward(self, u, output_mixer_matrix=False, cache_params=None, document_boundary=None, attention_mask=None, **kwargs):
        """
        Args:
            u (tensor): Input tensor of shape (B, L, D).
            output_mixer_matrix (bool): Whether to output the mixer matrix.
            cache_params (dict): Cache parameters.
            document_boundary (tensor): Document boundary tensor of shape (B, L).
            attention_mask (tensor): Attention mask tensor of shape (B, L).
        Returns:
            outputs (dict): Dictionary containing the output states and mixer matrix.
            outputs["output_states"] (tensor): Output states of shape (B, L, D).
            outputs["mixer_matrix"] (tensor): (Optional) the materialized mixer matrix of shape (B, L, L).
        """
        bsz, seq_len, dim = u.shape
        # Convert a 2D attention_mask (B, L) into a boundary-like mask where
        # we reset state on the first non-padding token after padding tokens.
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                if not self._attention_mask_warned_shape:
                    warnings.warn(
                        "Mamba2: received attention_mask not of shape (B, L); ignoring it for Mamba computation.",
                        RuntimeWarning,
                    )
                    self._attention_mask_warned_shape = True
            else:
                # Detect transitions from padding (False) to non-padding (True)
                attn_bool = attention_mask.to(dtype=torch.bool)
                prev = F.pad(attn_bool[:, :-1], (1, 0), value=False)
                boundary_from_mask = attn_bool & (~prev)
                if document_boundary is None:
                    document_boundary = boundary_from_mask
                else:
                    document_boundary = (
                        document_boundary.to(dtype=torch.bool) | boundary_from_mask
                    )

        if (
            document_boundary is not None
            and not self.use_discrete_mode
        ):
            if not self._document_boundary_warned:
                warnings.warn(
                    "Mamba2: document_boundary is ignored; state resets across documents are not supported. Ignoring the document_boundary parameter.",
                    RuntimeWarning,
                )
                self._document_boundary_warned = True
            document_boundary = None

        # Mode-specific A parameter handling
        if not self.use_discrete_mode:
            # Standard mode: Ensure negative value for the A matrix
            A = -torch.exp(self.A_log)  # (n_kv_heads) or (d_inner, d_state)

            # If learnable initial states, prepare them
            dt_limit_kwargs = (
                {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
            )
        else:
            A = None  # Will be handled differently in discrete mode
            dt_limit_kwargs = {}

        # Initialize the cache if it does not exist (will use it both in forward() and step())
        state = None
        if cache_params is not None:
            state = cache_params.get_mamba_state(self.layer_idx)
            if state is None:
                state = self.allocate_inference_cache(u.shape[0], u.dtype)
                cache_params.update_mamba_state(
                    self.layer_idx, state["ssm_state"], state["conv_state"]
                )
        elif seq_len == 1:
            # We also need to make sure that we have a state in this case
            state = self.allocate_inference_cache(u.shape[0], u.dtype)

        # Choose step() or forward() based on the sequence length
        if seq_len == 1:
            token = u.squeeze(1)
            out_token, state = self.step(token, state=state, cache_params=cache_params)
            if cache_params is not None:
                cache_params.update_mamba_state(self.layer_idx, state["ssm_state"], state["conv_state"])
            return {"output_states": out_token.unsqueeze(1)}

        # Pad input to nearest multiple of chunk_size
        padded_len = (1 + (seq_len - 1) // self.chunk_size) * self.chunk_size
        u = F.pad(u, (0, 0, 0, padded_len - seq_len))
        if document_boundary is not None:
            document_boundary = F.pad(document_boundary, (0, padded_len - document_boundary.shape[1]))

        # Project input and split the projection
        if not self.use_discrete_mode:
            # Standard mode: split into xBC and dt
            zxBCdt = self.in_proj(u)
            z, xBC, dt = torch.split(
                zxBCdt,
                [
                    self.out_dim,
                    self.key_dim + self.query_dim + self.value_dim,
                    self.n_q_heads,
                ],
                dim=-1,
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, n_q_heads)
        else:
            # Discrete mode: split into xBC and A_log
            zxBCA_log = self.in_proj(u)
            z, xBC, A_log = torch.split(
                zxBCA_log,
                [
                    self.out_dim,
                    self.key_dim + self.query_dim + self.value_dim,
                    self.n_q_heads,
                ],
                dim=-1,
            )

        if self.use_qk_norm:
            x, B, C = torch.split(
                xBC,
                [
                    self.value_dim,
                    self.key_dim,
                    self.query_dim,
                ],
                dim=-1,
            )
            B = self.k_norm(B.view((B.shape[0], B.shape[1], -1, self.d_state)))
            C = self.q_norm(C.view((C.shape[0], C.shape[1], -1, self.d_state)))
            B = B.view((B.shape[0], B.shape[1], -1))
            C = C.view((C.shape[0], C.shape[1], -1))
            xBC = torch.cat([x, B, C], dim=-1)

        # Convolutional layer (pass cache when available)
        xBC, conv_state = self.short_conv(
            xBC,
            document_boundary=document_boundary,
            cache=state["conv_state"] if state is not None else None,
            output_final_state=state is not None,
        )
        if state is not None:
            state["conv_state"] = conv_state

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.value_dim, self.key_dim, self.query_dim], dim=-1)
        x = rearrange(x, "b l (h n) -> b l h n", h=self.n_v_heads)  # V
        B = rearrange(B, "b l (h n) -> b l h n", h=self.n_k_heads)  # K
        C = rearrange(C, "b l (h n) -> b l h n", h=self.n_q_heads)  # Q

        # GQA: Expand x and B (correspond to V and K) to match number of heads with C
        if self.use_gqa_setup and self.n_q_heads > max(self.n_k_heads, self.n_v_heads):
            repeat_factor = self.n_q_heads // max(self.n_k_heads, self.n_v_heads)
            B = B.repeat_interleave(repeat_factor, dim=2)
            x = x.repeat_interleave(repeat_factor, dim=2)
        elif self.use_gva_setup and self.n_v_heads > max(self.n_q_heads, self.n_k_heads):
            # GVA: Expand C and B (correspond to Q and K) to match number of heads with x (V)
            repeat_factor = self.n_v_heads // max(self.n_q_heads, self.n_k_heads)
            C = C.repeat_interleave(repeat_factor, dim=2)
            B = B.repeat_interleave(repeat_factor, dim=2)
            if not self.use_discrete_mode:
                dt = dt.repeat_interleave(repeat_factor, dim=2)
                # A repeating is handled later, since it gets created from the self.A_log
            else:
                # In discrete mode A_log is (B, L, n_q) if I recall correctly? No, A_log is param
                # Wait, check A_log usage.
                pass

        # Compute the SSM
        if not self.use_discrete_mode:
            # Ensure Triton launches on the same CUDA device as tensors
            A = -torch.exp(self.A_log)  # (n_q_heads)
            if self.use_gva_setup and self.n_q_heads != self.n_v_heads:
                repeat_factor_q = self.n_v_heads // self.n_q_heads
                A = A.repeat_interleave(repeat_factor_q, dim=0)
                # dt was already repeated above if gva setup

            with torch.cuda.device(u.device) if u.is_cuda else contextlib.nullcontext():
                result = mamba_chunk_scan_combined(
                    x=x,
                    dt=dt,
                    A=A,
                    B=B,
                    C=C,
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    initial_states=state["ssm_state"] if state is not None else None,
                    dt_softplus=False,
                    return_final_states=(state is not None),
                    **dt_limit_kwargs,
                )
        else:
            # Discrete mode
            # Compute a numerically stable denominator in float32 and clamp away from zero
            denom = F.softplus(A_log.float()).clamp_min(1e-6)

            # Discrete Mamba makes input selectvitiy independent of dt, by rescaling X -> x/dt
            # so then the actual dt does not impact how the input gets added to the state
            # --> we can modify the dt to reset the hidden states at the document boundaries
            if document_boundary is not None:
                boundary_mask = document_boundary.to(dtype=torch.bool).unsqueeze(-1)  # (B, L, 1)
                denom = denom.masked_fill(boundary_mask, denom.new_tensor(100))
            if self.use_gva_setup and self.n_q_heads != self.n_v_heads:
                repeat_factor_q = self.n_v_heads // self.n_q_heads
                denom = denom.repeat_interleave(repeat_factor_q, dim=-1)
            x_discrete = (x.float() / denom.unsqueeze(-1)).to(x.dtype)  # division in FP32

            A_fixed = -torch.ones(self.n_q_heads, device=A_log.device)
            if self.use_gva_setup and self.n_q_heads != self.n_v_heads:
                repeat_factor_q = self.n_v_heads // self.n_q_heads
                A_fixed = A_fixed.repeat_interleave(repeat_factor_q, dim=0)

            # Ensure Triton kernels launch on the correct CUDA device context
            with torch.cuda.device(u.device) if u.is_cuda else contextlib.nullcontext():
                result = mamba_chunk_scan_combined(
                    x=x_discrete,
                    dt=denom,
                    dt_softplus=False,
                    A=A_fixed,
                    B=B,
                    C=C,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    initial_states=state["ssm_state"] if state is not None else None,
                    return_final_states=(state is not None),
                )

        if state is not None:
            o, ssm_state = result
            state["ssm_state"] = ssm_state
            if cache_params is not None:
                cache_params.update_mamba_state(self.layer_idx, state["ssm_state"], state["conv_state"])
        else:
            o = result

        # Add skip connection - already handled in the kernel itself!
        # Du = torch.einsum("h,blhp->blhp", self.D, x)
        # o = o + Du  # (B, L, n_q_heads, d_state)

        # Norm, gate and projection
        # handle both GQA and GVA
        z = rearrange(z, "b l (h n) -> b l h n", h=max(self.n_q_heads, self.n_v_heads))
        # z_bias is of shape (n_q_heads, head_q_dim) or (n_v_heads, head_v_dim)
        # o and z are of shape (B, L, n_q_heads, head_q_dim) or (B, L, n_v_heads, head_v_dim)
        # --> need to unsqueeze z_bias
        z_bias = rearrange(self.z_bias, "(h n) -> h n", h=max(self.n_q_heads, self.n_v_heads))
        z_sum = z + z_bias.unsqueeze(0).unsqueeze(0)
        if self.use_post_ssm_norm:
            o = self.o_norm(o, z_sum)
        else:
            o = (o * F.silu(z_sum.float())).to(o.dtype)

        o = rearrange(o, "b l h p -> b l (h p)")  # (B, L, D)
        o = self.out_proj(o)

        # Materialize the mixer matrix if required
        outputs = {}
        if output_mixer_matrix:
            if not self.use_discrete_mode:
                mixer_matrix = materialize_mixer(
                    A_log=A[None, None, :] * dt, B=B * dt[..., None], C=C, D=self.D
                )
            else:
                A_log_mat = A_log
                if self.use_gva_setup and self.n_q_heads != self.n_v_heads:
                    repeat_factor_q = self.n_v_heads // self.n_q_heads
                    A_log_mat = A_log_mat.repeat_interleave(repeat_factor_q, dim=0)

                mixer_matrix = materialize_mixer(A_log=A_log_mat, B=B, C=C, D=self.D)
            outputs["mixer_matrix"] = mixer_matrix[..., :seq_len, :seq_len]

        # Prepare output dictionary
        outputs["output_states"] = o[:, :seq_len, :]
        return outputs

    def step(self, u, state, cache_params=None, document_boundary=None, **kwargs):
        """
        Args:
            u: (B, D),
            ca
            state: dict.

        Returns:
            out: (B, D),
            state: dict.

        """
        if document_boundary is not None:
            if not self._document_boundary_warned:
                warnings.warn(
                    "Mamba2: document_boundary is ignored in step(); state resets across documents are not supported.",
                    RuntimeWarning,
                )
                self._document_boundary_warned = True
            document_boundary = None

        # Project input and split the projection
        if not self.use_discrete_mode:
            # Standard mode
            zxBCdt = self.in_proj(u)
            z, xBC, dt = torch.split(
                zxBCdt,
                [
                    self.out_dim,
                    self.query_dim + self.key_dim + self.value_dim,
                    self.n_q_heads,
                ],
                dim=-1,
            )
            # dt = F.softplus(dt + self.dt_bias)  # (B, L, n_kv_heads)
            # NOTE: Bug in the selective_state_update requires us to pass the dt_bias, so we will
            # not precompute the softplus, but rather pass the dt and dt_bias separately?
        else:
            # Discrete mode
            zxBCA_log = self.in_proj(u)
            z, xBC, A_log = torch.split(
                zxBCA_log,
                [
                    self.out_dim,
                    self.query_dim + self.key_dim + self.value_dim,
                    self.n_q_heads,
                ],
                dim=-1,
            )

        if self.use_qk_norm:
            x, B, C = torch.split(xBC, [self.value_dim, self.key_dim, self.query_dim], dim=-1)
            # Reshape to (B, heads, d_state), apply RMSNorm, then flatten back
            B = B.view(B.shape[0], self.n_k_heads, self.d_state)
            C = C.view(C.shape[0], self.n_q_heads, self.d_state)
            B = self.k_norm(B)
            C = self.q_norm(C)
            B = B.view(B.shape[0], -1)
            C = C.view(C.shape[0], -1)
            xBC = torch.cat([x, B, C], dim=-1)

        # Convolutional layer
        xBC, conv_state = self.short_conv.step(
            xBC, residual=False, cache=state["conv_state"], document_boundary=document_boundary
        )
        state["conv_state"] = conv_state

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.value_dim, self.key_dim, self.query_dim], dim=-1)
        x_reshaped = rearrange(x, "b (h s) -> b h s", h=self.n_v_heads)
        B = rearrange(B, "b (h s) -> b h s", h=self.n_k_heads)
        C = rearrange(C, "b (h s) -> b h s", h=self.n_q_heads)

        # GQA: expand x and B to match C's head count
        if self.use_gqa_setup and self.n_q_heads > max(self.n_k_heads, self.n_v_heads):
            repeat_factor = self.n_q_heads // max(self.n_k_heads, self.n_v_heads)
            B = B.repeat_interleave(repeat_factor, dim=1)
            x_reshaped = x_reshaped.repeat_interleave(repeat_factor, dim=1)
        elif self.use_gva_setup and self.n_v_heads > max(self.n_q_heads, self.n_k_heads):
            repeat_factor = self.n_v_heads // max(self.n_q_heads, self.n_k_heads)
            C = C.repeat_interleave(repeat_factor, dim=1)
            B = B.repeat_interleave(repeat_factor, dim=1)
            if not self.use_discrete_mode:
                dt = dt.repeat_interleave(repeat_factor, dim=1)

        # Compute the SSM
        state["ssm_state"] = state["ssm_state"].to(x.dtype)

        if not self.use_discrete_mode:
            dt = repeat(dt, "b h -> b h p", p=self.d_state)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.d_state)
            A_param = -torch.exp(self.A_log.float())  # (n_q_heads,)
            if self.use_gva_setup and self.n_q_heads != self.n_v_heads:
                repeat_factor = self.n_v_heads // self.n_q_heads
                dt_bias = dt_bias.repeat_interleave(repeat_factor, dim=0)
                A_param = A_param.repeat_interleave(repeat_factor, dim=0)

            A_param = repeat(A_param, "h -> h p n", p=self.d_state, n=self.d_state).to(
                dtype=torch.float32, device=x_reshaped.device
            )
            # zeros_D = torch.zeros(
            #     (self.n_v_heads if self.use_gva_setup else self.n_q_heads, self.d_state),
            #     device=x_reshaped.device,
            #     dtype=x_reshaped.dtype,
            # )
            # Ensure CUDA kernels execute on the tensor's device
            # with torch.cuda.device(x_reshaped.device) if x_reshaped.is_cuda else contextlib.nullcontext():
            o = selective_state_update(
                state=state["ssm_state"],
                x=x_reshaped,
                dt=dt,
                A=A_param,
                B=B,
                C=C,
                D=self.D[:, None].expand(-1, self.d_state),
                dt_softplus=True,
                dt_bias=dt_bias,
            )
        else:
            # Discrete mode
            # Compute a numerically stable denominator in float32 and clamp away from zero
            denom = F.softplus(A_log.float()).clamp_min(1e-6)  # (B, n_q_heads)
            A_fixed = -torch.ones(
                (self.n_q_heads, self.d_state, self.d_state),
                device=x_reshaped.device,
                dtype=x_reshaped.dtype,
            )
            if self.use_gva_setup and self.n_q_heads != self.n_v_heads:
                repeat_factor_q = self.n_v_heads // self.n_q_heads
                denom = denom.repeat_interleave(repeat_factor_q, dim=-1)
                A_fixed = A_fixed.repeat_interleave(repeat_factor_q, dim=0)

            # If a boundary is specified at this step, set denom to a large value to reset
            if document_boundary is not None:
                # Support shapes (B,) or (B,1)
                if document_boundary.dim() == 1:
                    boundary_mask = document_boundary.to(dtype=torch.bool).unsqueeze(-1)  # (B, 1)
                else:
                    boundary_mask = document_boundary.to(dtype=torch.bool)  # (B, 1)
                denom = denom.masked_fill(boundary_mask, 100.0)
            x_discrete = (x_reshaped.float() / denom.unsqueeze(-1)).to(
                x_reshaped.dtype
            )  # (B, n_q_heads, head_dim)

            # with torch.cuda.device(x_reshaped.device) if x_reshaped.is_cuda else contextlib.nullcontext():
            o = selective_state_update(
                x=x_discrete,
                dt=repeat(denom, "b h -> b h p", p=self.d_state),
                dt_softplus=False,
                A=A_fixed,
                B=B,
                C=C,
                state=state["ssm_state"],  # will be updated in place
                dt_bias=torch.zeros(
                    (self.n_v_heads if self.use_gva_setup else self.n_q_heads, self.d_state),
                    device=x_reshaped.device,
                    dtype=x_reshaped.dtype,
                ),
                D=self.D[:, None].expand(-1, self.d_state),
                # D=torch.zeros(
                #     (self.n_v_heads if self.use_gva_setup else self.n_q_heads, self.d_state),
                #     device=x_reshaped.device,
                #     dtype=x_reshaped.dtype,
                # ),
            )

        # Skip connection (match forward): y = y + D * x
        # o = o + self.D[:, None] * x_reshaped

        # Gate, Norm and project
        # z_bias is of shape (n_q_heads, head_q_dim) or (n_v_heads, head_v_dim)
        # o and z are of shape (B, n_q_heads, head_q_dim) or (B, n_v_heads, head_v_dim)
        # --> need to unsqueeze z_bia
        z = rearrange(z, "b (h n) -> b h n", h=max(self.n_q_heads, self.n_v_heads))
        z_bias = rearrange(self.z_bias, "(h n) -> h n", h=max(self.n_q_heads, self.n_v_heads))
        z_sum = z + z_bias.unsqueeze(0)
        if self.use_post_ssm_norm:
            o = self.o_norm(o, z_sum)
        else:
            o = (o * F.silu(z_sum.float())).to(o.dtype)
        o = rearrange(o, "b h p -> b (h p)")
        o = self.out_proj(o)
        return o, state

    def allocate_inference_cache(self, batch_size, dtype=None, **kwargs):
        device = self.in_proj.weight.device
        # conv_state:
        conv_dtype = self.short_conv.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_conv,
            self.short_conv.weight.shape[0],
            device=device,
            dtype=conv_dtype,
        ).transpose(1, 2)
        # ssm_state:
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.n_v_heads if self.use_gva_setup else self.n_q_heads,
            self.d_state,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return {"conv_state": conv_state, "ssm_state": ssm_state}


# Backwards compatibility alias
class DiscreteMamba2(Mamba2):
    """Backwards compatibility alias for DiscreteMamba2."""

    def __init__(self, *args, **kwargs):
        # Force discrete mode for backwards compatibility
        kwargs["use_discrete_mode"] = True
        super().__init__(*args, **kwargs)
