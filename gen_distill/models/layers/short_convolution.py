import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import warnings
from einops import rearrange

try:
    from causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update as causal_conv1d_update_cuda,
    )
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update_cuda = None

try:
    from fla.ops.triton.causal_conv1d import causal_conv1d, causal_conv1d_update  # type: ignore
except ImportError:
    causal_conv1d = None
    causal_conv1d_update = None

try:
    from fla.utils import prepare_sequence_ids
except ImportError:
    prepare_sequence_ids = None


class BoundaryAwareShortConvolution(nn.Conv1d):
    """
    Simple wrapper around `nn.Conv1d` that accepts dimension last.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = "silu",
        backend: Optional[str] = "cuda",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None

        if activation == "identity":
            self.activation = None
        elif activation is not None:
            assert activation in ["silu", "swish"], (
                f"Activation `{activation}` not supported yet."
            )
            self.activation = activation

        if "use_fast_conv1d" in kwargs:
            warnings.warn(
                "The `use_fast_conv1d` parameter is deprecated and will be ignored. "
                "Please use the `backend` parameter instead."
            )
        import os

        self.backend = os.environ.get("FLA_CONV_BACKEND", backend)
        if self.backend not in ["cuda", "triton", "pytorch"]:
            raise ValueError(
                f"Invalid backend: {self.backend}, must be one of ['cuda', 'triton', 'pytorch']"
            )
        # Test self.backend, not the `backend` parameter: otherwise an FLA_CONV_BACKEND
        # override is read and then immediately discarded by this branch.
        if self.backend == "cuda":
            if causal_conv1d_fn is None:
                warnings.warn(
                    "The `backend` parameter is set to `cuda`, but `causal_conv1d_fn` is not available. "
                    "Switching to the Triton implementation instead. "
                    "Consider installing `causal_conv1d` to enable the CUDA backend."
                )
                self.backend = "triton"

        # Track whether fallback warnings have been issued
        self._fallback_warned_forward = False
        self._fallback_warned_step = False

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        if self.activation is not None:
            s += ", activation={activation}"
        s += f", backend={self.backend}"
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        document_boundary: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`. `B` must be 1 if `seq_idx` is provided.
            residual (`Optional[torch.Tensor]`):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]
            document_boundary (Optional[torch.Tensor]):
                Binary tensor of shape `[B, T]` indicating document boundaries (1 = new document start).
                Default: `None`.

        Returns:
            Tensor of shape `[B, T, D]`.
        """

        B, T, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError(
                    "`mask` and `cu_seqlens` cannot be provided at the same time"
                )
            x = x.mul_(mask.unsqueeze(-1))

        # in decoding phase, the cache (if provided) is updated inplace
        if B * T == N:
            y, cache = self.step(
                x=x,
                residual=residual,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                document_boundary=document_boundary,
            )
            return y, cache

        # check if cu_seqlens and cache are both provided
        # Sequence index for each token. Used for varlen.
        # Suppose a batch consists of two sequences with lengths 3 and 4,
        # seq_idx=[0, 0, 0, 1, 1, 1, 1] for this batch.
        # NOTE: No need to provide this arg if `cu_seqlens` is passed.
        # This arg is just for BC, and will be removed in the future.
        # [B, T]
        seq_idx = kwargs.get("seq_idx", None)
        # cuda backend do not support:
        # 1. both `cu_seqlens` and `cache` being provided
        # 2. both `cu_seqlens` and `output_final_state` being provided
        if self.backend == "cuda" and (
            ((cu_seqlens is not None or seq_idx is not None) and cache is not None)
            or (cu_seqlens is not None and output_final_state)
        ):
            warnings.warn(
                "The CUDA backend does not support both `cu_seqlens` and `cache` being provided, "
                "or both `cu_seqlens` and `output_final_state` being provided. "
                "Switching to the Triton backend instead. ",
                stacklevel=2,
            )
            self.backend = "triton"

        # Handle document boundary-aware convolution
        if document_boundary is not None:
            # Expand sequence to insert zeros at document boundaries
            # Calculate max boundaries across all batches
            num_boundaries_per_batch = document_boundary.sum(dim=1)  # [B]
            max_num_boundaries = int(num_boundaries_per_batch.max().item())

            # Calculate expanded sequence length based on max boundaries
            expanded_seq_len = T + max_num_boundaries * (W - 1)
            chunk_size = 1  # Default chunk size for padding
            expanded_padded_len = (
                1 + (expanded_seq_len - 1) // chunk_size
            ) * chunk_size

            # Create expanded tensor and compute mapping
            x_expanded = torch.zeros(
                B, expanded_padded_len, D, device=x.device, dtype=x.dtype
            )
            offsets = torch.cumsum(
                document_boundary * (W - 1), dim=1, dtype=torch.int64
            )  # [B, T]
            mapping = (
                torch.arange(T, dtype=torch.int64, device=x.device).unsqueeze(0)
                + offsets
            )  # [B, T]
            mapping = mapping.unsqueeze(-1).expand(B, -1, D)  # [B, T, D]
            x_expanded.scatter_(dim=-2, index=mapping, src=x)  # [B, L_exp, D]

            x = x_expanded
            T_expanded = expanded_padded_len
        else:
            T_expanded = T

        # Check if CUDA backend is callable based on the input
        # x is channel last! For causal_conv1d_fn we need the channel to be last dim and aligned
        # to 8 bytes --> stride(1) needs to be 8
        allowed_activation = (self.activation in ["silu", "swish"]) or (
            self.activation is None
        )
        aligned_strides = x.stride(1) % 8 == 0  # channel needs to be aligned to 8 bytes
        varlen_active = (cu_seqlens is not None) or (seq_idx is not None)
        use_cuda_kernel = (
            (causal_conv1d_fn is not None)
            and allowed_activation
            and (varlen_active or aligned_strides)
        )

        if self.backend == "triton":
            y, cache = causal_conv1d(
                x=x,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                residual=residual if document_boundary is None else None,
                initial_state=cache,
                output_final_state=output_final_state,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )
            if document_boundary is not None:
                # Extract back the original positions
                y = torch.gather(y, dim=1, index=mapping)
                if residual is not None:
                    y.add_(residual)
            return y, cache
        elif self.backend == "pytorch" or not use_cuda_kernel:
            # Force naive PyTorch implementation
            if self.backend != "pytorch" and not use_cuda_kernel:
                if not self._fallback_warned_forward:
                    warnings.warn(
                        "Falling back to naive PyTorch convolution implementation. "
                        "This is less efficient than the CUDA kernel. "
                        "Consider ensuring input tensors have 8-byte aligned strides, "
                        "or install causal-conv1d for better performance."
                        f"The input x is of shape: {x.shape} with strides {x.stride()}\n"
                        f"varlen_active: {varlen_active}\n"
                        f"aligned_strides: {aligned_strides}\n"
                        f"allowed_activation: {allowed_activation}\n",
                        stacklevel=2,
                    )
                    self._fallback_warned_forward = True

            # Note: varlen (cu_seqlens/seq_idx) is not supported in the naive path
            varlen_active_py = (cu_seqlens is not None) or (
                kwargs.get("seq_idx", None) is not None
            )
            if varlen_active_py:
                raise ValueError(
                    "The 'pytorch' backend does not support varlen (cu_seqlens/seq_idx)."
                )

            # Channel-first input
            x_cd = x.transpose(1, 2)  # [B, D, L_eff]

            # Prepend W-1 states from cache (or zeros) to emulate causal padding
            if W > 1:
                if cache is not None:
                    assert cache.shape[-1] == W, (
                        "Cache must have size equal to kernel_size"
                    )
                    pre = cache[:, :, -(W - 1) :]
                else:
                    pre = x_cd.new_zeros(x_cd.size(0), x_cd.size(1), W - 1)
                eff_in = torch.cat([pre, x_cd], dim=-1)
            else:
                eff_in = x_cd

            # Depthwise conv without padding; output length == L_eff
            y_cd = F.conv1d(
                eff_in,
                self.weight,
                self.bias,
                stride=self.stride[0],
                padding=0,
                dilation=self.dilation[0],
                groups=self.groups,
            )
            y = y_cd.transpose(1, 2)
            if self.activation is not None:
                y = F.silu(y)

            # Final cache is last W-1 elements of effective history
            if output_final_state:
                cache_out = x.new_zeros(N, D, W)
                cache_out.copy_(eff_in[:, :, -W:])
                cache = cache_out

            if document_boundary is not None:
                # Extract back the original positions
                y = torch.gather(y, dim=1, index=mapping)
            if residual is not None:
                y.add_(residual)
            return y, cache
        else:
            # CUDA kernel path
            x = rearrange(x, "b t d -> b d t")

            if cu_seqlens is not None and seq_idx is None:
                seq_idx = prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0)

            initial_state = None
            if cache is not None:
                B_cache, _, T_cache = cache.shape
                # To make causal-conv1d happy
                initial_state = (
                    cache[:, :, -(W - 1) :]  # [B, C, W-1]
                    .transpose(1, 2)  # [B, W-1, C]
                    .contiguous()  # [B, W-1, C] and stride(2)==1
                    .transpose(1, 2)  # [B, C, W-1] and stride(1)==1
                ).to(x.dtype)  # ensure it is of the same dtype as the input!
            else:
                # Ensure initial_state dtype matches input dtype when no cache is provided
                # causal-conv1d CUDA kernel expects initial_states.scalar_type == input_type
                # the transpose is because the causal_conv1d expects the dim 1 to be contiguous
                if W > 1:
                    initial_state = x.new_zeros(
                        x.size(0), W - 1, x.size(1), dtype=x.dtype, device=x.device
                    ).transpose(1, 2)  # [B, C, W-1]

            result = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
                seq_idx=seq_idx,
                initial_states=initial_state,
                return_final_states=output_final_state,
            )
            y, final_state = result if output_final_state else (result, None)
            y = rearrange(y, "b d t -> b t d")
            if output_final_state:
                # Build a W-length cache consistently across paths
                cache_out = x.new_zeros(N, D, W)
                if varlen_active:
                    # For varlen we rely on kernel-provided final_state (W-1) per sequence
                    T_for_cache = T_expanded if document_boundary is not None else T
                    copy_len = min(W - 1, T_for_cache)
                    if final_state is not None and copy_len > 0:
                        cache_out[:, :, -copy_len:].copy_(final_state[:, :, -copy_len:])
                else:
                    # Non-varlen: reconstruct final cache from previous (W-1) state and inputs
                    pre = (
                        initial_state
                        if initial_state is not None
                        else x.new_zeros(B, D, max(W - 1, 0))
                    )
                    eff_in = torch.cat([pre, x], dim=-1)
                    cache_out = eff_in[:, :, -W:].contiguous()
                if cache is not None:
                    cache.copy_(cache_out)
                else:
                    cache = cache_out

            if document_boundary is not None:
                # Extract back the original positions
                y = torch.gather(y, dim=1, index=mapping)

            if residual is not None:
                y.add_(residual)

            return y, cache

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        document_boundary: Optional[torch.Tensor] = None,
    ):
        """
        Performs a single step of boundary-aware causal convolution.

        When document_boundary is set, the conv_state is zeroed out to reset
        the causal history, ensuring no information leaks across document boundaries.

        Args:
            x: (B, 1, D) or (B, D) - Current input token embeddings
            residual: Residual tensor
            cache: (B, D, kernel_size) or (N, D, kernel_size) - Convolution state from previous tokens
            output_final_state: Whether to output final state
            cu_seqlens: Cumulative sequence lengths
            document_boundary: (B,) or (B, 1) - Binary indicator (1 = new document start)
        """
        if len(x.shape) == 2:
            B, D, W = *x.shape, self.kernel_size[0]
        else:
            B, _, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if output_final_state and cache is None:
            cache = x.new_zeros(N, W, D).transpose(
                1, 2
            )  # to ensure timestep have stride % 8 == 0

        # Handle document boundary by zeroing out cache
        if document_boundary is not None and cache is not None:
            # Ensure document_boundary has correct shape
            if document_boundary.dim() == 1:
                document_boundary = document_boundary.unsqueeze(-1)  # [B, 1]
            # Create boundary mask and zero out conv_state at document boundaries
            # This prevents information leakage from previous documents
            boundary_mask = document_boundary.view(N, 1, 1).expand(N, D, W).bool()
            cache = cache.masked_fill(boundary_mask, 0.0)

        # Check if we can use CUDA backedn
        # x is of shape [B, 1, D] or [B, D] and channels should be aligned to 8 bytes
        # --> we can check x.stride(0) % 8 == 0
        # NOTE: Unit tests pass also without the check on x strides!!??
        x_strides_ok = (
            x.stride(0) % 8 == 0
        )  # x is 1 timestep, we can use 0th dim for stride check
        # x_strides_ok = True
        # cache is of shape [B, D, W] --> we can check cache.stride(2) % 8 == 0?
        # NOTE: Unit tests pass also without any check on the cache strides!!
        # cache_strides_ok = cache.stride(2) % 8 == 0
        cache_strides_ok = True

        allowed_activation = (self.activation in ["silu", "swish"]) or (
            self.activation is None
        )
        use_cuda_update = (
            self.backend == "cuda"
            and (causal_conv1d_update_cuda is not None)
            and allowed_activation
            and cache_strides_ok
            and x_strides_ok
        )

        # NOTE: we follow the fast mode that updates the cache in-place
        if self.backend == "triton":
            return causal_conv1d_update(
                x=x,
                cache=cache,
                residual=residual,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
        elif self.backend == "cuda" and use_cuda_update:
            shape = x.shape
            x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
            cache = cache.to(x.dtype)  # such that we use the autocast precision

            # Let's force differnt strides to see if it still works or not
            # x = x.transpose(0, 1).contiguous().transpose(0, 1)  # does not impact output?
            # cache = cache.transpose(1, 2).contiguous().transpose(1, 2)  # also works
            # cache = cache.transpose(0, 1).contiguous().transpose(0, 1)  # also works

            y = causal_conv1d_update_cuda(
                x=x,
                conv_state=cache,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
            y = y.view(shape)
            if residual is not None:
                y.add_(residual)
            return y, cache
        else:
            if not self._fallback_warned_step and self.backend != "pytorch":
                warnings.warn(
                    "Falling back to naive PyTorch step implementation. "
                    "This is less efficient than the CUDA kernel. "
                    "Consider ensuring cache tensors have 8-byte aligned strides, "
                    "or install causal-conv1d for better performance."
                    f"The input x is of shape: {x.shape} with strides {x.stride()}\n"
                    f"cache_strides_ok: {cache_strides_ok}\n"
                    f"x_strides_ok: {x_strides_ok}\n"
                    f"allowed_activation: {allowed_activation}\n",
                    stacklevel=2,
                )
                self._fallback_warned_step = True

            shape = x.shape
            x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
            cache.copy_(torch.roll(cache, shifts=-1, dims=-1))  # Update state (B D W)
            cache = cache.to(x.dtype)
            cache[:, :, -1] = x
            y = torch.sum(
                cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.bias is not None:
                y = y + self.bias
            if self.activation is not None:
                y = F.silu(y).to(y.dtype)
            y = y.view(shape)
            if residual is not None:
                y.add_(residual)
            return y, cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size
