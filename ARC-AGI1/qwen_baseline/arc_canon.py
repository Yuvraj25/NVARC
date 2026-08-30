"""Residual Canon-AC building blocks for the ARC Qwen experiments.

The operator follows Allen-Zhu's released Canon implementation at the two
clean Transformer insertion points:

* A: after the input RMSNorm and before self attention;
* C: after the post-attention RMSNorm and before the MLP.

Unlike the reference CUDA implementation, this module deliberately uses
plain PyTorch.  That keeps the first controlled experiment independent of a
new compiled dependency.  The implementation is causal, returns immutable
incremental states for DFS sibling safety, and supports explicit sequence
boundaries for packed batches.

Reference implementation (Apache-2.0):
https://github.com/zhuzeyuan/PhysicsLM4/tree/main/huggingface
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn


class ResidualCanon1d(nn.Module):
    """Depthwise causal local mixer with an explicit residual connection.

    ``kernel_size=4`` means the current representation and the preceding
    three representations, matching the paper.  Missing predecessors at a
    sequence boundary are zeros.  Weights are zero-initialized when adding
    Canon to an already-trained model, so installation is exactly identity.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        *,
        zero_init: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.hidden_size = int(hidden_size)
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(
            torch.empty(self.kernel_size, self.hidden_size, dtype=dtype, device=device)
        )
        self.reset_parameters(zero_init=zero_init)

    def reset_parameters(self, *, zero_init: bool = True) -> None:
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            # This matches the scale of the reference depthwise Conv1d
            # initializer.  Zero init is the intended pretrained-model path.
            nn.init.uniform_(
                self.weight,
                -self.kernel_size**-0.5,
                self.kernel_size**-0.5,
            )

    @property
    def history_length(self) -> int:
        return self.kernel_size - 1

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Return recent-first zero history: ``[batch, K-1, hidden]``."""
        return torch.zeros(
            batch_size,
            self.history_length,
            self.hidden_size,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply residual Canon to ``[batch, sequence, hidden]``.

        ``sequence_ids`` prevents mixing across packed-example boundaries.
        ``valid_mask`` prevents padding representations from contributing.
        Neither is needed for the production global curriculum's batch-size-1
        un-packed records, but supporting them makes the boundary semantics
        explicit and testable.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected [batch, sequence, {self.hidden_size}], got "
                f"{tuple(hidden_states.shape)}"
            )
        batch_size, sequence_length, _ = hidden_states.shape
        if valid_mask is not None and valid_mask.shape != (batch_size, sequence_length):
            raise ValueError("valid_mask must have shape [batch, sequence]")
        if sequence_ids is not None and sequence_ids.shape != (batch_size, sequence_length):
            raise ValueError("sequence_ids must have shape [batch, sequence]")

        source = hidden_states
        if valid_mask is not None:
            source = source * valid_mask.to(dtype=source.dtype).unsqueeze(-1)

        mixed = source * self.weight[0]
        for lag in range(1, self.kernel_size):
            if lag >= sequence_length:
                break
            zeros = source.new_zeros(batch_size, lag, self.hidden_size)
            shifted = torch.cat((zeros, source[:, :-lag]), dim=1)
            if sequence_ids is not None:
                same_sequence = sequence_ids[:, lag:].eq(sequence_ids[:, :-lag])
                boundary_mask = torch.cat(
                    (
                        torch.zeros(
                            batch_size,
                            lag,
                            dtype=torch.bool,
                            device=sequence_ids.device,
                        ),
                        same_sequence,
                    ),
                    dim=1,
                )
                shifted = shifted * boundary_mask.to(shifted.dtype).unsqueeze(-1)
            mixed = mixed + shifted * self.weight[lag]
        # Prefill needs only the final K-1 inputs.  Detaching prevents this
        # diagnostic cache from retaining a training graph.
        if self.history_length:
            available = min(sequence_length, self.history_length)
            recent = source[:, sequence_length - available :].flip(1).detach()
            if available < self.history_length:
                recent = torch.cat(
                    (
                        recent,
                        source.new_zeros(
                            batch_size,
                            self.history_length - available,
                            self.hidden_size,
                        ),
                    ),
                    dim=1,
                )
            self._last_prefill_state = recent
        else:
            self._last_prefill_state = source.new_zeros(batch_size, 0, self.hidden_size)
        return hidden_states + mixed

    def step(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        *,
        reset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Canon to one token and return a new, non-mutating state.

        ``state[:, 0]`` is the immediately preceding representation.  The
        input state is never modified, which allows DFS siblings to share a
        parent state safely.
        """
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(1)
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (1, self.hidden_size):
            raise ValueError(
                f"Expected [batch, 1, {self.hidden_size}], got {tuple(hidden_states.shape)}"
            )
        expected_state = (
            hidden_states.shape[0],
            self.history_length,
            self.hidden_size,
        )
        if state.shape != expected_state:
            raise ValueError(f"Expected state {expected_state}, got {tuple(state.shape)}")
        if reset is not None:
            if reset.shape != (hidden_states.shape[0],):
                raise ValueError("reset must have shape [batch]")
            state = state * (~reset.bool()).to(state.dtype).view(-1, 1, 1)

        current = hidden_states[:, 0]
        mixed = current * self.weight[0]
        for lag in range(1, self.kernel_size):
            mixed = mixed + state[:, lag - 1] * self.weight[lag]
        output = hidden_states + mixed.unsqueeze(1)
        if self.history_length == 0:
            next_state = state
        else:
            next_state = torch.cat((current.unsqueeze(1), state[:, :-1]), dim=1)
        return output, next_state

    def step_sequence(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply Canon causally to a cached token block.

        The returned states correspond to every accepted prefix of the block.
        Multi-token DFS needs these snapshots because a q9 draft may accept
        only its first ``m`` tokens before backtracking to a sibling.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected [batch, sequence, {self.hidden_size}], got "
                f"{tuple(hidden_states.shape)}"
            )
        batch_size, sequence_length, _ = hidden_states.shape
        expected_state = (batch_size, self.history_length, self.hidden_size)
        if state.shape != expected_state:
            raise ValueError(f"Expected state {expected_state}, got {tuple(state.shape)}")

        # Apply all q cached positions at once.  The previous implementation
        # called ``step`` q times, which placed two Python loops per decoder
        # layer on every q9 DFS model call.  That dominated Canon inference.
        mixed = hidden_states * self.weight[0]
        for lag in range(1, self.kernel_size):
            prefix_count = min(lag, sequence_length)
            prefix = state[:, lag - prefix_count : lag].flip(1)
            if sequence_length > lag:
                shifted = torch.cat(
                    (prefix, hidden_states[:, : sequence_length - lag]), dim=1
                )
            else:
                shifted = prefix
            mixed = mixed + shifted * self.weight[lag]

        if self.history_length == 0:
            states = tuple(state for _ in range(sequence_length))
        else:
            # Chronological timeline -> rolling windows -> recent-first Canon
            # histories for every accepted prefix of this q-block.
            timeline = torch.cat((state.flip(1), hidden_states), dim=1)
            windows = timeline.unfold(1, self.history_length, 1)[:, 1:]
            state_matrix = windows.permute(0, 1, 3, 2).flip(2)
            states = tuple(state_matrix[:, offset] for offset in range(sequence_length))
        return hidden_states + mixed, states


@dataclass(frozen=True)
class CanonACState:
    """Immutable branch-local A/C histories, indexed by decoder layer."""

    a: tuple[torch.Tensor, ...]
    c: tuple[torch.Tensor, ...]

    def select(self, indices: torch.Tensor | Sequence[int]) -> "CanonACState":
        """Select/reorder batch lanes without modifying the parent state."""
        return CanonACState(
            a=tuple(tensor[indices] for tensor in self.a),
            c=tuple(tensor[indices] for tensor in self.c),
        )


@dataclass(frozen=True)
class CanonDFSCache:
    """The KV cache and matching branch-local Canon histories."""

    kv: object
    canon: CanonACState


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Find Qwen/Llama decoder layers through optional PEFT wrappers."""
    candidates = [model]
    visited: set[int] = set()
    while candidates:
        current = candidates.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        layers = getattr(current, "layers", None)
        if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers):
            if hasattr(layers[0], "self_attn") and hasattr(layers[0], "mlp"):
                return layers
        for name in ("model", "base_model"):
            child = getattr(current, name, None)
            if isinstance(child, nn.Module):
                candidates.append(child)
    raise ValueError("Could not find Qwen/Llama decoder layers")


def add_canon_ac_modules(
    model: nn.Module,
    *,
    kernel_size: int = 4,
    zero_init: bool = True,
) -> Sequence[nn.Module]:
    """Attach Canon-A and Canon-C modules without changing fused kernels."""
    layers = _decoder_layers(model)
    for layer in layers:
        if hasattr(layer, "canonA") or hasattr(layer, "canonC"):
            raise ValueError("Canon modules are already installed")
        reference = layer.input_layernorm.weight
        layer.canonA = ResidualCanon1d(
            reference.numel(),
            kernel_size,
            zero_init=zero_init,
            dtype=reference.dtype,
            device=reference.device,
        )
        layer.canonC = ResidualCanon1d(
            reference.numel(),
            kernel_size,
            zero_init=zero_init,
            dtype=reference.dtype,
            device=reference.device,
        )
    return layers


def canon_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for layer in _decoder_layers(model):
        yield from layer.canonA.parameters()
        yield from layer.canonC.parameters()


def new_canon_ac_state(model: nn.Module, batch_size: int) -> CanonACState:
    layers = _decoder_layers(model)
    a = tuple(
        layer.canonA.initial_state(
            batch_size,
            dtype=layer.canonA.weight.dtype,
            device=layer.canonA.weight.device,
        )
        for layer in layers
    )
    c = tuple(
        layer.canonC.initial_state(
            batch_size,
            dtype=layer.canonC.weight.dtype,
            device=layer.canonC.weight.device,
        )
        for layer in layers
    )
    return CanonACState(a=a, c=c)


@dataclass
class CanonACHooks:
    """The removable A/C pre-hooks used by full-sequence training/prefill."""

    handles: list[torch.utils.hooks.RemovableHandle]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _replace_hidden_input(
    args: tuple,
    kwargs: dict,
    replacement: torch.Tensor,
) -> tuple[tuple, dict]:
    if "hidden_states" in kwargs:
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = replacement
        return args, kwargs
    if not args:
        raise ValueError("Expected hidden states as a positional or named argument")
    return (replacement, *args[1:]), kwargs


def install_canon_ac_training_hooks(model: nn.Module) -> CanonACHooks:
    """Insert Canon-AC around existing fused attention/MLP module calls.

    These hooks cover training and uncached prefill without changing either
    fused kernel.  Unsloth's cached-generation fast path calls its kernels as
    plain functions and therefore needs the separate explicit incremental
    adapter; relying on these hooks for DFS would silently skip Canon-C.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer in _decoder_layers(model):
        if not hasattr(layer, "canonA") or not hasattr(layer, "canonC"):
            raise ValueError("Attach Canon modules before installing hooks")

        def attention_hook(module, args, kwargs, *, decoder_layer=layer):
            if getattr(decoder_layer, "_arc_canon_explicit_step", False):
                return args, kwargs
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                raise ValueError("Attention call did not supply hidden_states")
            return _replace_hidden_input(args, kwargs, decoder_layer.canonA(hidden))

        def mlp_hook(module, args, kwargs, *, decoder_layer=layer):
            if getattr(decoder_layer, "_arc_canon_explicit_step", False):
                return args, kwargs
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                raise ValueError("MLP call did not supply hidden_states")
            return _replace_hidden_input(args, kwargs, decoder_layer.canonC(hidden))

        handles.append(
            layer.self_attn.register_forward_pre_hook(attention_hook, with_kwargs=True)
        )
        handles.append(layer.mlp.register_forward_pre_hook(mlp_hook, with_kwargs=True))
    return CanonACHooks(handles)


def canon_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for index, layer in enumerate(_decoder_layers(model)):
        state[f"layers.{index}.canonA.weight"] = layer.canonA.weight.detach().cpu()
        state[f"layers.{index}.canonC.weight"] = layer.canonC.weight.detach().cpu()
    return state


def save_canon_state(model: nn.Module, path: str | Path) -> None:
    """Save the tiny Canon sidecar independently of the PEFT adapter."""
    torch.save(canon_state_dict(model), Path(path))


def load_canon_state_dict(
    model: nn.Module,
    payload: dict[str, torch.Tensor],
) -> None:
    """Restore Canon weights from an in-memory CPU state dictionary."""
    expected = set(canon_state_dict(model))
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(f"Canon state mismatch: missing={missing[:4]} extra={extra[:4]}")
    for index, layer in enumerate(_decoder_layers(model)):
        for site in ("A", "C"):
            module = getattr(layer, f"canon{site}")
            key = f"layers.{index}.canon{site}.weight"
            module.weight.data.copy_(payload[key].to(module.weight.device, module.weight.dtype))


def load_canon_state(model: nn.Module, path: str | Path) -> None:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    load_canon_state_dict(model, payload)


def captured_canon_ac_state(model: nn.Module) -> CanonACState:
    """Collect the small histories captured during the latest full prefill."""
    layers = _decoder_layers(model)
    missing = [
        f"{index}{site}"
        for index, layer in enumerate(layers)
        for site in ("A", "C")
        if not hasattr(getattr(layer, f"canon{site}"), "_last_prefill_state")
    ]
    if missing:
        raise RuntimeError(f"Canon prefill state was not captured for {missing[:6]}")
    return CanonACState(
        a=tuple(layer.canonA._last_prefill_state for layer in layers),
        c=tuple(layer.canonC._last_prefill_state for layer in layers),
    )


def _causal_lm_and_backbone(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    candidates = [model]
    visited: set[int] = set()
    while candidates:
        current = candidates.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        backbone = getattr(current, "model", None)
        if (
            hasattr(current, "lm_head")
            and isinstance(backbone, nn.Module)
            and hasattr(backbone, "embed_tokens")
            and hasattr(backbone, "layers")
        ):
            return current, backbone
        for name in ("model", "base_model"):
            child = getattr(current, name, None)
            if isinstance(child, nn.Module):
                candidates.append(child)
    raise ValueError("Could not locate the causal-LM head and Qwen/Llama backbone")


def prepare_canon_inference(model: nn.Module) -> None:
    """Disable only Unsloth's decoder shortcut that bypasses Canon-C hooks."""
    for layer in _decoder_layers(model):
        if hasattr(layer, "_flag_for_generation"):
            delattr(layer, "_flag_for_generation")
    model._arc_canon_enabled = True


def canon_prefill(model: nn.Module, input_ids: torch.Tensor):
    """Run the ordinary fused prefill and return its KV plus Canon state."""
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    return outputs, CanonDFSCache(
        kv=outputs.past_key_values,
        canon=captured_canon_ac_state(model),
    )


def canon_cached_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cache: CanonDFSCache,
):
    """One-token cached Qwen step with explicit branch-local Canon-AC state.

    Attention and MLP modules remain the existing Unsloth modules.  This code
    only recreates the lightweight decoder-block wiring so Canon can be placed
    immediately after both RMSNorm operations.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] < 1:
        raise ValueError("Canon cached inference requires at least one token")
    from unsloth.models.llama import (
        dtype_from_config,
        fast_rms_layernorm_inference,
        fast_swiglu_inference,
    )
    from unsloth.models.qwen3 import Qwen3Attention_fast_forward_inference
    from unsloth_zoo.utils import _get_dtype

    causal_lm, backbone = _causal_lm_and_backbone(model)
    layers = _decoder_layers(model)
    if len(cache.kv) != len(layers):
        raise ValueError("KV cache layer count does not match the decoder")
    if len(cache.canon.a) != len(layers) or len(cache.canon.c) != len(layers):
        raise ValueError("Canon cache layer count does not match the decoder")

    hidden = backbone.embed_tokens(input_ids)
    hidden = hidden.to(_get_dtype(dtype_from_config(causal_lm.config)))
    batch_size, query_length, hidden_size = hidden.shape
    intermediate_size = causal_lm.config.intermediate_size
    residual = torch.empty(
        (batch_size, query_length, hidden_size),
        dtype=torch.float32,
        device=hidden.device,
    )
    work = torch.empty(
        (2, batch_size, query_length, hidden_size),
        dtype=torch.float32,
        device=hidden.device,
    )
    work_one, work_two = work[0], work[1]
    variance = torch.empty(
        (batch_size, query_length, 1),
        dtype=torch.float32,
        device=hidden.device,
    )
    mlp_work = torch.empty(
        (2, batch_size, query_length, intermediate_size),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    temp_gate, temp_up = mlp_work[0], mlp_work[1]
    next_kv = []
    next_a = []
    next_c = []
    prefix_a = [[] for _ in range(query_length)]
    prefix_c = [[] for _ in range(query_length)]
    for index, layer in enumerate(layers):
        residual.copy_(hidden)
        normalized = fast_rms_layernorm_inference(
            layer.input_layernorm,
            hidden,
            XX=work_one,
            XX2=work_two,
            variance=variance,
        )
        normalized, states_a = layer.canonA.step_sequence(
            normalized, cache.canon.a[index]
        )
        state_a = states_a[-1]
        prior_key = cache.kv[index][0]
        kv_sequence_length = prior_key.shape[-2] + input_ids.shape[1]
        rotary_sequence_length = max(
            kv_sequence_length,
            int(position_ids.max().item()) + 1,
        )
        attention_hidden, present_view = Qwen3Attention_fast_forward_inference(
            layer.self_attn,
            hidden_states=normalized,
            past_key_value=cache.kv[index],
            position_ids=position_ids,
            attention_mask=None,
            # Seed the module-local paged buffer once from the ordinary
            # prefill KV.  Recursive DFS is depth-first: backtracking to a
            # sibling retains the shared prefix and overwrites only positions
            # at and after that parent's sequence length.
            do_prefill=not hasattr(layer.self_attn, "paged_attention"),
            rotary_seq_len=rotary_sequence_length,
        )
        # Keep the short views into the single paged buffer.  Cloning the full
        # 4k-8k-token KV at every recursion depth exhausts a 24 GB L4.
        present = present_view
        hidden = attention_hidden
        hidden += residual

        residual.copy_(hidden)
        normalized = fast_rms_layernorm_inference(
            layer.post_attention_layernorm,
            hidden,
            XX=work_one,
            XX2=work_two,
            variance=variance,
        )
        normalized, states_c = layer.canonC.step_sequence(
            normalized, cache.canon.c[index]
        )
        state_c = states_c[-1]
        layer._arc_canon_explicit_step = True
        try:
            hidden = fast_swiglu_inference(
                layer.mlp,
                normalized,
                temp_gate=temp_gate,
                temp_up=temp_up,
            )
        finally:
            layer._arc_canon_explicit_step = False
        hidden += residual
        next_kv.append(present)
        next_a.append(state_a)
        next_c.append(state_c)
        for offset in range(query_length):
            prefix_a[offset].append(states_a[offset])
            prefix_c[offset].append(states_c[offset])

    hidden = fast_rms_layernorm_inference(
        backbone.norm,
        hidden,
        XX=work_one,
        XX2=work_two,
        variance=variance,
    )
    logits = causal_lm.lm_head(hidden.to(causal_lm.lm_head.weight.dtype))
    return SimpleNamespace(
        logits=logits,
        past_key_values=CanonDFSCache(
            kv=next_kv,
            canon=CanonACState(a=tuple(next_a), c=tuple(next_c)),
        ),
        canon_prefix_states=tuple(
            CanonACState(a=tuple(a), c=tuple(c))
            for a, c in zip(prefix_a, prefix_c)
        ),
    )
