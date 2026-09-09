# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-Omni Thinker text/image training on verl's VeOmni FSDP2/EP engine.

VeOmni owns model construction, sharding, optimization and checkpoint loading.
This adapter supplies prompt-only modality masks and converts fused expert
weights to the per-expert layout used by vLLM-Omni for full-weight updates.
"""

import inspect
import logging
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from tensordict import TensorDict
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead
from verl.workers.engine.veomni.utils import MOE_PARAM_HANDERS

from verl_omni.workers.engine.packed_prompt_region import build_prompt_region_mask

logger = logging.getLogger(__name__)

_MODEL_TYPE = "qwen3_omni_moe"
_MODALITY_TOKEN_ATTRS = {
    "image": "image_token_id",
    "video": "video_token_id",
    "audio": "audio_token_id",
}
_VEOMNI_PLACEHOLDER_IDS = {
    "image": -200,
    "video": -300,
    "audio": -400,
}
# Feature tensor each modality arrives under in verl's packed model inputs.
_MODALITY_TENSOR_KEYS = {
    "image": "pixel_values",
    "video": "pixel_values_videos",
    "audio": "input_features",
}


def _is_qwen3_omni(module: Any) -> bool:
    return getattr(getattr(module, "config", None), "model_type", None) == _MODEL_TYPE


def patch_veomni_causal_mask_kwargs() -> bool:
    """Drop ``create_causal_mask`` kwargs the pinned Transformers no longer takes.

    veomni's generated ``qwen3_omni_moe`` modeling targets an older
    ``create_causal_mask`` signature that still accepted ``cache_position``.
    Transformers removed it, so the Thinker forward raises ``TypeError`` on every
    log-prob and training pass. Rebind the module-level symbol to a wrapper that
    removes only the obsolete ``cache_position`` argument.

    Returns True when the shim was installed, False when it is unnecessary.
    """

    from transformers.masking_utils import create_causal_mask
    from veomni.models.transformers.qwen3_omni_moe.generated import (
        patched_modeling_qwen3_omni_moe_gpu as veomni_modeling,
    )

    supported = set(inspect.signature(create_causal_mask).parameters)
    if "cache_position" in supported:
        return False
    if getattr(veomni_modeling.create_causal_mask, "_verl_omni_kwarg_shim", False):
        return False

    def create_causal_mask_compat(**kwargs):
        kwargs.pop("cache_position", None)
        return create_causal_mask(**kwargs)

    create_causal_mask_compat._verl_omni_kwarg_shim = True
    veomni_modeling.create_causal_mask = create_causal_mask_compat
    logger.info("Patched VeOmni Qwen3-Omni create_causal_mask to omit cache_position.")
    return True


def _get_thinker_config(config: Any) -> Any:
    return getattr(config, "thinker_config", config)


def build_modality_masks(
    input_ids: torch.Tensor,
    config: Any,
    available: Iterable[str] = (),
    prompt_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build the placeholder masks veomni's Thinker forward asserts on.

    The generated forward pops ``image_mask`` / ``video_mask`` / ``audio_mask``
    and ``masked_scatter``s encoder outputs onto exactly those positions, so a
    modality whose feature tensor is in the batch needs a mask over its
    placeholder tokens and every other modality needs an all-false mask.

    Args:
        input_ids: packed token ids, shape ``(1, total_nnz)``.
        config: Qwen3-Omni config (top level or thinker level).
        available: modalities whose feature tensor is present in this batch.
        prompt_mask: optional ``(1, total_nnz)`` marker of the prompt region.
            Placeholder ids outside it were sampled by the policy and stay text.

    Returns:
        ``{"image_mask": ..., "video_mask": ..., "audio_mask": ...}``.
    """

    thinker_config = _get_thinker_config(config)
    available = set(available)
    masks: dict[str, torch.Tensor] = {}

    for modality, token_attr in _MODALITY_TOKEN_ATTRS.items():
        token_ids = {_VEOMNI_PLACEHOLDER_IDS[modality]}
        token_id = getattr(thinker_config, token_attr, None)
        if token_id is not None:
            token_ids.add(int(token_id))

        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for candidate in token_ids:
            mask |= input_ids == candidate
        if prompt_mask is not None:
            mask &= prompt_mask

        if modality in available:
            if not bool(mask.any()):
                raise ValueError(
                    f"Batch carries {modality} features but no {modality} placeholder token "
                    f"({sorted(token_ids)}) — the processor and the model config disagree."
                )
            masks[f"{modality}_mask"] = mask
        else:
            if prompt_mask is not None and bool(mask.any()):
                raise ValueError(f"Prompt contains {modality} placeholder tokens but no {modality} features.")
            masks[f"{modality}_mask"] = torch.zeros_like(input_ids, dtype=torch.bool)

    return masks


def freeze_modality_towers(module: torch.nn.Module) -> tuple[str, ...]:
    """Freeze the encoder towers, while training the full text backbone.

    The vision tower still runs in the forward pass for image inputs; freezing
    only keeps its parameters out of the optimizer so both backends optimize the
    same parameter set.
    """

    thinker = getattr(module, "thinker", module)
    frozen: list[str] = []
    for tower_name in ("visual", "audio_tower"):
        tower = getattr(thinker, tower_name, None)
        if tower is None:
            raise AttributeError(f"Qwen3-Omni Thinker is missing expected {tower_name!r} module")
        tower.requires_grad_(False)
        frozen.append(tower_name)
    return tuple(frozen)


def map_qwen3_omni_moe_param(
    name: str,
    tensor: torch.Tensor,
    expert_id_base: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Expand a fused expert stack into Hugging Face per-expert weights.

    Follows verl's ``default_moe_param_handler`` contract: dim 0 of *tensor* is a
    contiguous run of experts and ``expert_id_base`` is the global expert id of
    slice 0 (``ep_rank * experts_per_rank`` for a local shard, ``0`` for an
    already-global stack). verl also probes this handler with a single dummy row
    to enumerate its output slots, so it must stay valid for any leading dim.
    """

    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3-D fused expert tensor for {name!r}, got shape={tuple(tensor.shape)}")

    if "gate_up_proj" in name:
        if tensor.shape[1] % 2:
            raise ValueError(f"gate_up_proj dimension 1 must be even, got shape={tuple(tensor.shape)}")
        gate, up = tensor.chunk(2, dim=1)
        projections = {
            name.replace("gate_up_proj", "gate_proj"): gate,
            name.replace("gate_up_proj", "up_proj"): up,
        }
    else:
        projections = {name: tensor}

    for projection_name, projection in projections.items():
        for local_expert_idx in range(tensor.shape[0]):
            global_expert_idx = expert_id_base + local_expert_idx
            expert_name = projection_name.replace(
                "mlp.experts.",
                f"mlp.experts.{global_expert_idx}.",
                1,
            )
            if not expert_name.endswith(".weight"):
                expert_name = f"{expert_name}.weight"
            yield expert_name, projection[local_expert_idx].contiguous()


@EngineRegistry.register(model_type="omni_model", backend="veomni", device="cuda")
class Qwen3OmniThinkerVeOmniEngine(VeOmniEngineWithLMHead):
    """VeOmni engine for Qwen3-Omni Thinker with frozen modality towers."""

    def _get_model_config_path(self):
        """Validate supported architecture and install the Transformers compatibility shim."""
        if self.model_config.hf_config.model_type != _MODEL_TYPE or self.model_config.model_stage != "thinker":
            raise NotImplementedError("The Omni VeOmni engine currently supports Qwen3-Omni Thinker only.")
        if self.engine_config.ulysses_parallel_size != 1:
            raise NotImplementedError("Qwen3-Omni Thinker with VeOmni requires ulysses_parallel_size=1.")
        if not self.model_config.use_remove_padding:
            raise NotImplementedError("Qwen3-Omni Thinker with VeOmni requires use_remove_padding=True.")
        if self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0:
            raise NotImplementedError("Qwen3-Omni Thinker with VeOmni supports full-parameter training only.")
        patch_veomni_causal_mask_kwargs()
        return super()._get_model_config_path()

    def _apply_veomni_input_transforms(self, model_inputs: dict, micro_batch: TensorDict):
        """Add masks for prompt placeholders, retaining response token embeddings."""
        super()._apply_veomni_input_transforms(model_inputs, micro_batch)
        available = {modality for modality, key in _MODALITY_TENSOR_KEYS.items() if model_inputs.get(key) is not None}
        unsupported = available - {"image"}
        if unsupported:
            raise NotImplementedError(
                f"Qwen3-Omni Thinker with VeOmni does not support {sorted(unsupported)} inputs yet."
            )
        prompt_mask = build_prompt_region_mask(micro_batch, model_inputs["input_ids"].size(-1))
        if prompt_mask is None:
            raise ValueError(
                "Qwen3-Omni VeOmni requires jagged input_ids and response_mask to locate prompt placeholders."
            )
        model_inputs.update(
            build_modality_masks(model_inputs["input_ids"], self.module.config, available, prompt_mask=prompt_mask)
        )

    def _build_optimizer(self, module):
        if _is_qwen3_omni(module):
            frozen = freeze_modality_towers(module)
            logger.info("Frozen Qwen3-Omni modality towers before optimizer creation: %s", frozen)
        return super()._build_optimizer(module)


MOE_PARAM_HANDERS[_MODEL_TYPE] = map_qwen3_omni_moe_param
