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

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# Load the adapter against a stub of the optional distributed engine. The
# masks and tensor conversions use real torch operations on CPU.
_ROOT = Path(__file__).resolve().parents[2]


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, _ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompt_module = _load_source("packed_prompt_region", "verl_omni/workers/engine/packed_prompt_region.py")


class _BaseEngine:
    def _apply_veomni_input_transforms(self, model_inputs, micro_batch):
        model_inputs["base_transform_called"] = True

    def _get_model_config_path(self):
        return self.model_config.local_hf_config_path

    def _build_optimizer(self, module):
        return [p for p in module.parameters() if p.requires_grad]


_registry = MagicMock()
_registry.register.return_value = lambda cls: cls
_handlers = {}
with patch.dict(
    sys.modules,
    {
        "verl.workers.engine.base": SimpleNamespace(EngineRegistry=_registry),
        "verl.workers.engine.veomni.transformer_impl": SimpleNamespace(VeOmniEngineWithLMHead=_BaseEngine),
        "verl.workers.engine.veomni.utils": SimpleNamespace(MOE_PARAM_HANDERS=_handlers),
        "verl_omni.workers.engine.packed_prompt_region": prompt_module,
    },
):
    adapter = _load_source("qwen3_omni_veomni", "verl_omni/workers/engine/veomni/qwen3_omni_thinker_impl.py")

build_modality_masks = adapter.build_modality_masks
freeze_modality_towers = adapter.freeze_modality_towers
map_qwen3_omni_moe_param = adapter.map_qwen3_omni_moe_param
Qwen3OmniThinkerVeOmniEngine = adapter.Qwen3OmniThinkerVeOmniEngine


def _config():
    return SimpleNamespace(
        model_type="qwen3_omni_moe",
        thinker_config=SimpleNamespace(
            image_token_id=151655,
            video_token_id=151656,
            audio_token_id=151675,
        ),
    )


def test_build_modality_masks_text_only():
    input_ids = torch.tensor([[1, 2, 3]])

    masks = build_modality_masks(input_ids, _config())

    assert set(masks) == {"image_mask", "video_mask", "audio_mask"}
    assert all(mask.dtype == torch.bool for mask in masks.values())
    assert all(mask.shape == input_ids.shape for mask in masks.values())
    assert not any(mask.any() for mask in masks.values())


def test_build_modality_masks_marks_image_placeholders():
    input_ids = torch.tensor([[1, 151655, 151655, 2]])

    masks = build_modality_masks(input_ids, _config(), available={"image"})

    torch.testing.assert_close(masks["image_mask"], torch.tensor([[False, True, True, False]]))
    assert not masks["video_mask"].any()
    assert not masks["audio_mask"].any()


def test_build_modality_masks_accepts_veomni_image_placeholder():
    masks = build_modality_masks(torch.tensor([[1, -200, 2]]), _config(), available={"image"})

    torch.testing.assert_close(masks["image_mask"], torch.tensor([[False, True, False]]))


def test_build_modality_masks_preserves_sampled_tokens_without_features():
    masks = build_modality_masks(
        torch.tensor([[1, 151655, 2]]), _config(), prompt_mask=torch.tensor([[True, False, False]])
    )
    assert not masks["image_mask"].any()


def test_build_modality_masks_rejects_image_features_without_tokens():
    with pytest.raises(ValueError, match="placeholder token"):
        build_modality_masks(torch.tensor([[1, 2, 3]]), _config(), available={"image"})


def test_build_modality_masks_ignores_placeholders_sampled_into_the_response():
    # A random policy can emit <|image_pad|> itself. Scattering the encoder
    # output onto those positions overruns it, so only the prompt copy counts.
    input_ids = torch.tensor([[1, 151655, 2, 151655, 3]])
    prompt_mask = torch.tensor([[True, True, True, False, False]])

    masks = build_modality_masks(input_ids, _config(), available={"image"}, prompt_mask=prompt_mask)

    torch.testing.assert_close(masks["image_mask"], torch.tensor([[False, True, False, False, False]]))


def test_build_modality_masks_rejects_a_prompt_that_lost_its_placeholder():
    # Features present but the only placeholder sits in the response: the batch
    # is malformed, and silently dropping the image would corrupt training.
    with pytest.raises(ValueError, match="no image placeholder token"):
        build_modality_masks(
            torch.tensor([[1, 2, 151655]]),
            _config(),
            available={"image"},
            prompt_mask=torch.tensor([[True, True, False]]),
        )


@pytest.mark.parametrize(
    ("token_id", "modality"),
    [
        (151656, "video"),
        (151675, "audio"),
        (-300, "video"),
        (-400, "audio"),
    ],
)
def test_build_modality_masks_rejects_missing_prompt_features(token_id, modality):
    with pytest.raises(ValueError, match=f"no {modality} features"):
        build_modality_masks(
            torch.tensor([[151655, token_id, 2]]),
            _config(),
            available={"image"},
            prompt_mask=torch.ones(1, 3, dtype=torch.bool),
        )


def test_freeze_modality_towers():
    module = torch.nn.Module()
    module.thinker = torch.nn.Module()
    module.thinker.visual = torch.nn.Linear(2, 2)
    module.thinker.audio_tower = torch.nn.Linear(2, 2)
    module.thinker.model = torch.nn.Linear(2, 2)

    frozen = freeze_modality_towers(module)

    assert frozen == ("visual", "audio_tower")
    assert not any(param.requires_grad for param in module.thinker.visual.parameters())
    assert not any(param.requires_grad for param in module.thinker.audio_tower.parameters())
    assert all(param.requires_grad for param in module.thinker.model.parameters())


def test_map_qwen3_omni_moe_param_splits_gate_up_and_offsets_expert_ids():
    tensor = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    name = "thinker.model.layers.0.mlp.experts.gate_up_proj"

    mapped = dict(map_qwen3_omni_moe_param(name, tensor, expert_id_base=2))

    assert set(mapped) == {
        "thinker.model.layers.0.mlp.experts.2.gate_proj.weight",
        "thinker.model.layers.0.mlp.experts.3.gate_proj.weight",
        "thinker.model.layers.0.mlp.experts.2.up_proj.weight",
        "thinker.model.layers.0.mlp.experts.3.up_proj.weight",
    }
    torch.testing.assert_close(
        mapped["thinker.model.layers.0.mlp.experts.2.gate_proj.weight"],
        tensor[0, :2],
    )
    torch.testing.assert_close(
        mapped["thinker.model.layers.0.mlp.experts.3.up_proj.weight"],
        tensor[1, 2:],
    )


def test_map_qwen3_omni_moe_param_maps_down_projection():
    tensor = torch.arange(2 * 3 * 2).reshape(2, 3, 2)
    name = "thinker.model.layers.0.mlp.experts.down_proj"

    mapped = dict(map_qwen3_omni_moe_param(name, tensor, expert_id_base=0))

    assert list(mapped) == [
        "thinker.model.layers.0.mlp.experts.0.down_proj.weight",
        "thinker.model.layers.0.mlp.experts.1.down_proj.weight",
    ]
    torch.testing.assert_close(mapped["thinker.model.layers.0.mlp.experts.1.down_proj.weight"], tensor[1])


def test_map_qwen3_omni_moe_param_accepts_single_row_slot_probe():
    # verl enumerates a handler's output slots by calling it per expert with a
    # one-row dummy stack, so the handler must key names off expert_id_base
    # instead of deriving them from the number of rows it received.
    dummy_row = torch.zeros(1, 3, 2)
    name = "thinker.model.layers.0.mlp.experts.down_proj"

    mapped = dict(map_qwen3_omni_moe_param(name, dummy_row, expert_id_base=3))

    assert list(mapped) == ["thinker.model.layers.0.mlp.experts.3.down_proj.weight"]


def test_registers_only_the_omni_engine():
    _registry.register.assert_called_once_with(model_type="omni_model", backend="veomni", device="cuda")
    assert _handlers["qwen3_omni_moe"] is map_qwen3_omni_moe_param


def _engine():
    engine = Qwen3OmniThinkerVeOmniEngine()
    engine.module = SimpleNamespace(config=_config())
    engine.model_config = SimpleNamespace(
        hf_config=_config(),
        model_stage="thinker",
        use_remove_padding=True,
        lora_rank=0,
        lora={},
        local_hf_config_path="checkpoint",
    )
    engine.engine_config = SimpleNamespace(ulysses_parallel_size=1)
    return engine


def _batch():
    from tensordict import TensorDict

    return TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.tensor([1, 151655, 2, 151655])], layout=torch.jagged),
            "response_mask": torch.nested.as_nested_tensor([torch.ones(2, dtype=torch.bool)], layout=torch.jagged),
        },
        batch_size=1,
    )


def test_engine_masks_the_prompt_image_and_preserves_response_ids():
    engine = _engine()
    batch = _batch()
    inputs = {"input_ids": batch["input_ids"].values().unsqueeze(0), "pixel_values": torch.ones(2, 3)}
    expected = inputs["input_ids"].clone()
    engine._apply_veomni_input_transforms(inputs, batch)
    torch.testing.assert_close(inputs["input_ids"], expected)
    torch.testing.assert_close(inputs["image_mask"], torch.tensor([[False, True, False, False]]))
    assert inputs["base_transform_called"]


@pytest.mark.parametrize("key", ["pixel_values_videos", "input_features"])
def test_engine_rejects_unsupported_features(key):
    engine = _engine()
    with pytest.raises(NotImplementedError, match="inputs yet"):
        engine._apply_veomni_input_transforms({"input_ids": torch.tensor([[1]]), key: torch.ones(1)}, _batch())


def test_engine_requires_prompt_boundaries():
    with pytest.raises(ValueError, match="response_mask"):
        _engine()._apply_veomni_input_transforms({"input_ids": torch.tensor([[1]])}, {})


@pytest.mark.parametrize(
    ("owner", "key", "value"),
    [
        ("model_config", "model_stage", "talker"),
        ("model_config", "lora_rank", 8),
        ("model_config", "lora", {"rank": 8}),
        ("model_config", "use_remove_padding", False),
        ("engine_config", "ulysses_parallel_size", 2),
    ],
)
def test_engine_rejects_unsupported_config_before_model_loading(owner, key, value):
    engine = _engine()
    setattr(getattr(engine, owner), key, value)
    with pytest.raises(NotImplementedError):
        engine._get_model_config_path()


def test_engine_patches_before_loading():
    with patch.object(adapter, "patch_veomni_causal_mask_kwargs") as compat:
        assert _engine()._get_model_config_path() == "checkpoint"
    compat.assert_called_once_with()


def test_optimizer_excludes_modality_towers():
    model = torch.nn.Module()
    model.config = _config()
    model.thinker = torch.nn.Module()
    model.thinker.visual = torch.nn.Linear(2, 2)
    model.thinker.audio_tower = torch.nn.Linear(2, 2)
    model.thinker.model = torch.nn.Linear(2, 2)
    params = _engine()._build_optimizer(model)
    assert {id(p) for p in params} == {id(p) for p in model.thinker.model.parameters()}


def test_causal_mask_shim_drops_only_cache_position():
    def create_causal_mask(config, inputs_embeds):
        return config, inputs_embeds

    modeling = SimpleNamespace(create_causal_mask=create_causal_mask)
    with patch.dict(
        sys.modules,
        {
            "transformers.masking_utils": SimpleNamespace(create_causal_mask=create_causal_mask),
            "veomni.models.transformers.qwen3_omni_moe.generated": SimpleNamespace(
                patched_modeling_qwen3_omni_moe_gpu=modeling
            ),
        },
    ):
        assert adapter.patch_veomni_causal_mask_kwargs()
        assert modeling.create_causal_mask(config=1, inputs_embeds=2, cache_position=None) == (1, 2)
        assert not adapter.patch_veomni_causal_mask_kwargs()
        with pytest.raises(TypeError):
            modeling.create_causal_mask(config=1, inputs_embeds=2, unknown=True)
