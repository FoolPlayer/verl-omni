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
"""Derive input-only modality masks from packed prompt/response boundaries."""

import torch
from tensordict import TensorDict


def build_prompt_region_mask(
    micro_batch: TensorDict,
    packed_length: int,
) -> torch.Tensor | None:
    """Mark the prompt half of every packed sequence, ``(1, packed_length)``.

    Args:
        micro_batch: the packed batch. ``input_ids`` is jagged over
            ``prompt + response`` and ``response_mask`` is jagged over
            ``response`` alone, so their offsets give both lengths.
        packed_length: width of ``model_inputs["input_ids"]``, i.e. ``total_nnz``
            plus any right pad added after packing.

    Returns:
        The boolean mask, or ``None`` when the batch is not jagged or carries no
        ``response_mask`` (e.g. a pure-inference call), in which case the caller
        must fall back to matching the whole sequence.
    """

    input_ids = micro_batch.get("input_ids", None)
    response_mask = micro_batch.get("response_mask", None)
    if input_ids is None or response_mask is None:
        return None
    if not input_ids.is_nested or not response_mask.is_nested:
        return None

    seq_offsets = input_ids.offsets().to(torch.long)
    response_offsets = response_mask.offsets().to(device=seq_offsets.device, dtype=torch.long)
    batch_size = seq_offsets.numel() - 1
    if batch_size < 1 or response_offsets.numel() - 1 != batch_size:
        return None

    # Absolute end of each prompt = sequence end - response length.
    prompt_ends = seq_offsets[1:] - response_offsets.diff()

    positions = torch.arange(packed_length, device=seq_offsets.device)
    # Which sequence each packed position belongs to. Positions in the trailing
    # pad land at ``batch_size`` and are dropped by ``in_batch``. Everything
    # stays in tensor-land so this costs no device sync.
    sequence_index = torch.searchsorted(seq_offsets[1:].contiguous(), positions, right=True)
    in_batch = sequence_index < batch_size
    return (in_batch & (positions < prompt_ends[sequence_index.clamp(max=batch_size - 1)])).unsqueeze(0)
