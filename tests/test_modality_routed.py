from __future__ import annotations

import pytest
from torch import nn

from anchorstep.adapters.modality_routed import (
    RoutedAdapterConfig,
    RoutedLowRankAdapter,
    install_modality_routed_adapters,
    parameter_modality,
)


class TinyLingBotBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_attention = nn.Linear(8, 8)
        self.video_ffn = nn.Linear(8, 8)
        self.action_ffn = nn.Linear(8, 8)
        self.temporal_projection = nn.Linear(8, 8)


class TinyLingBot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyLingBotBlock(), TinyLingBotBlock()])


def test_install_routes_paper_ranks_and_freezes_shared_attention() -> None:
    model = TinyLingBot()
    topology = install_modality_routed_adapters(model, RoutedAdapterConfig())

    for block in model.blocks:
        assert isinstance(block.video_ffn, RoutedLowRankAdapter)
        assert isinstance(block.action_ffn, RoutedLowRankAdapter)
        assert isinstance(block.temporal_projection, RoutedLowRankAdapter)
        assert block.video_ffn.rank == 64
        assert block.action_ffn.rank == 64
        assert block.temporal_projection.rank == 128
        assert all(
            not parameter.requires_grad
            for parameter in block.shared_attention.parameters()
        )
    assert topology.video_parameter_names.isdisjoint(topology.action_parameter_names)
    assert topology.video_parameter_names
    assert topology.action_parameter_names
    assert {parameter_modality(name) for name in topology.video_parameter_names} == {
        "video"
    }
    assert {parameter_modality(name) for name in topology.action_parameter_names} == {
        "action"
    }


def test_parameter_modality_rejects_unclassified_trainable_name() -> None:
    with pytest.raises(ValueError, match="Unclassified trainable parameter"):
        parameter_modality("blocks.0.shared_attention.weight")
