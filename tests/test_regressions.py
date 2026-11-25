import torch
from core.corr import CorrBlockGather1D


def test_half_correlation_pooling_does_not_overflow():
    # Each correlation is 40000: summing two before division overflows fp16.
    features = torch.full((1, 1, 1, 16), 200.0, dtype=torch.float16)
    block = CorrBlockGather1D(features, features)
    for level in block.corr_pyramid:
        torch.testing.assert_close(level, torch.full_like(level, 40000.0))


def test_missing_optional_sampler_has_actionable_error(monkeypatch):
    import core.corr as corr
    import pytest

    monkeypatch.setattr(corr, "corr_sampler", None)
    with pytest.raises(RuntimeError, match="optional sampler extension"):
        corr.CorrSampler.apply(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), 1)
