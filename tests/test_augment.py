import numpy as np
import pytest

from ecg.augment import NOISE_TYPES, perturb, query_rng
from ecg.preprocess import SAMPLE_RATE


@pytest.mark.parametrize("kind", NOISE_TYPES)
def test_perturbation_is_reproducible_rms_scaled_and_non_mutating(kind):
    x = np.sin(2 * np.pi * 7 * np.arange(10 * SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
    original = x.copy()
    mild = perturb(x, kind, 0.25, query_rng("record-a", 42))
    strong = perturb(x, kind, 0.5, query_rng("record-a", 42))
    another = perturb(x, kind, 0.25, query_rng("record-b", 42))
    assert mild.dtype == x.dtype
    np.testing.assert_array_equal(x, original)
    np.testing.assert_allclose(strong - x, 2 * (mild - x), atol=3e-7)
    assert np.linalg.norm(mild - x) / np.linalg.norm(x) == pytest.approx(0.25)
    assert not np.allclose(mild, another)
