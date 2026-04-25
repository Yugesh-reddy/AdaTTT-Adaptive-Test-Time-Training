"""
Phase 2 shift generation: severity behaviour and cache reproducibility.

The 5-view feature cache (original + 4 AugMix views) is only valid if the views
are reproducible across processes and sessions; otherwise a rerun silently
evaluates different inputs than the cache it compares against.
"""

import subprocess
import sys
import zlib

import numpy as np
import pytest
import torch
from PIL import Image

from ttt.shifts import (
    GAUSSIAN_NOISE_STD,
    SEVERITIES,
    augmix_views,
    corrupt,
    seed_for,
)


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))


def _arr(img):
    return np.asarray(img, dtype=np.float32)


def _distance(a, b):
    return float(np.abs(_arr(a) - _arr(b)).mean())


@pytest.mark.parametrize("kind", ["gaussian_noise", "gaussian_blur"])
def test_severity_is_monotone(image, kind):
    distances = [_distance(image, corrupt(image, kind, s, "42")) for s in SEVERITIES]
    assert distances == sorted(distances), distances
    assert distances[0] > 0


@pytest.mark.parametrize("kind", ["gaussian_noise", "gaussian_blur"])
def test_corruption_is_reproducible(image, kind):
    first = corrupt(image, kind, 3, "42")
    assert np.array_equal(_arr(first), _arr(corrupt(image, kind, 3, "42")))


def test_noise_differs_per_sample(image):
    a = corrupt(image, "gaussian_noise", 3, "42")
    b = corrupt(image, "gaussian_noise", 3, "43")
    assert not np.array_equal(_arr(a), _arr(b))


def test_unknown_corruption_and_severity_raise(image):
    with pytest.raises(ValueError, match="Unknown corruption"):
        corrupt(image, "snow", 3)
    with pytest.raises(ValueError, match="Severity"):
        corrupt(image, "gaussian_noise", 7)


def test_identity_is_a_pixel_noop(image):
    out = corrupt(image, "identity", 1, "42")
    assert np.array_equal(_arr(out.convert("RGB")), _arr(image.convert("RGB")))
    assert np.array_equal(
        _arr(corrupt(image, "identity", 1, "42")),
        _arr(corrupt(image, "identity", 5, "99")),
    )


def test_views_are_distinct_and_reproducible(image):
    views = augmix_views(image, n_views=4, sample_id="42")
    assert len(views) == 4
    again = augmix_views(image, n_views=4, sample_id="42")
    for view, repeat in zip(views, again):
        assert np.array_equal(_arr(view), _arr(repeat))
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    assert all(not np.array_equal(_arr(views[i]), _arr(views[j])) for i, j in pairs)
    assert all(_distance(image, v) > 0 for v in views)


def test_building_views_leaves_the_global_rng_alone(image):
    torch.manual_seed(1234)
    expected = torch.rand(3)
    torch.manual_seed(1234)
    augmix_views(image, n_views=2, sample_id="42")
    assert torch.equal(torch.rand(3), expected)


def test_seed_derivation_is_checksum_not_salted_hash():
    # Pins the field order too: a change here invalidates every existing cache.
    assert seed_for("42", 0) == zlib.crc32(b"42:0:0") & 0xFFFFFFFF


def test_views_survive_a_different_process_hash_seed(tmp_path):
    """hash() is salted per process; a cache built today must match one built tomorrow."""
    code = (
        "import numpy as np, hashlib;"
        "from PIL import Image;"
        "from ttt.shifts import augmix_views;"
        "rng=np.random.default_rng(0);"
        "img=Image.fromarray(rng.integers(0,255,(64,64,3),dtype=np.uint8));"
        "v=augmix_views(img, n_views=2, sample_id='42');"
        "print(hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in v)).hexdigest())"
    )
    digests = []
    for hash_seed in ("0", "1"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
                             cwd=str(tmp_path.parent.parent))
        assert out.returncode == 0, out.stderr[-400:]
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1]
