import numpy as np
import pytest

from paperpen.config import FilterConfig
from paperpen.track.filter import OneEuro, Smoother


def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    f = OneEuro(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    t = 0.0
    raw = []
    out = []
    for i in range(300):
        t += 1 / 30
        x = 100.0 + rng.normal(0, 2.0)  # static point with 2px noise
        raw.append(x)
        out.append(f.filter(x, t))
    # skip warmup: output jitter must be well below input jitter
    assert np.std(out[100:]) < 0.5 * np.std(raw[100:])
    assert abs(np.mean(out[100:]) - 100.0) < 1.0


def test_one_euro_follows_motion():
    f = OneEuro(min_cutoff=1.0, beta=1.0, d_cutoff=1.0)
    t = 0.0
    f.filter(0.0, t)
    lagged = []
    for i in range(30):
        t += 1 / 30
        lagged.append(abs(f.filter(100.0, t) - 100.0))
    # should get close quickly while moving fast (beta raises cutoff)
    assert lagged[-1] < 5.0


def test_smoother_observes_and_valid():
    s = Smoother(FilterConfig())
    out = s.update((10.0, 20.0), 0.0)
    assert out.valid and out.observed and not out.coasting


def test_smoother_coasts_through_occlusion():
    cfg = FilterConfig(max_coast_frames=5)
    s = Smoother(cfg)
    t = 0.0
    # moving right at 100 px/s
    for i in range(10):
        t += 1 / 30
        s.update((10.0 + i * 100 / 30, 50.0), t)
    last_x = s.position[0]
    out = s.update(None, t + 1 / 30)
    assert out.valid and out.coasting
    assert out.x > last_x  # continues in direction of motion
    # beyond the coast limit -> invalid, position held
    for i in range(cfg.max_coast_frames):
        out = s.update(None, t + (i + 2) / 30)
    assert not out.valid
    assert s.position[0] >= last_x


def test_smoother_without_observation_is_invalid():
    s = Smoother()
    out = s.update(None, 0.0)
    assert not out.valid


def test_smoother_recovers_after_loss():
    s = Smoother(FilterConfig(max_coast_frames=3))
    t = 0.0
    for i in range(5):
        t += 1 / 30
        s.update((float(i), 1.0), t)
    for i in range(6):  # exceed coast
        t += 1 / 30
        s.update(None, t)
    out = s.update((42.0, 1.0), t + 1 / 30)
    assert out.valid and out.observed
    # one-euro lags, but must move decisively toward the new observation
    assert out.x > 30.0
    for _ in range(30):  # and settle there
        t += 1 / 30
        out = s.update((42.0, 1.0), t)
    assert out.x == pytest.approx(42.0, abs=1.0)
