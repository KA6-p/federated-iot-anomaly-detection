"""Tests for the detector core.

Structural tests check invariants that must hold for any valid artifacts. Tests marked `behavior` check
results that depend on the real trained model (run all with `pytest`; skip those with -m "not behavior").
"""
import numpy as np
import pytest

from detector import Detector
from features import decode_feature

det = Detector()
DEVICES = list(det.devices)


def some_window(dev):
    return det.demo[dev]["benign_typical"]


# ---------------- structural ----------------
def test_all_feature_names_decode():
    for name in det.feature_names:
        assert decode_feature(name)["stream"] is not None, name
    assert decode_feature("HH_jit_L0.01_variance")["window"] == "~1 min"
    assert decode_feature("not_a_feature")["text"] == "not_a_feature"


@pytest.mark.parametrize("alias,expected", [
    ("Doorbell", "Doorbell"), ("danmini doorbell", "Doorbell"), ("Ecobee Thermostat", "Thermostat"),
    ("baby monitor", "BabyMonitor"), ("Philips_B120N10_Baby_Monitor", "BabyMonitor"), ("camera", "Camera"),
])
def test_device_aliases(alias, expected):
    assert det.resolve_device(alias).short == expected


@pytest.mark.parametrize("kwargs", [
    dict(device="toaster", demo_window="benign_typical"),
    dict(device="Doorbell"),                                           # neither input
    dict(device="Doorbell", features=[0.0] * 115, demo_window="benign_typical"),  # both inputs
    dict(device="Doorbell", features=[0.0] * 10),                      # wrong length
    dict(device="Doorbell", features=[float("nan")] + [0.0] * 114),    # NaN
    dict(device="Doorbell", demo_window="no_such_window"),
])
def test_bad_inputs_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        det.score(**kwargs)


def test_top_k_bounds():
    for k in (0, det.D + 1):
        with pytest.raises(ValueError):
            det.explain("Doorbell", demo_window="benign_typical", top_k=k)


@pytest.mark.parametrize("dev", DEVICES)
def test_explain_invariants(dev):
    r = det.explain(dev, demo_window="benign_typical", top_k=8)
    shares = [t["share_of_total_error_pct"] for t in r["top_features"]]
    assert len(shares) == 8 and shares == sorted(shares, reverse=True)
    assert r["top_features_cumulative_share_pct"] <= 100.0 + 1e-6
    assert abs(sum(r["error_share_by_stream_pct"].values()) - 100.0) < 0.5
    assert abs(sum(r["error_share_by_time_window_pct"].values()) - 100.0) < 0.5
    assert all(np.isfinite(t["model_expected_value"]) for t in r["top_features"])
    # the reported score equals the mean per-feature error implied by the shares
    full = det.explain(dev, demo_window="benign_typical", top_k=det.D)
    assert abs(sum(t["share_of_total_error_pct"] for t in full["top_features"]) - 100.0) < 0.5


@pytest.mark.parametrize("dev", DEVICES)
def test_device_mean_is_zero_deviation(dev):
    d = det.devices[dev]
    r = det.explain(dev, features=d.mean.tolist(), top_k=5)
    assert all(abs(t["deviation_from_device_normal_in_std"]) < 1e-6 for t in r["top_features"])


@pytest.mark.parametrize("dev", DEVICES)
def test_percentile_is_monotone_and_bounded(dev):
    d = det.devices[dev]
    scores = np.linspace(0, d.grid_err[-1] * 3, 200)
    pcts = [det.benign_percentile(d, s) for s in scores]
    assert all(0.0 <= p <= 100.0 for p in pcts)
    assert all(b >= a - 1e-9 for a, b in zip(pcts, pcts[1:]))


@pytest.mark.parametrize("dev", DEVICES)
def test_band_matches_thresholds(dev):
    d = det.devices[dev]
    assert d.thr95 <= d.thr99
    r = det.score(dev, demo_window="benign_typical")
    expected = "flagged" if r["anomaly_score"] > d.thr99 else "elevated" if r["anomaly_score"] > d.thr95 else "normal"
    assert r["band"] == expected and r["flagged"] == (expected == "flagged")


def test_forward_matches_independent_computation():
    x = np.asarray(some_window("Doorbell"), dtype=np.float64)
    d = det.devices["Doorbell"]
    xs = ((x - d.mean) / d.std).astype(np.float32)
    w = det.w
    h = np.maximum(np.einsum("ij,j->i", w["W1"], xs) + w["b1"], 0)
    z = np.maximum(np.einsum("ij,j->i", w["W2"], h) + w["b2"], 0)
    h2 = np.maximum(np.einsum("ij,j->i", w["W3"], z) + w["b3"], 0)
    out = np.einsum("ij,j->i", w["W4"], h2) + w["b4"]
    assert np.allclose(det._forward(xs), out, atol=1e-5)


def test_list_and_card_have_every_device():
    assert {d["device"] for d in det.list_devices()["devices"]} == set(DEVICES)
    assert set(det.model_card()["per_device"]) == set(DEVICES)
    assert det.model_card()["limitations"]


# ---------------- behavior (needs the real trained model) ----------------
@pytest.mark.behavior
@pytest.mark.parametrize("dev", DEVICES)
def test_easy_attack_is_flagged(dev):
    r = det.score(dev, demo_window="mirai_syn")
    assert r["flagged"] and r["benign_percentile"] > 99.9


@pytest.mark.behavior
@pytest.mark.parametrize("dev", DEVICES)
def test_typical_benign_is_normal(dev):
    assert det.score(dev, demo_window="benign_typical")["band"] == "normal"


@pytest.mark.behavior
@pytest.mark.parametrize("dev", DEVICES)
def test_model_card_matches_export_claims(dev):
    c = det.card[dev]
    assert c["benign_test_recall_at_p99"] > 0.98
    assert c["attack_recall_at_p99_8_well_posed"] > 0.99
