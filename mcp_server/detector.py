"""Anomaly detector core: loads the exported autoencoder and scores traffic windows.

Pure NumPy inference (no PyTorch needed at serve time). The model is the
FedAvg + local-scalers autoencoder exported from notebook 04:
    115 -> 64 -> 16 -> 64 -> 115, ReLU, MSE reconstruction error as the anomaly score.

Per device, a window is:
  1. standardized with that device's own benign-training mean/std,
  2. passed through the shared (federated) autoencoder,
  3. scored by mean squared reconstruction error,
  4. compared with that device's benign-TRAIN error percentiles (no attack labels used).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from features import decode_feature

ARTIFACT_DIR = Path(os.environ.get("ANOMALY_ARTIFACTS", Path(__file__).parent / "artifacts"))

# torch state-dict key -> short name used in model.npz
KEYMAP = {
    "encoder.0.weight": "W1", "encoder.0.bias": "b1",
    "encoder.2.weight": "W2", "encoder.2.bias": "b2",
    "decoder.0.weight": "W3", "decoder.0.bias": "b3",
    "decoder.2.weight": "W4", "decoder.2.bias": "b4",
}

REASONING_NOTE = (
    "Reading guide: this is reconstruction-error attribution. It shows which features the "
    "autoencoder could not reproduce for this window, not a causal diagnosis. Features are "
    "standardized with this device's own benign statistics, so a feature that barely varies in "
    "normal traffic can show a large deviation from a small absolute change."
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _load_weights(d: Path) -> dict:
    npz = d / "model.npz"
    if npz.exists():
        z = np.load(npz)
        return {k: z[k].astype(np.float32) for k in KEYMAP.values()}
    pt = d / "model.pt"
    if pt.exists():
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "model.npz not found in artifacts/ and PyTorch is not installed. Generate model.npz with "
                "the export-check cell at the end of notebook 04 and copy it into mcp_server/artifacts/."
            ) from e
        sd = torch.load(pt, map_location="cpu")
        return {KEYMAP[k]: v.numpy().astype(np.float32) for k, v in sd.items()}
    raise FileNotFoundError(f"No model.npz or model.pt in {d}")


class _Device:
    def __init__(self, short: str, cfg: dict, card: dict):
        self.short = short
        self.full_name = cfg["full_name"]
        self.mean = np.asarray(cfg["scaler_mean"], dtype=np.float64)
        self.std = np.asarray(cfg["scaler_std"], dtype=np.float64)
        self.thr95 = float(cfg["threshold_p95"])
        self.thr99 = float(cfg["threshold_p99"])
        self.median_err = float(cfg["benign_train_median_error"])
        self.feat_err = np.asarray(cfg["benign_feature_error"], dtype=np.float64)
        if "benign_error_grid" not in cfg:
            raise RuntimeError(
                f"config.json has no 'benign_error_grid' for {short}. Rerun the percentile-grid cell "
                "(Cell 3) in notebook 04 and re-download config.json."
            )
        grid = cfg["benign_error_grid"]
        order = np.argsort(grid["err"], kind="stable")
        self.grid_pct = np.asarray(grid["pct"], dtype=np.float64)[order]
        self.grid_err = np.asarray(grid["err"], dtype=np.float64)[order]
        self.card = card


class Detector:
    def __init__(self, artifact_dir: Path | str = ARTIFACT_DIR):
        d = Path(artifact_dir)
        with open(d / "config.json") as f:
            cfg = json.load(f)
        with open(d / "model_card.json") as f:
            self.card = json.load(f)
        with open(d / "demo_windows.json") as f:
            self.demo = json.load(f)

        self.method = cfg["method"]
        self.feature_names: list[str] = cfg["feature_names"]
        self.D = len(self.feature_names)
        arch = cfg["architecture"]
        if arch["input_dim"] != self.D:
            raise ValueError("config.json: input_dim does not match number of feature names")

        self.w = _load_weights(d)
        expected = {"W1": (arch["hidden"], self.D), "W2": (arch["latent"], arch["hidden"]),
                    "W3": (arch["hidden"], arch["latent"]), "W4": (self.D, arch["hidden"])}
        for k, shape in expected.items():
            if self.w[k].shape != shape:
                raise ValueError(f"weight {k} has shape {self.w[k].shape}, expected {shape}")

        self.devices = {s: _Device(s, c, self.card.get(s, {})) for s, c in cfg["devices"].items()}
        self._decoded = [decode_feature(n) for n in self.feature_names]

    # ---- helpers -------------------------------------------------------------------------
    def resolve_device(self, name: str) -> _Device:
        key = _norm(name)
        if not key:
            raise ValueError(f"Device name is empty. Valid devices: {list(self.devices)}")
        exact = [d for d in self.devices.values() if key in (_norm(d.short), _norm(d.full_name))]
        if len(exact) == 1:
            return exact[0]
        loose = [d for d in self.devices.values() if key in _norm(d.full_name) or _norm(d.short) in key]
        if len(loose) == 1:
            return loose[0]
        raise ValueError(f"Unknown or ambiguous device '{name}'. Valid devices: {list(self.devices)}")

    def get_features(self, features, demo_window, device: _Device) -> tuple[np.ndarray, str]:
        if (features is None) == (demo_window is None):
            raise ValueError("Provide exactly one of: features (list of 115 numbers) or demo_window (a name).")
        if demo_window is not None:
            windows = self.demo.get(device.short, {})
            if demo_window not in windows:
                raise ValueError(f"Unknown demo_window '{demo_window}'. Available: {sorted(windows)}")
            return np.asarray(windows[demo_window], dtype=np.float64), f"demo:{demo_window}"
        x = np.asarray(features, dtype=np.float64)
        if x.shape != (self.D,):
            raise ValueError(f"features must be a flat list of {self.D} numbers (got shape {x.shape}). "
                             "Use the raw N-BaIoT feature values, in dataset column order.")
        if not np.all(np.isfinite(x)):
            raise ValueError("features contains NaN or infinite values.")
        return x, "provided"

    def _forward(self, xs: np.ndarray) -> np.ndarray:
        w = self.w
        h = np.maximum(xs @ w["W1"].T + w["b1"], 0.0)
        z = np.maximum(h @ w["W2"].T + w["b2"], 0.0)
        h2 = np.maximum(z @ w["W3"].T + w["b3"], 0.0)
        return h2 @ w["W4"].T + w["b4"]

    def _error_vector(self, x: np.ndarray, dev: _Device):
        xs = ((x - dev.mean) / dev.std).astype(np.float32)
        recon = self._forward(xs)
        return xs, recon, (xs - recon).astype(np.float64) ** 2

    @staticmethod
    def benign_percentile(dev: _Device, score: float) -> float:
        """Percent of this device's benign TRAINING windows with a lower score (0 to 100)."""
        return float(np.interp(score, dev.grid_err, dev.grid_pct, left=0.0, right=100.0))

    @staticmethod
    def _band(dev: _Device, score: float) -> str:
        if score > dev.thr99:
            return "flagged"
        if score > dev.thr95:
            return "elevated"
        return "normal"

    def blind_spots(self, dev: _Device) -> list[str]:
        pa = dev.card.get("per_attack_recall_at_p99", {})
        out = []
        for a in ("gafgyt_tcp", "gafgyt_udp"):
            if a in pa and pa[a] < 0.5:
                out.append(f"{a}: only {pa[a]:.1%} of these attack windows exceed the alert threshold on this device")
        return out

    # ---- public API ----------------------------------------------------------------------
    def score(self, device: str, features=None, demo_window=None) -> dict:
        dev = self.resolve_device(device)
        x, source = self.get_features(features, demo_window, dev)
        _, _, err = self._error_vector(x, dev)
        s = float(err.mean())
        band = self._band(dev, s)
        pct = self.benign_percentile(dev, s)
        out = {
            "device": dev.short, "window_source": source,
            "anomaly_score": s,
            "alert_threshold_p99": dev.thr99, "elevated_threshold_p95": dev.thr95,
            "score_over_alert_threshold": s / dev.thr99,
            "benign_percentile": round(pct, 2),
            "band": band, "flagged": band == "flagged",
            "interpretation": self._interpret(dev, s, pct, band),
            "known_blind_spots_on_this_device": self.blind_spots(dev),
        }
        if s > dev.grid_err[-1]:
            out["note"] = "score is above every benign training window for this device"
        return out

    @staticmethod
    def _interpret(dev: _Device, s: float, pct: float, band: str) -> str:
        if band == "flagged":
            return (f"Flagged: score {s / dev.thr99:.1f}x the alert threshold, higher than {pct:.2f}% "
                    f"of {dev.short} normal training windows.")
        if band == "elevated":
            return (f"Not flagged, but elevated: higher than {pct:.2f}% of {dev.short} normal training "
                    "windows, below the alert threshold (the 99th percentile of normal). A near-miss is possible.")
        return f"Looks normal: higher than only {pct:.2f}% of {dev.short} normal training windows."

    def explain(self, device: str, features=None, demo_window=None, top_k: int = 8) -> dict:
        if not 1 <= int(top_k) <= self.D:
            raise ValueError(f"top_k must be between 1 and {self.D}")
        top_k = int(top_k)
        dev = self.resolve_device(device)
        x, source = self.get_features(features, demo_window, dev)
        xs, recon, err = self._error_vector(x, dev)
        base = self.score(device, features, demo_window)
        total = float(err.sum())
        share = err / total if total > 0 else np.zeros_like(err)
        ratio = err / np.maximum(dev.feat_err, 1e-6)
        expected_raw = recon.astype(np.float64) * dev.std + dev.mean

        top = []
        for rank, i in enumerate(np.argsort(-err)[:top_k], start=1):
            d = self._decoded[i]
            top.append({
                "rank": rank, "feature": self.feature_names[i], "meaning": d["text"],
                "observed_value": float(x[i]), "model_expected_value": float(expected_raw[i]),
                "deviation_from_device_normal_in_std": float(xs[i]),
                "share_of_total_error_pct": round(float(share[i]) * 100, 2),
                "error_vs_typical_benign": round(float(ratio[i]), 1),
            })

        by_stream: dict[str, float] = {}
        by_window: dict[str, float] = {}
        for i, d in enumerate(self._decoded):
            by_stream[d["stream"] or "other"] = by_stream.get(d["stream"] or "other", 0.0) + float(share[i])
            by_window[d["window"] or "other"] = by_window.get(d["window"] or "other", 0.0) + float(share[i])
        pct = lambda dct: {k: round(v * 100, 1) for k, v in sorted(dct.items(), key=lambda kv: -kv[1])}

        base.update({
            "top_features": top,
            "top_features_cumulative_share_pct": round(sum(t["share_of_total_error_pct"] for t in top), 2),
            "error_share_by_stream_pct": pct(by_stream),
            "error_share_by_time_window_pct": pct(by_window),
            "reading_guide": REASONING_NOTE,
        })
        return base

    def list_devices(self) -> dict:
        return {
            "model": self.method,
            "devices": [{
                "device": d.short, "full_name": d.full_name,
                "alert_threshold_p99": d.thr99, "elevated_threshold_p95": d.thr95,
                "benign_train_rows": d.card.get("benign_train_rows"),
                "demo_windows": sorted(self.demo.get(d.short, {})),
                "known_blind_spots": self.blind_spots(d),
            } for d in self.devices.values()],
            "feature_count": self.D,
            "input_format": "raw N-BaIoT feature values (115 numbers, dataset column order), or a demo_window name",
        }

    def model_card(self) -> dict:
        devices = {}
        for s, d in self.devices.items():
            c = d.card
            devices[s] = {k: c.get(k) for k in (
                "benign_train_rows", "benign_test_recall_at_p99", "attack_recall_at_p99_all10",
                "attack_recall_at_p99_8_well_posed", "auc_all10", "auc_8_well_posed", "auc_tcp_udp",
                "per_attack_recall_at_p99")}
        return {
            "model": self.method,
            "what_it_is": ("Autoencoder trained with federated averaging on benign traffic only, from 4 N-BaIoT "
                           "devices; each device standardizes with its own benign statistics. Score = reconstruction "
                           "error; thresholds are benign-train percentiles (no attack labels)."),
            "metrics_scope": self.card.get("_note"),
            "per_device": devices,
            "limitations": [
                "Trained and evaluated on one lab dataset (N-BaIoT), 4 of 9 devices; not validated in the field.",
                "Metrics are for this single exported model (seed 0). Benign recall is measured on the last 30% "
                "of each benign capture, from the same session as training.",
                "gafgyt_tcp and gafgyt_udp are near-copy attack files (about 20 distinct rows each). They are "
                "ranked above most normal windows but fall under the alert threshold on 3 of 4 devices.",
                "A 'normal' verdict does not prove a window is safe.",
                "Explanations are reconstruction-error attributions, not causal diagnoses.",
                "No privacy analysis of the federated training (no differential privacy or secure aggregation).",
            ],
            "source": "see the repository README and notebook 04 for experiments and caveats",
        }

    def feature_table(self) -> list[dict]:
        return [{"index": i, "name": n, "meaning": self._decoded[i]["text"]}
                for i, n in enumerate(self.feature_names)]
