"""Plain-language decoding of N-BaIoT feature names.

N-BaIoT features are named like ``HH_jit_L0.1_variance``:
    <stream>_<decay window>_<statistic>

The descriptions below are approximate, written from the dataset's documentation,
and exist so explanations are readable. They are not part of the model.
"""
import re

STREAMS = {
    "MI_dir": "packet sizes sent from one source (MAC + IP)",
    "H": "packet sizes sent from one source IP",
    "HH": "packet sizes on one source-IP to destination-IP channel",
    "HH_jit": "packet timing jitter on one source-IP to destination-IP channel",
    "HpHp": "packet sizes on one source IP:port to destination IP:port socket",
}

# decay parameter L -> approximate length of the time window
WINDOWS = {
    "5": "~100 ms",
    "3": "~500 ms",
    "1": "~1.5 s",
    "0.1": "~10 s",
    "0.01": "~1 min",
}

STATS = {
    "weight": "packet count in the window",
    "mean": "mean value",
    "variance": "variance",
    "std": "standard deviation",
    "magnitude": "combined magnitude of the two directions",
    "radius": "combined spread of the two directions",
    "covariance": "covariance between the two directions",
    "pcc": "correlation between the two directions",
}

_PATTERN = re.compile(r"^(MI_dir|HH_jit|HpHp|HH|H)_L([0-9.]+)_([A-Za-z]+)$")


def decode_feature(name: str) -> dict:
    """Return {stream, window, stat, text} for a feature name.

    Unknown names come back with text equal to the name, never an error.
    """
    m = _PATTERN.match(name)
    if not m:
        return {"stream": None, "window": None, "stat": None, "text": name}
    stream, decay, stat = m.groups()
    window = WINDOWS.get(decay)
    parts = [
        f"{STATS.get(stat, stat)} of {STREAMS[stream]}",
        f"over a {window} window" if window else f"(decay L{decay})",
    ]
    return {"stream": stream, "window": window or f"L{decay}", "stat": stat, "text": " ".join(parts)}
