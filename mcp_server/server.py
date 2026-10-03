"""MCP server exposing the trained IoT anomaly detector to an LLM.

Run (stdio):  python server.py
Never print to stdout in this file: stdout carries the MCP protocol.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # lets Claude Desktop run this file from any directory

from mcp.server.fastmcp import FastMCP

from detector import Detector

det = Detector()  # fail fast at startup if artifacts are missing or inconsistent

mcp = FastMCP(
    "iot-anomaly-detector",
    instructions=(
        "Query a trained anomaly detection model for IoT network traffic (N-BaIoT features). "
        "Always report benign_percentile and band, not only the flagged boolean, and mention "
        "known_blind_spots_on_this_device when a window is not flagged. Explanations are "
        "reconstruction-error attributions, not proof of cause."
    ),
)


@mcp.tool()
def list_devices() -> dict:
    """List the devices this model covers, their alert thresholds, available demo windows and
    known blind spots. Call this first to see valid device names."""
    return det.list_devices()


@mcp.tool()
def get_model_card() -> dict:
    """Return the model's measured performance per device and its limitations.
    Use this to answer questions about how reliable the detector is."""
    return det.model_card()


@mcp.tool()
def score_window(device: str, features: list[float] | None = None, demo_window: str | None = None) -> dict:
    """Score one traffic window for a device.

    Provide exactly one of:
      - features: the 115 raw N-BaIoT feature values for the window, in dataset column order
      - demo_window: the name of a built-in sample window (see list_devices), e.g. 'mirai_syn'

    Returns the anomaly score, whether it exceeds the alert threshold, and where the score falls
    among that device's normal training windows (benign_percentile). A window can be 'elevated'
    (above 95% of normal, below the alert threshold). A 'normal' result does not rule out attacks
    the model is known to miss on that device.
    """
    return det.score(device, features, demo_window)


@mcp.tool()
def explain_window(device: str, features: list[float] | None = None, demo_window: str | None = None,
                   top_k: int = 8) -> dict:
    """Score a window and explain which features drove the score.

    Same inputs as score_window, plus top_k (1 to 115). For the top features it returns the observed
    value, the value the model expected, the deviation from this device's normal in standard deviations,
    and each feature's share of the total reconstruction error, plus error shares grouped by traffic
    stream and time window. This is attribution of reconstruction error, not a causal diagnosis.
    """
    return det.explain(device, features, demo_window, top_k)


@mcp.tool()
def list_features() -> list[dict]:
    """List the 115 input features in the order score_window expects, with plain-language meanings."""
    return det.feature_table()


@mcp.resource("anomaly://model-card")
def model_card_resource() -> str:
    """Model card: measured performance and limitations."""
    return json.dumps(det.model_card(), indent=2)


@mcp.prompt()
def triage_window(device: str, demo_window: str) -> str:
    """Ask for a short triage report on one built-in sample window."""
    return (
        f"Use explain_window on device '{device}' with demo_window '{demo_window}'. Then write a short "
        "triage report: the verdict and where the score falls among normal traffic, the top three features "
        "driving it in plain language, and any known blind spot on this device that affects how much to "
        "trust the verdict. State clearly that the explanation is reconstruction-error attribution."
    )


if __name__ == "__main__":
    mcp.run()  # stdio transport
