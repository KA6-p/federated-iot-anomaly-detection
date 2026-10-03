# IoT Anomaly Detector: MCP Server

An [MCP](https://modelcontextprotocol.io) server that lets an LLM query the anomaly detection model trained in this repository, score IoT traffic windows, and explain what drove a score. It serves the **FedAvg + local scalers** autoencoder from `04_extension_noniid_unseen.ipynb`, with the thresholds and per-feature baselines computed there.

The point is not a thin wrapper. The server exposes what the experiments found: how reliable the detector is per device, where it is blind, and how close a "not flagged" window came to the alert threshold.

## Tools

| Tool | What it does |
|---|---|
| `list_devices` | Devices covered, alert thresholds, demo windows, known blind spots |
| `get_model_card` | Measured performance per device and the model's limitations |
| `score_window` | Anomaly score, alert verdict, and where the score falls among that device's normal windows |
| `explain_window` | The same, plus the top features driving the score (observed vs expected value, deviation from the device's normal, share of error) and error grouped by traffic stream and time window |
| `list_features` | The 115 input features in order, with plain-language meanings |

Also: a resource `anomaly://model-card` and a prompt `triage_window`.

Input is either 115 raw N-BaIoT feature values, or `demo_window` (a built-in sample such as `mirai_syn`) so you can try it without pasting numbers.

## Example prompts

- "List the devices this detector covers and its known blind spots."
- "Explain the `mirai_syn` demo window on the Thermostat. Which features drove the score?"
- "Score the `gafgyt_tcp` demo window on the Camera. Is it flagged? How close is it?"
- "How reliable is this detector on the Baby Monitor?"

## Design decisions

- **Verdicts come with context.** Besides `flagged`, every score reports `benign_percentile` (share of that device's normal training windows scoring lower) and a band: `normal`, `elevated` (above the 95th percentile of normal) or `flagged` (above the 99th). The experiments showed that some attacks (`gafgyt_tcp`/`udp`) score above most normal traffic yet stay under the alert threshold on some devices, and a boolean hides that.
- **Thresholds need no attack labels.** They are percentiles of each device's own benign training error.
- **Explanations are honest about what they are.** Attribution is by per-feature reconstruction error. Each output carries a reading guide saying it is not a causal diagnosis, and features are decoded to plain language (stream, time window, statistic).
- **Limitations travel with the tool.** `known_blind_spots_on_this_device` is derived from the model card and returned with every score.
- **NumPy-only inference.** The 115-64-16-64-115 autoencoder is a few matrix multiplications, so the server needs no PyTorch. The export cell in notebook 04 converts the weights once and checks that NumPy matches PyTorch on every demo window.
- **Fail fast.** Missing or inconsistent artifacts stop the server at startup with a clear message. Bad inputs return a tool error the LLM can read and correct.

## Setup

```bash
cd mcp_server
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest                      # all tests
python demo.py --explain    # replay the demo windows without any MCP client
```

The weights ship as `artifacts/model.npz`. The last cell of notebook 04 produces it and checks that the NumPy forward pass matches PyTorch on every demo window, so serving needs no PyTorch.

Developed against `mcp` 1.30.0 (the 1.x `FastMCP` API; the 2.x SDK renamed it, hence the `<2` pin), NumPy 2.4 and Python 3.12.

## Connect it to Claude

**Claude Desktop.** Edit `claude_desktop_config.json` (Settings, Developer, Edit Config) and use the Python from the virtual environment, with absolute paths:

```json
{
  "mcpServers": {
    "iot-anomaly-detector": {
      "command": "/absolute/path/to/mcp_server/.venv/bin/python",
      "args": ["/absolute/path/to/mcp_server/server.py"]
    }
  }
}
```

On Windows the interpreter is `...\\.venv\\Scripts\\python.exe`. Restart Claude Desktop afterwards.

**Claude Code.**

```bash
claude mcp add iot-anomaly-detector -- /absolute/path/to/mcp_server/.venv/bin/python /absolute/path/to/mcp_server/server.py
```

If your app version's menus or flags differ, check the current Claude Desktop or Claude Code documentation.

## Limitations

These are also returned by `get_model_card`.

- Trained and evaluated on one lab dataset (N-BaIoT), 4 of 9 devices. Not validated in the field.
- Metrics are for this single exported model (seed 0), not the 3-seed means in the main README. Benign recall is measured on the last 30% of each benign capture, from the same session as training.
- `gafgyt_tcp` and `gafgyt_udp` are near-copy attack files (about 20 distinct rows each). They rank above most normal windows but fall under the alert threshold on 3 of 4 devices.
- A "normal" verdict does not prove a window is safe.
- No privacy analysis of the federated training (no differential privacy or secure aggregation).

## Demo output

<!-- Paste the output of `python demo.py --explain` here after running it on your trained model. -->
