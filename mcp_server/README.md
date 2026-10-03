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
- "Score the `gafgyt_tcp` demo window on the Camera. Is it flagged? How close is it?"
- "Score the `mirai_syn` demo window on the Thermostat. Is it flagged?"
- "How reliable is this detector on the Baby Monitor?"

## Design decisions

- **Verdicts come with context.** Besides `flagged`, every score reports `benign_percentile` (share of that device's normal training windows scoring lower) and a band: `normal`, `elevated` (above the 95th percentile of normal) or `flagged` (above the 99th). The experiments showed that some attacks (`gafgyt_tcp`/`udp`) score above most normal traffic yet stay under the alert threshold on some devices, and a boolean hides that.
- **Thresholds need no attack labels.** They are percentiles of each device's own benign training error.
- **Explanations are honest about what they are.** Attribution is by per-feature reconstruction error. Each output carries a reading guide saying it is not a causal diagnosis, and features are decoded to plain language (stream, time window, statistic).
- **Limitations travel with the tool.** `known_blind_spots_on_this_device` is derived from the model card and returned with every score.
- **NumPy-only inference.** The 115-64-16-64-115 autoencoder is a few matrix multiplications, so the server needs no PyTorch. The export cell in notebook 04 converts the weights once and checks that NumPy matches PyTorch on every demo window (worst relative difference 7.1e-07).
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

The weights ship as `artifacts/model.npz`. The last cells of notebook 04 produce the artifacts (`model.pt`, `model.npz`, `config.json`, `model_card.json`, `demo_windows.json`) and check that the NumPy forward pass matches PyTorch on every demo window, so serving needs no PyTorch.

Developed against `mcp` 1.30.0 (the 1.x `FastMCP` API; the 2.x SDK renamed it, hence the `<2` pin), NumPy 2.4 and Python 3.12. Tested with Claude Desktop on Windows.

## Connect it to Claude Desktop

Open Settings, Developer, Edit Config in Claude Desktop and add the server, using the Python from the virtual environment and absolute paths:

```json
{
  "mcpServers": {
    "iot-anomaly-detector": {
      "command": "C:/path/to/mcp_server/.venv/Scripts/python.exe",
      "args": ["C:/path/to/mcp_server/server.py"]
    }
  }
}
```

On macOS/Linux the interpreter is `/path/to/mcp_server/.venv/bin/python`. If the config file already has other settings, add `mcpServers` alongside them. Fully quit and reopen Claude Desktop, then check Settings, Developer for the running server. If your app version's menus differ, check the current Claude Desktop documentation.

## Demo output

`python demo.py` on the exported model (seed 0, thresholds from benign training error):

| Device | Window | Score | x alert threshold | Benign percentile | Verdict |
|---|---|---|---|---|---|
| Doorbell | benign_typical | 0.09601 | 0.05 | 50.15 | normal |
| Doorbell | gafgyt_combo | 1.273e+06 | 691900.21 | 100.00 | flagged |
| Doorbell | gafgyt_tcp | 2.581 | 1.40 | 99.13 | flagged |
| Doorbell | mirai_syn | 4.089e+06 | 2221866.07 | 100.00 | flagged |
| Thermostat | benign_typical | 0.1996 | 0.16 | 52.13 | normal |
| Thermostat | gafgyt_combo | 6.035e+05 | 489733.96 | 100.00 | flagged |
| Thermostat | gafgyt_tcp | 0.6492 | 0.53 | 97.27 | elevated |
| Thermostat | mirai_syn | 1.875e+06 | 1521566.08 | 100.00 | flagged |
| BabyMonitor | benign_typical | 0.007243 | 0.01 | 57.23 | normal |
| BabyMonitor | gafgyt_combo | 829.8 | 1004.14 | 100.00 | flagged |
| BabyMonitor | gafgyt_tcp | 0.3723 | 0.45 | 97.17 | elevated |
| BabyMonitor | mirai_syn | 1.156e+04 | 13992.84 | 100.00 | flagged |
| Camera | benign_typical | 0.04638 | 0.03 | 49.30 | normal |
| Camera | gafgyt_combo | 996.1 | 623.58 | 100.00 | flagged |
| Camera | gafgyt_tcp | 0.5535 | 0.35 | 90.32 | normal |
| Camera | mirai_syn | 1460 | 914.05 | 100.00 | flagged |

The same `gafgyt_tcp` window is flagged on the Doorbell, elevated but under the threshold on the Thermostat and Baby Monitor, and in the upper tail of normal on the Camera. This matches the experiments: the Doorbell catches these attacks, and the other three devices miss almost all of them at the alert threshold.

An observation from `explain_window` on the `gafgyt_tcp` window: on the Doorbell, Thermostat and Camera, the largest error contributions are 1-minute-window values with packet count 1, mean packet size 60 and variance 0, which looks like a window summarizing a single small packet. On the Baby Monitor the top features are 10-second-window packet counts of 1. This fits the finding that these attack files have only about 20 distinct rows, but I have not tested why the model scores these windows so close to normal.

In Claude Desktop, scoring the Camera's `gafgyt_tcp` window (a missed attack, with the blind spot reported):

![Claude Desktop scoring the Camera gafgyt_tcp window](../images/mcp_demo.png)

And the Thermostat's `mirai_syn` window (flagged by a large margin):

![Claude Desktop scoring the Thermostat mirai_syn window](../images/mcp_demo_flagged.png)

## Checked against the LLM's answers

I compared Claude Desktop's answers with the tool output. Scores, thresholds, percentiles and blind-spot rates matched. In some follow-up explanations Claude added claims the tool output does not support (for example, naming an attack subtype, or explaining device differences with a threshold comparison that the numbers contradict). The server's instructions and its explanation output now tell the LLM to report only the returned numbers and to label any reasoning as an untested guess. That reduces the problem but does not remove it, since these are requests to the model, not enforcement.

## Limitations

These are also returned by `get_model_card`.

- Trained and evaluated on one lab dataset (N-BaIoT), 4 of 9 devices. Not validated in the field.
- Metrics are for this single exported model (seed 0), not the 3-seed means in the main README. Benign recall is measured on the last 30% of each benign capture, from the same session as training.
- `gafgyt_tcp` and `gafgyt_udp` are near-copy attack files (about 20 distinct rows each). They rank above most normal windows but fall under the alert threshold on 3 of 4 devices.
- A "normal" verdict does not prove a window is safe.
- No privacy analysis of the federated training (no differential privacy or secure aggregation).