"""Replay the built-in demo windows through the detector (no MCP client needed).

    python demo.py            # table of all devices x windows
    python demo.py --explain  # also print the top features for each window
"""
import argparse

from detector import Detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", action="store_true", help="print top features for each window")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    det = Detector()
    print("| Device | Window | Score | x alert threshold | Benign percentile | Verdict |")
    print("|---|---|---|---|---|---|")
    details = []
    for dev in det.devices.values():
        for name in sorted(det.demo.get(dev.short, {})):
            r = det.score(dev.short, demo_window=name)
            print(f"| {dev.short} | {name} | {r['anomaly_score']:.4g} | {r['score_over_alert_threshold']:.2f} | "
                  f"{r['benign_percentile']:.2f} | {r['band']} |")
            if args.explain:
                details.append((dev.short, name, det.explain(dev.short, demo_window=name, top_k=args.top_k)))

    for dev, name, e in details:
        print(f"\n### {dev} / {name}: {e['interpretation']}")
        for t in e["top_features"]:
            print(f"- {t['feature']}: observed {t['observed_value']:.4g}, model expected "
                  f"{t['model_expected_value']:.4g}, {t['deviation_from_device_normal_in_std']:+.1f} std from "
                  f"normal, {t['share_of_total_error_pct']}% of error ({t['meaning']})")
        print(f"- error by stream: {e['error_share_by_stream_pct']}")
        bs = e["known_blind_spots_on_this_device"]
        if bs:
            print(f"- known blind spots on {dev}: {'; '.join(bs)}")


if __name__ == "__main__":
    main()
