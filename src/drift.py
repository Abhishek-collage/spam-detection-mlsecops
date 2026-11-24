import json
import os
import sys
from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd


def _extract_drifted_share(report_result) -> float:
    data = report_result.dict() if hasattr(report_result, "dict") else {}
    for metric in data.get("metrics", []):
        result = metric.get("result", {})
        if isinstance(result, dict) and "share_of_drifted_columns" in result:
            return float(result["share_of_drifted_columns"])
    return 0.0


def run_drift_check(current_data_path: str):
    ref_path = "artifacts/processed/processed_train.csv"
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference data nahi mila: {ref_path}")
    if not os.path.exists(current_data_path):
        raise FileNotFoundError(f"Current data nahi mila: {current_data_path}")

    ref = pd.read_csv(ref_path)
    if len(ref) > 5000:
        ref = ref.sample(5000, random_state=0)

    cur = pd.read_csv(current_data_path)

    cols = [c for c in ["Message", "Category"] if c in ref.columns and c in cur.columns]
    if not cols:
        raise ValueError("No common columns ('Message', 'Category') found between reference and current data")

    ref = ref[cols].fillna("")
    cur = cur[cols].fillna("")

    rep = Report(metrics=[DataDriftPreset()])
    res = rep.run(reference_data=ref, current_data=cur)

    os.makedirs("artifacts", exist_ok=True)
    res.save_html("artifacts/drift.html")

    drift = _extract_drifted_share(res)
    with open("artifacts/drift.json", "w", encoding="utf-8") as f:
        json.dump({"drifted_share": drift}, f, indent=2)

    print(f"Drift check complete! Drifted share: {drift:.0%}")

    if drift > 0.3:
        print(f"::warning::drift {drift:.0%} — possible data poisoning or concept drift detected!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/monitor/drift.py <path_to_current_csv>")
        sys.exit(1)
    run_drift_check(sys.argv[1])