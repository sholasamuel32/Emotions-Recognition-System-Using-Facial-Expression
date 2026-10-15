"""Evaluate the deployed facial-expression CNN on FER-2013 PrivateTest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from facial_evaluation import (
    ensure_fer2013_csv,
    evaluate_private_test,
    evaluation_artifacts,
    load_private_test,
)
from facial_service import FACIAL_LABELS, MODEL_PATH


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "models" / "facial_model_metadata.json"


def main() -> None:
    csv_path = ensure_fer2013_csv()
    data = load_private_test(csv_path)
    truth, predictions, probabilities = evaluate_private_test(data)
    artifacts = evaluation_artifacts(truth, predictions, probabilities)
    metadata = {
        "project": "MoodSense AI Local — Facial Expression Engine",
        "model_name": "FER-2013 compact CNN (ONNX)",
        "model_file": MODEL_PATH.name,
        "classes": FACIAL_LABELS,
        "evaluation_split": "FER-2013 PrivateTest",
        "evaluation_rows": len(data),
        "important_boundary": "Facial expression classification is not direct measurement of internal mood or a diagnosis.",
        **artifacts,
    }
    OUTPUT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = pd.DataFrame(metadata["classification_report"]).T
    print(report.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nMetrics")
    for name, value in metadata["metrics"].items():
        print(f"{name}: {value:.6f}")
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
