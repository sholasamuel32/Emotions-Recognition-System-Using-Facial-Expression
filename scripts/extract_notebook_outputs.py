"""Export notebook figures and metric tables for direct use in documentation."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MoodSense_Model_Training_Evaluation.ipynb"
OUTPUT_DIR = ROOT / "evaluation_outputs"
FIGURE_NAMES = [
    "figure_1_class_distribution.png",
    "figure_2_model_comparison.png",
    "figure_3_confusion_matrix.png",
    "figure_4_multiclass_roc_curves.png",
    "figure_5_per_class_metrics.png",
    "figure_6_influential_features.png",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    images = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                images.append(base64.b64decode(encoded))
    if len(images) != len(FIGURE_NAMES):
        raise RuntimeError(f"Expected {len(FIGURE_NAMES)} figures but found {len(images)}")
    for name, payload in zip(FIGURE_NAMES, images):
        (OUTPUT_DIR / name).write_bytes(payload)

    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    pd.DataFrame(metadata["candidate_results"]).to_csv(
        OUTPUT_DIR / "table_1_model_comparison.csv", index=False
    )
    pd.DataFrame(metadata["classification_report"]).T.to_csv(
        OUTPUT_DIR / "table_2_classification_report.csv"
    )
    summary = [
        "MOODSENSE AI MODEL EVALUATION SUMMARY",
        "====================================",
        f"Selected model: {metadata['model_name']}",
        f"Dataset size: {metadata['dataset_rows']:,}",
        f"Training rows: {metadata['training_rows']:,}",
        f"Test rows: {metadata['test_rows']:,}",
        "",
    ]
    summary.extend(f"{key}: {value}" for key, value in metadata["metrics"].items())
    summary.extend(
        [
            "",
            "Interpretation boundary:",
            "These are prototype results on a balanced synthetic, curated dataset.",
            "They do not establish clinical or unrestricted real-world validity.",
        ]
    )
    (OUTPUT_DIR / "evaluation_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"Exported {len(images)} figures and evaluation tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
