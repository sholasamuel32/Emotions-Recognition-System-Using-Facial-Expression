"""Export facial-model notebook figures and metric tables for documentation."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MoodSense_Facial_Expression_Model_Evaluation.ipynb"
OUTPUT_DIR = ROOT / "facial_evaluation_outputs"
FIGURE_NAMES = [
    "facial_figure_1_class_distribution.png",
    "facial_figure_2_sample_faces.png",
    "facial_figure_3_confusion_matrix.png",
    "facial_figure_4_per_class_metrics.png",
    "facial_figure_5_multiclass_roc_curves.png",
    "facial_figure_6_reliability_diagram.png",
    "facial_figure_7_high_confidence_errors.png",
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

    metadata = json.loads((ROOT / "models" / "facial_model_metadata.json").read_text(encoding="utf-8"))
    pd.DataFrame(metadata["classification_report"]).T.to_csv(
        OUTPUT_DIR / "facial_table_1_classification_report.csv"
    )
    pd.DataFrame(
        [{"metric": name, "value": value} for name, value in metadata["metrics"].items()]
    ).to_csv(OUTPUT_DIR / "facial_table_2_overall_metrics.csv", index=False)
    pd.DataFrame(metadata["confusion_matrix"], index=metadata["classes"], columns=metadata["classes"]).to_csv(
        OUTPUT_DIR / "facial_table_3_confusion_matrix.csv"
    )
    metrics = metadata["metrics"]
    summary = [
        "MOODSENSE FACIAL-EXPRESSION MODEL EVALUATION",
        "============================================",
        f"Model: {metadata['model_name']}",
        f"Evaluation data: {metadata['evaluation_split']} ({metadata['evaluation_rows']:,} images)",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Macro ROC-AUC OvR: {metrics['macro_roc_auc_ovr']:.4f}",
        "",
        "Boundary: visible-expression classification is not direct measurement of internal mood or diagnosis.",
    ]
    (OUTPUT_DIR / "facial_evaluation_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"Exported {len(images)} figures and three metric tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
