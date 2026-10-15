"""Train, compare, evaluate, and save MoodSense text classifiers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from generate_dataset import write_dataset


RANDOM_STATE = 42


def make_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=20_000,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Multinomial Naive Bayes": make_pipeline(MultinomialNB(alpha=0.35)),
        "Logistic Regression": make_pipeline(
            LogisticRegression(
                C=3.0,
                class_weight="balanced",
                max_iter=2_000,
                random_state=RANDOM_STATE,
            )
        ),
        "SGD Log-Loss Classifier": make_pipeline(
            SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                class_weight="balanced",
                max_iter=2_000,
                random_state=RANDOM_STATE,
                tol=1e-4,
            )
        ),
    }


def metric_row(name: str, truth, prediction) -> dict[str, float | str]:
    macro = precision_recall_fscore_support(truth, prediction, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(truth, prediction, average="weighted", zero_division=0)
    return {
        "model": name,
        "accuracy": accuracy_score(truth, prediction),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_f1": weighted[2],
    }


def train_and_save(data_path: Path, model_path: Path, metadata_path: Path) -> dict:
    if not data_path.exists():
        write_dataset(data_path)
    frame = pd.read_csv(data_path).dropna(subset=["text", "mood"])
    frame = frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
    train_text, test_text, train_labels, test_labels = train_test_split(
        frame["text"],
        frame["mood"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=frame["mood"],
    )

    fitted: dict[str, Pipeline] = {}
    results: list[dict] = []
    for name, pipeline in candidate_models().items():
        pipeline.fit(train_text, train_labels)
        prediction = pipeline.predict(test_text)
        fitted[name] = pipeline
        results.append(metric_row(name, test_labels, prediction))

    comparison = pd.DataFrame(results).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).reset_index(drop=True)
    selected_name = str(comparison.iloc[0]["model"])
    selected = fitted[selected_name]
    selected_prediction = selected.predict(test_text)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, model_path)
    metadata = {
        "project": "MoodSense AI Local",
        "model_name": selected_name,
        "model_file": model_path.name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "dataset_rows": int(len(frame)),
        "training_rows": int(len(train_text)),
        "test_rows": int(len(test_text)),
        "classes": sorted(frame["mood"].unique().tolist()),
        "dataset_note": "Synthetic, curated demonstration data; not clinical data.",
        "metrics": {
            key: round(float(value), 6)
            for key, value in comparison.iloc[0].items()
            if key != "model"
        },
        "classification_report": classification_report(
            test_labels, selected_prediction, output_dict=True, zero_division=0
        ),
        "candidate_results": comparison.round(6).to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MoodSense classifiers.")
    parser.add_argument("--data", type=Path, default=Path("data/mood_text_dataset.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/mood_pipeline.joblib"))
    parser.add_argument("--metadata", type=Path, default=Path("models/model_metadata.json"))
    args = parser.parse_args()
    details = train_and_save(args.data, args.model, args.metadata)
    print(json.dumps({"selected_model": details["model_name"], "metrics": details["metrics"]}, indent=2))
