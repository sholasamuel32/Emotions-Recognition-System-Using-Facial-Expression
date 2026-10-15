"""Reproducible FER-2013 PrivateTest evaluation utilities."""

from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from facial_service import FACIAL_LABELS, classify_face_crop


ROOT = Path(__file__).resolve().parent
DATA_URL = "https://huggingface.co/datasets/DerrickUnleashed/FER-2013/resolve/main/fer2013.tar.gz"
ARCHIVE_SHA256 = "bdc255c109a1bbf994e6a7dd02a6f4566eb21beaece7643cd988166ee3527a44"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_fer2013_csv(cache_directory: Path | None = None) -> Path:
    """Return FER-2013 CSV, downloading and checksum-verifying it when needed."""
    environment_path = os.getenv("FER2013_CSV")
    if environment_path and Path(environment_path).is_file():
        return Path(environment_path)

    cache = cache_directory or ROOT / "data" / "fer2013_cache"
    csv_path = cache / "fer2013" / "fer2013.csv"
    archive_path = cache / "fer2013.tar.gz"
    if csv_path.is_file():
        return csv_path
    cache.mkdir(parents=True, exist_ok=True)
    print("Downloading FER-2013 archive (approximately 92 MB) for evaluation...")
    urllib.request.urlretrieve(DATA_URL, archive_path)
    observed_digest = _sha256(archive_path)
    if observed_digest != ARCHIVE_SHA256:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError("FER-2013 archive checksum verification failed.")
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember("fer2013/fer2013.csv")
        destination = (cache / member.name).resolve()
        if cache.resolve() not in destination.parents:
            raise RuntimeError("Unsafe path found in FER-2013 archive.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError("FER-2013 CSV was missing from its archive.")
        with destination.open("wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
    return csv_path


def load_private_test(csv_path: Path) -> pd.DataFrame:
    """Load only the official 3,589-image PrivateTest partition."""
    selected_chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=5_000):
        selected = chunk.loc[chunk["Usage"] == "PrivateTest", ["emotion", "pixels", "Usage"]]
        if not selected.empty:
            selected_chunks.append(selected)
    data = pd.concat(selected_chunks, ignore_index=True)
    if len(data) != 3_589:
        raise RuntimeError(f"Expected 3,589 PrivateTest images but found {len(data):,}.")
    return data


def evaluate_private_test(data: pd.DataFrame, progress_every: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the bundled CNN on every PrivateTest face crop."""
    truth = data["emotion"].to_numpy(dtype=int)
    probabilities = np.empty((len(data), len(FACIAL_LABELS)), dtype=float)
    for index, pixels in enumerate(data["pixels"]):
        image = np.fromstring(pixels, sep=" ", dtype=np.uint8).reshape(48, 48)
        probabilities[index] = classify_face_crop(image)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"Evaluated {index + 1:,} / {len(data):,} images")
    predictions = probabilities.argmax(axis=1)
    return truth, predictions, probabilities


def expected_calibration_error(truth: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = predictions == truth
    boundaries = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value)


def evaluation_artifacts(truth: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    one_hot = label_binarize(truth, classes=np.arange(len(FACIAL_LABELS)))
    report = classification_report(
        truth,
        predictions,
        labels=np.arange(len(FACIAL_LABELS)),
        target_names=FACIAL_LABELS,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted")),
        "macro_roc_auc_ovr": float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")),
        "expected_calibration_error_10_bins": expected_calibration_error(truth, predictions, probabilities),
        "multiclass_brier_score": float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }
    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=np.arange(len(FACIAL_LABELS))).tolist(),
    }
