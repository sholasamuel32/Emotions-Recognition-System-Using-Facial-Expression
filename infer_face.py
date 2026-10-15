"""Command-line inference for one facial image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from facial_service import FacialAnalysisError, predict_facial_expression


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect the largest frontal face and classify its visible expression."
    )
    parser.add_argument("image", type=Path, help="Path to a JPG, PNG, or WEBP image")
    parser.add_argument("--json", action="store_true", help="Print the complete result as JSON")
    arguments = parser.parse_args()
    if not arguments.image.is_file():
        parser.error(f"Image not found: {arguments.image}")
    try:
        result = predict_facial_expression(arguments.image.read_bytes())
    except FacialAnalysisError as exc:
        parser.error(str(exc))
    if arguments.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Expression: {result['details']['label']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Faces detected: {result['face_count']}")
    print("Probability distribution:")
    for item in result["probabilities"]:
        print(f"  {item['label']:<24} {item['probability']:>6.2f}%")


if __name__ == "__main__":
    main()
