"""Local face detection and facial-expression inference for MoodSense.

The detector locates frontal faces with OpenCV. The largest detected face is
then classified by a compact CNN executed through ONNX Runtime. No image is
written to disk by this module.
"""

from __future__ import annotations

import base64
import binascii
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "facial_expression_model.onnx"
CASCADE_PATH = ROOT / "models" / "haarcascade_frontalface_default.xml"
FACIAL_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
MAX_IMAGE_BYTES = 3 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000

FACIAL_DETAILS = {
    "angry": {
        "label": "Angry expression",
        "emoji": "😠",
        "colour": "#CF5C59",
        "description": "The visible facial pattern is most similar to the model's angry-expression class.",
        "suggestion": "Pause before reacting and check whether the expression matches how you actually feel.",
    },
    "disgust": {
        "label": "Disgusted expression",
        "emoji": "🤢",
        "colour": "#73936A",
        "description": "The visible facial pattern is most similar to the model's disgust-expression class.",
        "suggestion": "Consider what in the present situation feels unpleasant or unacceptable to you.",
    },
    "fear": {
        "label": "Fearful expression",
        "emoji": "😨",
        "colour": "#8878C4",
        "description": "The visible facial pattern is most similar to the model's fear-expression class.",
        "suggestion": "Take a slow breath and identify whether there is a specific concern you can address.",
    },
    "happy": {
        "label": "Happy expression",
        "emoji": "😊",
        "colour": "#4DA784",
        "description": "The visible facial pattern is most similar to the model's happy-expression class.",
        "suggestion": "Notice what contributed to this positive moment and consider recording it in words.",
    },
    "sad": {
        "label": "Sad expression",
        "emoji": "😔",
        "colour": "#5F8FB8",
        "description": "The visible facial pattern is most similar to the model's sad-expression class.",
        "suggestion": "Be gentle with yourself and consider speaking with someone you trust.",
    },
    "surprise": {
        "label": "Surprised expression",
        "emoji": "😮",
        "colour": "#E7AB43",
        "description": "The visible facial pattern is most similar to the model's surprise-expression class.",
        "suggestion": "Give yourself a moment to understand the unexpected event before deciding what it means.",
    },
    "neutral": {
        "label": "Neutral expression",
        "emoji": "😐",
        "colour": "#788785",
        "description": "The visible facial pattern is most similar to the model's neutral-expression class.",
        "suggestion": "A neutral expression may hide many feelings; add a text reflection if you want more context.",
    },
}


class FacialAnalysisError(ValueError):
    """A safe, user-facing facial analysis error."""


_runtime_lock = threading.Lock()
_session: ort.InferenceSession | None = None
_input_name: str | None = None
_output_name: str | None = None
_face_detector: cv2.CascadeClassifier | None = None


def _load_runtime() -> tuple[ort.InferenceSession, str, str, cv2.CascadeClassifier]:
    global _session, _input_name, _output_name, _face_detector
    if not MODEL_PATH.exists() or not CASCADE_PATH.exists():
        raise RuntimeError("The facial model assets are missing. Re-extract the complete source package.")
    if _session is None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        _input_name = _session.get_inputs()[0].name
        _output_name = _session.get_outputs()[0].name
    if _face_detector is None:
        _face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))
        if _face_detector.empty():
            raise RuntimeError("The bundled frontal-face detector could not be loaded.")
    return _session, _input_name, _output_name, _face_detector


def decode_data_url(data_url: str) -> bytes:
    """Decode a browser image data URL with strict size and format checks."""
    if not isinstance(data_url, str) or "," not in data_url:
        raise FacialAnalysisError("Choose a JPG, PNG, or WEBP photo, or capture a webcam image.")
    header, encoded = data_url.split(",", 1)
    if not header.lower().startswith(("data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64")):
        raise FacialAnalysisError("Only JPG, PNG, and WEBP images are supported.")
    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3) + 8:
        raise FacialAnalysisError("The image is too large. Use a photo below 3 MB.")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FacialAnalysisError("The supplied image data is invalid.") from exc
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise FacialAnalysisError("The image is empty or larger than 3 MB.")
    return payload


def _decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FacialAnalysisError("The photo could not be read. Try another JPG, PNG, or WEBP image.")
    height, width = image.shape[:2]
    if height * width > MAX_IMAGE_PIXELS:
        raise FacialAnalysisError("The image resolution is too large. Use a photo below 16 megapixels.")
    return image


def _expanded_square(box: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x, y, width, height = (int(value) for value in box)
    side = int(max(width, height) * 1.18)
    center_x, center_y = x + width // 2, y + height // 2
    x1 = max(0, center_x - side // 2)
    y1 = max(0, center_y - side // 2)
    x2 = min(shape[1], x1 + side)
    y2 = min(shape[0], y1 + side)
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    return x1, y1, x2 - x1, y2 - y1


def classify_face_crop(gray_face: np.ndarray) -> np.ndarray:
    """Return the seven probabilities for a cropped grayscale face."""
    if gray_face is None or gray_face.size == 0:
        raise FacialAnalysisError("The detected face crop was empty.")
    resized = cv2.resize(gray_face, (64, 64), interpolation=cv2.INTER_CUBIC)
    tensor = ((resized.astype(np.float32) / 255.0) - 0.5) * 2.0
    tensor = tensor.reshape(1, 64, 64, 1)
    with _runtime_lock:
        session, input_name, output_name, _ = _load_runtime()
        probabilities = session.run([output_name], {input_name: tensor})[0][0].astype(float)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("The facial model returned an invalid probability distribution.")
    return probabilities / total


def predict_facial_expression(image_bytes: bytes) -> dict:
    """Detect the largest frontal face and classify its visible expression."""
    image = _decode_image(image_bytes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detection_gray = cv2.equalizeHist(gray)
    with _runtime_lock:
        _, _, _, detector = _load_runtime()
        boxes = detector.detectMultiScale(
            detection_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(50, 50),
        )
    if boxes is None or len(boxes) == 0:
        raise FacialAnalysisError(
            "No clear frontal face was detected. Face the camera, improve the lighting, and try again."
        )
    ordered_boxes = sorted(
        (tuple(int(value) for value in box) for box in boxes),
        key=lambda box: box[2] * box[3],
        reverse=True,
    )
    selected_box = _expanded_square(ordered_boxes[0], gray.shape)
    x, y, width, height = selected_box
    probabilities = classify_face_crop(gray[y : y + height, x : x + width])
    best_index = int(np.argmax(probabilities))
    label = FACIAL_LABELS[best_index]
    detail = FACIAL_DETAILS[label]
    probability_rows = [
        {
            "mood": class_label,
            "label": FACIAL_DETAILS[class_label]["label"],
            "probability": round(float(score) * 100, 2),
            "colour": FACIAL_DETAILS[class_label]["colour"],
        }
        for class_label, score in sorted(
            zip(FACIAL_LABELS, probabilities), key=lambda item: item[1], reverse=True
        )
    ]
    return {
        "mood": label,
        "confidence": round(float(probabilities[best_index]) * 100, 2),
        "probabilities": probability_rows,
        "details": detail,
        "signals": [f"{len(ordered_boxes)} face{'s' if len(ordered_boxes) != 1 else ''} detected", "Largest frontal face analysed"],
        "face_count": len(ordered_boxes),
        "face_box": {"x": x, "y": y, "width": width, "height": height},
        "model_name": "FER-2013 compact CNN (ONNX)",
        "needs_urgent_support": False,
        "modality": "facial expression",
    }
