"""Runtime model loading, inference, explanation, and safety messaging."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "mood_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

MOOD_DETAILS = {
    "happy": {
        "label": "Happy", "emoji": "😊", "colour": "#E7AB43",
        "description": "Your words reflect joy, satisfaction, or gratitude.",
        "suggestion": "Pause and notice what contributed to this positive moment.",
    },
    "calm": {
        "label": "Calm", "emoji": "😌", "colour": "#4DA784",
        "description": "Your language suggests ease, balance, and emotional steadiness.",
        "suggestion": "Protect this calm with a short breath or mindful pause.",
    },
    "excited": {
        "label": "Excited", "emoji": "🤩", "colour": "#DE784D",
        "description": "Your message carries anticipation, energy, or enthusiasm.",
        "suggestion": "Channel the energy into one clear next action.",
    },
    "anxious": {
        "label": "Anxious", "emoji": "😟", "colour": "#8878C4",
        "description": "Your words may reflect worry, pressure, or uncertainty.",
        "suggestion": "Name one concern you can act on and one you can release.",
    },
    "sad": {
        "label": "Sad", "emoji": "😔", "colour": "#5F8FB8",
        "description": "Your message suggests disappointment, loss, or low energy.",
        "suggestion": "Consider reaching out to someone safe or doing one gentle thing for yourself.",
    },
    "angry": {
        "label": "Angry", "emoji": "😠", "colour": "#CF5C59",
        "description": "Your language shows frustration, irritation, or a sense of unfairness.",
        "suggestion": "Pause before responding, then name the boundary you need.",
    },
    "neutral": {
        "label": "Neutral", "emoji": "🙂", "colour": "#788785",
        "description": "Your wording is mainly factual or emotionally balanced.",
        "suggestion": "For a deeper reading, add what the situation means to you.",
    },
}

URGENT_PATTERN = re.compile(
    r"\b(kill myself|suicide|suicidal|end my life|hurt myself|harm myself|"
    r"hurt someone|harm someone)\b",
    flags=re.IGNORECASE,
)

_pipeline = None
_metadata = None


def load_assets():
    global _pipeline, _metadata
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "The trained model is missing. Run: python train_model.py"
            )
        _pipeline = joblib.load(MODEL_PATH)
    if _metadata is None:
        _metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return _pipeline, _metadata


def _top_signals(pipeline, text: str, predicted_mood: str, limit: int = 5) -> list[str]:
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    vector = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    class_index = list(classifier.classes_).index(predicted_mood)

    if hasattr(classifier, "coef_"):
        weights = classifier.coef_[class_index]
    elif hasattr(classifier, "feature_log_prob_"):
        weights = classifier.feature_log_prob_[class_index] - classifier.feature_log_prob_.mean(axis=0)
    else:
        return []

    contributions = vector.toarray()[0] * np.asarray(weights)
    positive = np.flatnonzero(contributions > 0)
    ranked = positive[np.argsort(contributions[positive])[::-1]] if len(positive) else []
    return [str(feature_names[index]) for index in ranked[:limit]]


def predict_mood(text: str) -> dict:
    pipeline, metadata = load_assets()
    predicted = str(pipeline.predict([text])[0])
    probabilities = pipeline.predict_proba([text])[0]
    ordered = sorted(
        [
            {
                "mood": str(label),
                "label": MOOD_DETAILS[str(label)]["label"],
                "probability": round(float(probability) * 100, 2),
                "colour": MOOD_DETAILS[str(label)]["colour"],
            }
            for label, probability in zip(pipeline.classes_, probabilities)
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "mood": predicted,
        "confidence": ordered[0]["probability"],
        "details": MOOD_DETAILS[predicted],
        "probabilities": ordered,
        "signals": _top_signals(pipeline, text, predicted),
        "needs_urgent_support": bool(URGENT_PATTERN.search(text)),
        "model_name": metadata["model_name"],
    }
