from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import create_app
from database import get_db
from facial_service import FACIAL_LABELS, FacialAnalysisError, classify_face_crop, decode_data_url
from model_service import predict_mood


class ModelTests(unittest.TestCase):
    def test_canonical_moods(self):
        cases = {
            "happy": "I am genuinely happy, grateful, and proud today.",
            "calm": "I feel peaceful, settled, and completely at ease.",
            "excited": "I cannot wait for this amazing new opportunity.",
            "anxious": "I am worried about the deadline and cannot relax.",
            "sad": "I feel lonely, heartbroken, and very low.",
            "angry": "Their unfair and disrespectful decision made me furious.",
            "neutral": "The meeting is scheduled for Monday at four.",
        }
        for expected, text in cases.items():
            with self.subTest(expected=expected):
                result = predict_mood(text)
                self.assertEqual(result["mood"], expected)
                self.assertAlmostEqual(
                    sum(item["probability"] for item in result["probabilities"]),
                    100,
                    delta=0.1,
                )

    def test_urgent_support_phrase_is_flagged(self):
        self.assertTrue(predict_mood("I want to hurt myself")["needs_urgent_support"])


class FacialModelTests(unittest.TestCase):
    def test_cnn_returns_normalized_seven_class_distribution(self):
        sample_face = np.tile(np.arange(48, dtype=np.uint8), (48, 1))
        probabilities = classify_face_crop(sample_face)
        self.assertEqual(probabilities.shape, (7,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)
        self.assertIn(FACIAL_LABELS[int(probabilities.argmax())], FACIAL_LABELS)

    def test_invalid_image_data_is_rejected(self):
        with self.assertRaises(FacialAnalysisError):
            decode_data_url("data:text/plain;base64,SGVsbG8=")


class WebApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_directory.name) / "test.db"
        self.app = create_app(
            {"TESTING": True, "SECRET_KEY": "test-secret", "DATABASE": str(database_path)}
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_directory.cleanup()

    def csrf(self):
        with self.client.session_transaction() as state:
            return state["csrf_token"]

    def register(self, name, email):
        self.client.get("/register")
        return self.client.post(
            "/register",
            data={
                "csrf_token": self.csrf(),
                "full_name": name,
                "email": email,
                "password": "SecurePass123!",
            },
            follow_redirects=True,
        )

    def login(self, email):
        self.client.get("/login")
        return self.client.post(
            "/login",
            data={
                "csrf_token": self.csrf(),
                "email": email,
                "password": "SecurePass123!",
            },
            follow_redirects=True,
        )

    def logout(self):
        return self.client.post(
            "/logout", data={"csrf_token": self.csrf()}, follow_redirects=True
        )

    def test_registration_login_prediction_and_user_isolation(self):
        response = self.register("First User", "first@example.com")
        self.assertIn(b"Account created", response.data)
        self.login("first@example.com")

        response = self.client.post(
            "/api/analyze",
            json={"text": "I am happy and grateful today", "input_method": "text"},
            headers={"X-CSRF-Token": self.csrf()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["saved"])
        self.logout()

        self.register("Second User", "second@example.com")
        self.login("second@example.com")
        with self.app.app_context():
            users = get_db().execute("SELECT id, email FROM users ORDER BY id").fetchall()
            first_count = get_db().execute(
                "SELECT COUNT(*) FROM mood_records WHERE user_id = ?", (users[0]["id"],)
            ).fetchone()[0]
            second_count = get_db().execute(
                "SELECT COUNT(*) FROM mood_records WHERE user_id = ?", (users[1]["id"],)
            ).fetchone()[0]
        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 0)
        dashboard = self.client.get("/dashboard")
        self.assertNotIn(b"happy and grateful today", dashboard.data)

    def test_facial_prediction_is_saved_without_storing_image(self):
        self.register("Face User", "face@example.com")
        self.login("face@example.com")
        one_pixel_png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        ).decode("ascii")
        mocked_result = {
            "mood": "happy",
            "confidence": 91.2,
            "probabilities": [
                {"mood": "happy", "label": "Happy expression", "probability": 91.2, "colour": "#4DA784"}
            ],
            "details": {"label": "Happy expression", "emoji": "😊", "colour": "#4DA784", "description": "Test", "suggestion": "Test"},
            "signals": ["1 face detected"],
            "model_name": "FER-2013 compact CNN (ONNX)",
            "needs_urgent_support": False,
            "modality": "facial expression",
        }
        with patch("app.predict_facial_expression", return_value=mocked_result):
            response = self.client.post(
                "/api/analyze-face",
                json={"image": f"data:image/png;base64,{one_pixel_png}", "source": "camera"},
                headers={"X-CSRF-Token": self.csrf()},
            )
        self.assertEqual(response.status_code, 200)
        with self.app.app_context():
            row = get_db().execute(
                "SELECT input_text, input_method, predicted_mood FROM mood_records"
            ).fetchone()
        self.assertEqual(row["input_method"], "face_camera")
        self.assertEqual(row["predicted_mood"], "happy")
        self.assertIn("image not stored", row["input_text"])
        self.assertNotIn(one_pixel_png[:20], row["input_text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
