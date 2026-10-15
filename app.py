"""MoodSense AI local Flask application."""

from __future__ import annotations

import csv
import io
import json
import os
import secrets
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from database import get_db, init_app as init_database
from facial_service import (
    FACIAL_DETAILS,
    FacialAnalysisError,
    decode_data_url,
    predict_facial_expression,
)
from model_service import MOOD_DETAILS, predict_mood


ALL_MOOD_DETAILS = {**FACIAL_DETAILS, **MOOD_DETAILS}


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    project_root = Path(__file__).resolve().parent
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "local-development-key-change-me"),
        DATABASE=str(Path(app.instance_path) / "moodsense.db"),
        MAX_CONTENT_LENGTH=5 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    init_database(app)

    @app.before_request
    def load_user_and_protect_forms():
        user_id = session.get("user_id")
        g.user = (
            get_db().execute(
                "SELECT id, full_name, email, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if user_id
            else None
        )
        session.setdefault("csrf_token", secrets.token_hex(24))
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            supplied = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
            if not supplied or not secrets.compare_digest(supplied, session["csrf_token"]):
                abort(400, "Invalid or missing security token.")

    @app.context_processor
    def inject_globals():
        return {
            "csrf_token": session.get("csrf_token", ""),
            "mood_details": ALL_MOOD_DETAILS,
        }

    def login_required(view):
        @wraps(view)
        def wrapped_view(**kwargs):
            if g.user is None:
                flash("Please sign in to continue.", "info")
                return redirect(url_for("login"))
            return view(**kwargs)
        return wrapped_view

    @app.get("/")
    def index():
        if g.user:
            return redirect(url_for("dashboard"))
        return render_template("index.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if g.user:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            full_name = " ".join(request.form.get("full_name", "").split())
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            error = None
            if len(full_name) < 2:
                error = "Please enter your full name."
            elif "@" not in email or len(email) > 254:
                error = "Please enter a valid email address."
            elif len(password) < 8:
                error = "Your password must contain at least eight characters."
            if error is None:
                try:
                    connection = get_db()
                    connection.execute(
                        "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
                        (full_name, email, generate_password_hash(password)),
                    )
                    connection.commit()
                except Exception as exc:
                    if "UNIQUE" in str(exc).upper():
                        error = "An account already exists for that email address."
                    else:
                        raise
            if error:
                flash(error, "error")
            else:
                flash("Account created. You can now sign in.", "success")
                return redirect(url_for("login"))
        return render_template("auth.html", mode="register")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if g.user:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user = get_db().execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)
            ).fetchone()
            if user is None or not check_password_hash(user["password_hash"], password):
                flash("The email address or password is incorrect.", "error")
            else:
                session.clear()
                session["user_id"] = user["id"]
                session["csrf_token"] = secrets.token_hex(24)
                return redirect(url_for("dashboard"))
        return render_template("auth.html", mode="login")

    @app.post("/logout")
    @login_required
    def logout():
        session.clear()
        return redirect(url_for("index"))

    @app.get("/dashboard")
    @login_required
    def dashboard():
        connection = get_db()
        records = connection.execute(
            """
            SELECT id, input_text, input_method, predicted_mood, confidence, created_at
            FROM mood_records WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC LIMIT 100
            """,
            (g.user["id"],),
        ).fetchall()
        counts = connection.execute(
            """
            SELECT predicted_mood, COUNT(*) AS count
            FROM mood_records WHERE user_id = ?
            GROUP BY predicted_mood ORDER BY count DESC
            """,
            (g.user["id"],),
        ).fetchall()
        total = sum(row["count"] for row in counts)
        average = connection.execute(
            "SELECT AVG(confidence) AS value FROM mood_records WHERE user_id = ?",
            (g.user["id"],),
        ).fetchone()["value"]
        insights = {
            "total": total,
            "average_confidence": round(float(average), 1) if average is not None else 0,
            "dominant_mood": counts[0]["predicted_mood"] if counts else None,
            "counts": [dict(row) for row in counts],
        }
        return render_template("dashboard.html", records=records, insights=insights)

    @app.post("/api/analyze")
    @login_required
    def analyze():
        payload = request.get_json(silent=True) or {}
        text = " ".join(str(payload.get("text", "")).split())
        input_method = "speech" if payload.get("input_method") == "speech" else "text"
        if len(text) < 3:
            return jsonify(error="Please provide at least three characters."), 400
        if len(text) > 2_000:
            return jsonify(error="Please keep the entry below 2,000 characters."), 400
        result = predict_mood(text)
        connection = get_db()
        cursor = connection.execute(
            """
            INSERT INTO mood_records
            (user_id, input_text, input_method, predicted_mood, confidence, probabilities_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                g.user["id"],
                text,
                input_method,
                result["mood"],
                result["confidence"],
                json.dumps(result["probabilities"]),
            ),
        )
        connection.commit()
        result["record_id"] = cursor.lastrowid
        result["saved"] = True
        return jsonify(result)

    @app.post("/api/analyze-face")
    @login_required
    def analyze_face():
        payload = request.get_json(silent=True) or {}
        source = "camera" if payload.get("source") == "camera" else "upload"
        try:
            image_bytes = decode_data_url(payload.get("image", ""))
            result = predict_facial_expression(image_bytes)
        except FacialAnalysisError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError:
            app.logger.exception("Facial inference engine failed")
            return jsonify(error="The local facial engine is unavailable. Verify the model files and restart the app."), 503

        input_method = f"face_{source}"
        input_summary = (
            "Facial expression captured with the webcam (image not stored)."
            if source == "camera"
            else "Facial expression analysed from an uploaded photo (image not stored)."
        )
        connection = get_db()
        cursor = connection.execute(
            """
            INSERT INTO mood_records
            (user_id, input_text, input_method, predicted_mood, confidence, probabilities_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                g.user["id"],
                input_summary,
                input_method,
                result["mood"],
                result["confidence"],
                json.dumps(result["probabilities"]),
            ),
        )
        connection.commit()
        result["record_id"] = cursor.lastrowid
        result["saved"] = True
        result["source"] = source
        return jsonify(result)

    @app.post("/history/<int:record_id>/delete")
    @login_required
    def delete_record(record_id: int):
        connection = get_db()
        connection.execute(
            "DELETE FROM mood_records WHERE id = ? AND user_id = ?",
            (record_id, g.user["id"]),
        )
        connection.commit()
        flash("The mood record was deleted.", "success")
        return redirect(url_for("dashboard", _anchor="history"))

    @app.post("/history/clear")
    @login_required
    def clear_history():
        connection = get_db()
        connection.execute("DELETE FROM mood_records WHERE user_id = ?", (g.user["id"],))
        connection.commit()
        flash("Your mood history was cleared.", "success")
        return redirect(url_for("dashboard"))

    @app.get("/history/export.csv")
    @login_required
    def export_history():
        rows = get_db().execute(
            """
            SELECT input_text, input_method, predicted_mood, confidence, created_at
            FROM mood_records WHERE user_id = ? ORDER BY datetime(created_at) DESC
            """,
            (g.user["id"],),
        ).fetchall()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["text", "input_method", "predicted_mood", "confidence_percent", "created_at"])
        for row in rows:
            writer.writerow(list(row))
        response = make_response(output.getvalue())
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        response.headers["Content-Disposition"] = "attachment; filename=moodsense_multimodal_history.csv"
        return response

    @app.errorhandler(400)
    def bad_request(error):
        if request.path.startswith("/api/"):
            return jsonify(error=str(error.description)), 400
        return render_template("error.html", code=400, message=str(error.description)), 400

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", code=404, message="The requested page was not found."), 404

    @app.errorhandler(413)
    def request_too_large(_error):
        if request.path.startswith("/api/"):
            return jsonify(error="The request is too large. Use an image below 3 MB."), 413
        return render_template("error.html", code=413, message="The submitted file is too large."), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=os.getenv("FLASK_DEBUG") == "1")
