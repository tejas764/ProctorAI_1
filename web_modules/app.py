from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from voice_features import extract_voice_features
from web_modules.enrollment import EnrollmentApi, read_wav_bytes, save_wav, utc_now_iso
from web_modules.monitoring import MonitoringWorker


def create_app() -> Flask:
    project_root = Path(__file__).resolve().parents[1]
    app = Flask(__name__, template_folder=str(project_root / "templates"))
    enrollment_api = EnrollmentApi()
    monitor = MonitoringWorker(store=enrollment_api.store)

    @app.get("/")
    def home() -> str:
        return render_template("voice_enrollment.html")

    @app.get("/monitor")
    def monitor_page() -> str:
        return render_template("monitoring.html")

    @app.get("/api/monitor/frame")
    def monitor_frame() -> object:
        jpg = monitor.get_latest_jpeg()
        if not jpg:
            return ("", 204)
        return app.response_class(jpg, mimetype="image/jpeg")

    @app.get("/api/monitor/stream")
    def monitor_stream() -> Response:
        def generate():
            last = b""
            while True:
                jpg = monitor.get_latest_jpeg()
                if not jpg:
                    jpg = last
                if jpg:
                    last = jpg
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                import time
                time.sleep(0.04)  # ~25 FPS push

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/enrollment/questions")
    def questions() -> object:
        user_id = request.args.get("user_id", "").strip()
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        profile = enrollment_api.store.load_profile(user_id)
        locked = bool(profile and profile.enrollment_complete)
        return jsonify({"locked": locked, "questions": enrollment_api.service.questions()})

    @app.get("/api/enrollment/status/<user_id>")
    def status(user_id: str) -> object:
        profile = enrollment_api.store.load_profile(user_id)
        if profile is None:
            return jsonify({"enrollment_complete": False})
        return jsonify(
            {
                "enrollment_complete": bool(profile.enrollment_complete),
                "completed_at": profile.completed_at,
                "base_threshold": profile.base_threshold,
                "drift_threshold": profile.drift_threshold,
            }
        )

    @app.post("/api/enrollment/recording")
    def upload_recording() -> object:
        user_id = request.form.get("user_id", "").strip()
        question_id = request.form.get("question_id", "").strip()
        timestamp_iso = request.form.get("timestamp", utc_now_iso()).strip()
        validation_json = request.form.get("validation", "{}")
        file = request.files.get("audio")

        if not user_id or not question_id or file is None:
            return jsonify({"error": "user_id, question_id and audio are required."}), 400

        profile = enrollment_api.store.load_profile(user_id)
        if profile and profile.enrollment_complete:
            return jsonify({"error": "Re-enrollment disabled. Admin reset required."}), 403

        try:
            validation = json.loads(validation_json)
        except json.JSONDecodeError:
            validation = {}

        raw = file.read()
        try:
            audio, sample_rate = read_wav_bytes(raw)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Invalid WAV audio: {exc}"}), 400

        if audio.size == 0:
            return jsonify({"error": "No audio input detected"}), 400

        audio_path = Path("proctor_data") / "enrollment_audio" / user_id / f"{question_id}_{timestamp_iso.replace(':', '-')}.wav"
        save_wav(audio, sample_rate, audio_path)

        features = extract_voice_features(audio, sample_rate)
        enrollment_api.store.save_question_sample(
            user_id=user_id,
            question_id=question_id,
            audio_path=str(audio_path),
            recorded_at=timestamp_iso,
            features=features,
        )

        return jsonify(
            {
                "ok": True,
                "question_id": question_id,
                "timestamp": timestamp_iso,
                "audio_path": str(audio_path),
                "feature_frame_count": features.frame_count,
                "validation": validation,
            }
        )

    @app.post("/api/enrollment/complete")
    def complete() -> object:
        payload = request.get_json(silent=True) or {}
        user_id = str(payload.get("user_id", "")).strip()
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        profile = enrollment_api.store.load_profile(user_id)
        if profile and profile.enrollment_complete:
            return jsonify({"error": "Already completed. Re-enrollment disabled."}), 409

        result = enrollment_api.finalize_enrollment(user_id)
        if not bool(result.get("enrollment_complete")):
            return jsonify(result), 400
        return jsonify(result)

    @app.post("/api/enrollment/admin/reset/<user_id>")
    def admin_reset(user_id: str) -> object:
        monitor.stop()
        enrollment_api.store.mark_incomplete(user_id)
        return jsonify({"ok": True, "user_id": user_id, "message": "Enrollment reset by admin."})

    @app.post("/api/monitor/start")
    def monitor_start() -> object:
        payload = request.get_json(silent=True) or {}
        user_id = str(payload.get("user_id", "")).strip()
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        ok, message = monitor.start(user_id)
        code = 200 if ok else 400
        return jsonify({"ok": ok, "message": message, "state": monitor.get_state()}), code

    @app.post("/api/monitor/stop")
    def monitor_stop() -> object:
        monitor.stop()
        return jsonify({"ok": True, "state": monitor.get_state()})

    @app.get("/api/monitor/gaze")
    def monitor_gaze_state() -> object:
        return jsonify(monitor.get_gaze_state())

    @app.post("/api/monitor/gaze/start-step")
    def monitor_gaze_start_step() -> object:
        ok, message, state = monitor.begin_gaze_calibration_step()
        code = 200 if ok else 400
        return jsonify({"ok": ok, "message": message, "state": state}), code

    @app.post("/api/monitor/gaze/reset")
    def monitor_gaze_reset() -> object:
        ok, message, state = monitor.reset_gaze_calibration()
        code = 200 if ok else 400
        return jsonify({"ok": ok, "message": message, "state": state}), code

    @app.get("/api/monitor/state")
    def monitor_state() -> object:
        return jsonify(monitor.get_state())

    return app
