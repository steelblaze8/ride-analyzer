"""
Ride Analyzer — minimal Flask web app
---------------------------------------
Upload a .fit or .gpx file, get an AI-generated coach summary back.

This is intentionally bare-bones: one upload form, one result page,
no database, no user accounts, no styling framework. Its only job
right now is to prove the whole pipeline works end-to-end in a browser
instead of the command line, so you (and eventually a couple of test
riders) can actually try it without touching Python.

Run:
    py -m pip install flask fitparse pandas requests
    py app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import os
import tempfile
import traceback

from flask import Flask, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from ride_parser import load_ride, compute_metrics
from ride_analyzer import analyze_ride

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap

# Basic abuse protection: since this uses YOUR OpenRouter key for every
# visitor, an unlimited endpoint means anyone could burn through your
# free-tier rate limits (or run up costs if you ever switch to paid
# models). This caps each visitor's IP to a modest number of analyses.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",  # fine for a single small instance; not for multi-instance scaling
)

ALLOWED_EXTENSIONS = {".fit", ".gpx"}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
@limiter.limit("10 per hour")
def analyze():
    uploaded_file = request.files.get("ride_file")
    ftp_raw = request.form.get("ftp", "").strip()
    ftp = int(ftp_raw) if ftp_raw.isdigit() else None

    if not uploaded_file or uploaded_file.filename == "":
        return render_template("index.html", error="Please choose a .fit or .gpx file.")

    ext = os.path.splitext(uploaded_file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return render_template("index.html", error=f"Unsupported file type '{ext}'. Use .fit or .gpx.")

    # Save to a temp file — the parser reads from a filesystem path.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        uploaded_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        records = load_ride(tmp_path)
        metrics = compute_metrics(records, ftp=ftp)
        warnings = metrics.pop("warnings", [])
        segments = metrics.get("segments")  # kept for optional display, not required
        summary = analyze_ride(metrics)
        return render_template(
            "result.html",
            summary=summary,
            metrics=metrics,
            warnings=warnings,
            filename=uploaded_file.filename,
        )
    except Exception as e:
        # Log the real error server-side for debugging, but show visitors
        # a generic message — a raw traceback/exception can leak file
        # paths, library versions, or other internals.
        traceback.print_exc()
        return render_template(
            "index.html",
            error="Something went wrong processing that file. Double-check it's a valid .fit or .gpx export and try again."
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    # For local testing only. In production, a WSGI server (gunicorn) runs
    # this instead — see Procfile — so debug mode never runs on the host.
    app.run(debug=True)