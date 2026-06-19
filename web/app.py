import os
import threading
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from v_final.graph.analysis_graph import build_analysis_graph
from v_final.graph.rewrite_graph import build_rewrite_graph

# App setup

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"txt", "pdf"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB upload cap
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

analysis_graph = build_analysis_graph()
rewrite_graph = build_rewrite_graph()

# In-memory job store for async rewrite (single-worker deployment).
JOBS: dict[str, dict] = {}


# Helpers

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_resume_file(filepath: str) -> str:
    if filepath.endswith(".txt"):
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            return f.read()

    if filepath.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return ""


# Routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    mode = request.form.get("mode")
    jd_text = request.form.get("jd_text", "")

    if not jd_text:
        return jsonify({"error": "Job description required"}), 400

    # Resume input handling
    if mode == "upload":
        file = request.files.get("resume_file")
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid resume file"}), 400

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)
        resume_text = read_resume_file(path)
    else:
        resume_text = request.form.get("resume_text", "")

    initial_state = {
        "resume_raw_text": resume_text,
        "job_description_text": jd_text,
        "resume_source": mode,
    }

    final_state = analysis_graph.invoke(initial_state)

    return jsonify({
        "__state__": final_state,
        "overall_match_score": final_state.get("overall_match_score"),
        "score_breakdown": final_state.get("score_breakdown"),
        "matched_skills": final_state.get("matched_skills"),
        "missing_required_skills": final_state.get("missing_required_skills"),
        "rewrite_required": final_state.get("rewrite_required"),
        "rewrite_strategy": final_state.get("rewrite_strategy"),
        "rewrite_reason": final_state.get("rewrite_reason"),
    })


@app.route("/rewrite", methods=["POST"])
def rewrite():
    state = request.json
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running"}

    def run_rewrite():
        try:
            print(f"[JOB {job_id}] Rewrite started")
            result = rewrite_graph.invoke(state)
            JOBS[job_id] = {
                "status": "done",
                "result": {
                    "optimized_resume_text": result.get("optimized_resume_text"),
                    "outreach_dm_text": result.get("outreach_dm_text"),
                    "outreach_email_text": result.get("outreach_email_text"),
                },
            }
            print(f"[JOB {job_id}] Rewrite completed")
        except Exception as e:  # noqa: BLE001 - surface failure to the poller
            JOBS[job_id] = {"status": "error", "error": str(e)}
            print(f"[JOB {job_id}] Rewrite failed: {e}")

    threading.Thread(target=run_rewrite, daemon=True).start()
    return {"job_id": job_id}


@app.route("/rewrite/status/<job_id>")
def rewrite_status(job_id):
    return jsonify(JOBS.get(job_id, {"status": "unknown"}))


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
