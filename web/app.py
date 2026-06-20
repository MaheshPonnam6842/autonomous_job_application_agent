import os
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from v_final.export import build_ats_docx
from v_final.graph.analysis_graph import build_analysis_graph
from v_final.graph.rewrite_graph import build_rewrite_graph

# App setup

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DOWNLOAD_FOLDER = os.path.join("artifacts", "resumes")
ALLOWED_EXTENSIONS = {"txt", "pdf"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB upload cap
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

analysis_graph = build_analysis_graph()
rewrite_graph = build_rewrite_graph()


def _warm_model():
    """Load the model into RAM at startup so the first analyze isn't a cold load."""
    try:
        from v_final.llm import get_client
        get_client().chat("Reply with ok.", "ok", num_predict=1)
        print("[warmup] model loaded")
    except Exception as exc:  # noqa: BLE001
        print(f"[warmup] skipped: {exc}")


threading.Thread(target=_warm_model, daemon=True).start()

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
            t0 = time.perf_counter()
            result = rewrite_graph.invoke(state)
            elapsed = round(time.perf_counter() - t0, 1)

            # Build the ATS-safe .docx from the structured resume.
            docx_path = None
            struct = result.get("optimized_resume_struct")
            if struct:
                try:
                    docx_path = build_ats_docx(struct, os.path.join(DOWNLOAD_FOLDER, f"resume_{job_id}.docx"))
                except Exception as ex:  # noqa: BLE001 - docx is a bonus, never fatal
                    print(f"[JOB {job_id}] docx build failed: {ex}")

            JOBS[job_id] = {
                "status": "done",
                "docx_path": docx_path,
                "result": {
                    "optimized_resume_text": result.get("optimized_resume_text"),
                    "outreach_dm_text": result.get("outreach_dm_text"),
                    "outreach_email_text": result.get("outreach_email_text"),
                    "rewrite_score_comparison": result.get("rewrite_score_comparison"),
                    "rewrite_attempts": result.get("rewrite_attempts"),
                    "optimized_ats_match_score": result.get("optimized_ats_match_score"),
                    "optimized_ats_pass_score": result.get("optimized_ats_pass_score"),
                    "optimized_ats_pass_label": result.get("optimized_ats_pass_label"),
                    "resume_version": result.get("resume_version"),
                    "bullets_needing_metric": result.get("bullets_needing_metric") or [],
                    "docx_available": bool(docx_path),
                    "elapsed_seconds": elapsed,
                },
            }
            print(f"[JOB {job_id}] Rewrite completed (docx={bool(docx_path)})")
        except Exception as e:  # noqa: BLE001 - surface failure to the poller
            JOBS[job_id] = {"status": "error", "error": str(e)}
            print(f"[JOB {job_id}] Rewrite failed: {e}")

    threading.Thread(target=run_rewrite, daemon=True).start()
    return {"job_id": job_id}


@app.route("/rewrite/status/<job_id>")
def rewrite_status(job_id):
    job = JOBS.get(job_id, {"status": "unknown"})
    # Don't ship the server-side file path to the client.
    return jsonify({k: v for k, v in job.items() if k != "docx_path"})


@app.route("/download/<job_id>")
def download(job_id):
    job = JOBS.get(job_id)
    path = job.get("docx_path") if job else None
    if not path or not os.path.exists(path):
        return jsonify({"error": "No document available for this job"}), 404
    return send_file(path, as_attachment=True, download_name="resume_ats.docx")


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
