test_state = {
    "resume_raw_text": "Python developer with experience in AWS and ML.",
    "job_description_text": "Looking for a data scientist with Python, SQL, and AWS.",
    "missing_skills": ["SQL"],
    "rewrite_required": True,
}

from nodes.resume_rewrite import resume_rewrite_node
out = resume_rewrite_node(test_state)

print(out["optimized_resume_text"])
