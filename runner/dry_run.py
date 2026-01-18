from graph.job_application_graph import graph

# ---- Minimal realistic inputs ----
initial_state = {
    "resume_raw_text": """
    Data Scientist with 3+ years of experience in Python, machine learning, and AWS.
    Built end-to-end ML pipelines and deployed models using Docker.
    Experience with SQL and data analysis.
    """,

    "job_description_text": """
    We are looking for a Data Scientist with strong Python skills,
    experience in AWS, SQL, machine learning, and data pipelines.
    Experience deploying models is a plus.
    """,

    "resume_source": "manual_upload"
}

# ---- Run graph ----
final_state = graph.invoke(initial_state)

print("\n===== FINAL STATE =====\n")
for k, v in final_state.items():
    print(f"{k}: {v}\n")
