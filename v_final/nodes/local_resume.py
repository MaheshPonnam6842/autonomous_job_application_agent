from v_final.nodes.resume_ingest import resume_ingest_node

resume_text = """
MAHESH PONNAM
+1 (609) 379-3629  | maheshponnam723@gmail.com  | linkedin.com/in/maheshponnam3629/  | github.com/MaheshPonnam6842 
PROFESSIONAL SUMMARY
Data Scientist with 3+ years of experience in credit risk and fraud analytics, building and deploying ML solutions that improve underwriting 
decisions, reduce losses, and strengthen model governance. Strong analytical foundation in Python and SQL, with hands on experience across the 
full model lifecycle including feature engineering, training, validation, monitoring, and production deployment on AWS. Comfortable partnering 
with cross functional stakeholders to translate risk strategy needs into measurable analytics and decision ready insights.
SKILLS
GenAI & NLP: RAG, Agentic AI workflows (LangGraph), Prompt Engineering, PyTorch
Machine Learning: Classification/Regression, Ensemble Models, Feature Engineering, Model Validation, A/B Testing, Credit risk modeling, 
fraud analytic
Cloud & MLOps: AWS (Lambda, EC2, SageMaker, S3), Docker, CI/CD (GitHub Actions), Model Monitoring
Data Engineering: PySpark, Snowflake, SQL, AWS Glue
Explainability & Analytics: SHAP, LIME, Power BI
Domain: Credit Risk, Fraud Detection, Workflow Automation
PROFESSIONAL EXPERIENCE
Northern Trust
Data Scientist
USA
February 2025 - Present
•  Built ESG and factor data pipelines using Python, Snowflake, and AWS Glue to improve data accuracy by 18 percent and reduce manual 
reconciliation by 20 percent.
•  Developed NLP and ML models to process unstructured disclosures using PyTorch and SageMaker, accelerating analytics workflows by 
28 percent.
•  Designed near real time ingestion using AWS Lambda and S3 to enable scalable analytics across datasets used for risk and reporting.
•  Integrated model driven insights into dashboards to surface key drivers and improve decision speed by 33 percent.
Mphasis
Data Scientist
India
April 2020 - November 2022
•  Led development of credit risk and fraud models that improved approval accuracy by 18 percent and reduced fraudulent transactions by 15 
percent.
•  Built scalable ETL pipelines using PySpark and SQL to support model training and validation across large datasets.
•  Deployed production ML services on AWS Lambda and EC2 with monitoring, reducing inference latency by 20 percent.
•  Implemented explainability workflows using SHAP to support model governance and improve stakeholder trust in risk decisions.
PROJECTS & OUTSIDE EXPERIENCE
End to End Credit Risk Prediction System-  Link to project
•  Built an end to end credit risk system to predict probability of loan default, covering ingestion, preprocessing, training, evaluation, and 
production inference service.
•  Engineered domain features such as loan to income ratio, delinquency ratio, average DPD, and credit utilization, and handled class 
imbalance using SMOTE.
•  Trained and compared Logistic Regression, Random Forest, and XGBoost, selecting the best model using ROC AUC and producing 
probability based outputs.
•  Deployed a Dockerized Flask inference app and automated delivery using GitHub Actions, pushing images to Amazon ECR and deploying 
to EC2.
EDUCATION
Wilmington University
Master's, Information Systems
Vignana Bharathi Institute of Technology, India
Master's, Computer Engineering
July 2022 - December 2024
July 2018 - July 202
"""

state = {"resume_raw_text": resume_text, "resume_source": "local_test"}
state = resume_ingest_node(state)

print("\n=== resume_sections ===")
for k, v in (state.get("resume_sections") or {}).items():
    print(f"\n[{k}]")
    print(v if v else "(empty)")

print("\n=== experience_bullets_by_group ===")
exp = state.get("experience_bullets_by_group") or {}
for grp, bullets in exp.items():
    print(f"\n[{grp}]")
    for b in bullets:
        print("  -", b)

print("\n=== experience_bullets (flat) ===")
for b in (state.get("experience_bullets") or []):
    print("  -", b)

print("\n=== project_bullets_by_group ===")
proj = state.get("project_bullets_by_group") or {}
for grp, bullets in proj.items():
    print(f"\n[{grp}]")
    for b in bullets:
        print("  -", b)

print("\n=== resume_skills_structured ===")
print(state.get("resume_skills_structured") or {})
print("\n---- resume_links (from PDF annotations) ----")
print(state.get("resume_links", []))