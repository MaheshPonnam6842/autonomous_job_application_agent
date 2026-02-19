from v_final.nodes.jd_ingest import jd_ingest_node

SAMPLE_JD = """
Description

As part of the AWS Applied AI Solutions organization, we have a mission to build delightful AI solutions that improve human capabilities and business outcomes. We will accomplish this by accelerating our customers' businesses through delivery of intuitive and differentiated AI solutions that solve enduring business challenges. We blend vision with curiosity and Amazon's real-world experience to build opinionated, turnkey solutions. Where customers prefer to buy over build, we become their trusted partner with solutions that are no-brainers to buy and easy to use.

The Applied Scientist will contribute to the development of Agentic AI solutions leveraging Gen AI. The role requires extending existing scientific techniques and inventing new ones to address specific customer needs. You should be comfortable working semi-autonomously on difficult problems with visible risks or roadblocks. You'll work closely with technical leaders within the team. We're looking for scientists who can maintain high standards while moving quickly, prioritizing both rapid experimentation and responsible AI development to deliver measurable customer impact.

Key job responsibilities

 Design and implement solutions that extend or adapt scientific approaches for customer needs
 Deliver components into production that meet high quality standards (efficient, reproducible, testable code)
 Produce science outputs demonstrating correctness, scholarship, and scientific rigor
 Work with product, engineering, and science teams to deliver impactful AI features
 Contribute to operational excellence in the team's deliverables

About The Team

AWS Solutions

As part of the AWS solutions organization, we have a vision to provide business applications, leveraging Amazon's unique experience and expertise, that are used by millions of companies worldwide to manage day-to-day operations. We will accomplish this by accelerating our customers' businesses through delivery of intuitive and differentiated technology solutions that solve enduring business challenges. We blend vision with curiosity and Amazon's real-world experience to build opinionated, turnkey solutions. Where customers prefer to buy over build, we become their trusted partner with solutions that are no-brainers to buy and easy to use.

Basic Qualifications

 3+ years of building models for business application experience
 PhD, or Master's degree and 4+ years of CS, CE, ML or related field experience
 Experience programming in Java, C++, Python or related language
 Experience in any of the following areas: algorithms and data structures, parsing, numerical optimization, data mining, parallel and distributed computing, high-performance computing

Preferred Qualifications

 Experience building machine learning models or developing algorithms for business application

Amazon is an equal opportunity employer and does not discriminate on the basis of protected veteran status, disability, or other legally protected status.

Our inclusive culture empowers Amazonians to deliver the best results for our customers. If you have a disability and need a workplace accommodation or adjustment during the application and hiring process, including support for the interview or onboarding process, please visit https://amazon.jobs/content/en/how-we-hire/accommodations for more information. If the country/region you’re applying in isn’t listed, please contact your Recruiting Partner.

The base salary range for this position is listed below. Your Amazon package will include sign-on payments and restricted stock units (RSUs). Final compensation will be determined based on factors including experience, qualifications, and location. Amazon also offers comprehensive benefits including health insurance (medical, dental, vision, prescription, Basic Life & AD&D insurance and option for Supplemental life plans, EAP, Mental Health Support, Medical Advice Line, Flexible Spending Accounts, Adoption and Surrogacy Reimbursement coverage), 401(k) matching, paid time off, and parental leave. Learn more
"""

def main():
    state = {
        "job_description_text": SAMPLE_JD
    }

    out = jd_ingest_node(state)

    print("\n=== JD INGEST OUTPUT ===")
    print("jd_title:", out.get("jd_title"))
    print("jd_domain:", out.get("jd_domain"))
    print("jd_seniority_level:", out.get("jd_seniority_level"))

    print("\nrequired:", out.get("jd_skills_required"))
    print("preferred:", out.get("jd_skills_preferred"))
    print("tools_process:", out.get("jd_tools_process"))
    print("\nresponsibilities:", out.get("jd_responsibilities"))
    print("\nkeywords:", out.get("jd_keywords"))
    print("\nwarnings:", out.get("warnings"))

if __name__ == "__main__":
    main()
