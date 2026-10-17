"""
Create the SYNTHETIC resume / job-description texts and the pair list.

Run from the project root:
    python scripts/make_synthetic_pairs.py

WHAT IT WRITES
    data/synthetic/resumes/resume_01.txt ... resume_10.txt   (10 resumes)
    data/synthetic/jds/jd_01.txt ... jd_06.txt                (6 job descriptions)
    data/labels/pairs.csv                                     (24 pairs, ratings EMPTY)

IMPORTANT, FOR THE REPORT
    * Every person, company and institute here is FICTIONAL, and the texts
      were written by the AI assistant, not taken from real people. The report
      must call this a synthetic evaluation set.
    * The 24 pairs were written to cover a range of situations (clear matches,
      partial matches, clear mismatches, over-qualified, long resumes, resumes
      that use different words for the same work). No judgement of how well a
      pair fits is stored anywhere in these files.
    * THE RATINGS ARE YOURS. After running this script, read each pair and
      fill the two empty columns of data/labels/pairs.csv yourself:
          human_score_0_to_100   how well does this resume fit this job?
          human_verdict          Fit | Partial Fit | No-Fit
      Suggested guide (decide by reading, not by the number):
          Fit          you would shortlist this person for an interview
          Partial Fit  some core requirements are met, but there are real gaps
          No-Fit       you would not shortlist this person for this job
      Keep the score and the verdict consistent with each other.

PATHS IN pairs.csv are relative to the PROJECT ROOT (forward slashes).

SAFETY
    The script refuses to run if data/labels/pairs.csv already exists, so it
    can never overwrite ratings you have typed. Use --force to override.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------- resumes
RESUMES = {
    "resume_01": """
ANANYA RAO
ananya.rao@example.com | +91 90000 00001

SUMMARY
Final-year B.Tech student in Computer Science with hands-on machine learning experience. Comfortable taking a model from raw data to a small deployed service, and keen to start as a junior ML engineer.

SKILLS
Python, NumPy, pandas, scikit-learn, PyTorch, SQL, Git, Docker, FastAPI, Jupyter

PROJECTS
- Plant disease classifier: trained a convolutional neural network in PyTorch on 30,000 leaf images, reached 91% validation accuracy, and compared it with a scikit-learn random forest baseline using precision, recall and F1.
- Customer churn predictor: cleaned a telecom dataset with pandas, engineered features, and evaluated logistic regression and gradient boosting with cross-validation.
- Model API: wrapped the churn model in a FastAPI REST service and packaged it with Docker.

EXPERIENCE
- Machine Learning Intern, Brightwave Labs (3 months): built a text-classification prototype for support tickets and wrote the evaluation notebook.

EDUCATION
B.Tech, Computer Science, Example Institute of Technology, expected 2027. CGPA 8.4
""",
    "resume_02": """
ROHAN MEHTA
rohan.mehta@example.com | +91 90000 00002

SUMMARY
Data analyst with 3 years of experience turning sales and operations data into weekly reports and dashboards for business teams.

SKILLS
SQL, Excel, Power BI, Python, pandas, statistics, A/B testing, data visualization, data cleaning

EXPERIENCE
Data Analyst, Lumen Retail (2023 to present)
- Write SQL queries against a PostgreSQL warehouse to produce weekly sales, returns and inventory reports.
- Built 6 Power BI dashboards used by regional managers; cut manual reporting time from two days to a few hours.
- Cleaned and joined messy supplier data with pandas and flagged duplicate records.
- Designed and analysed an A/B test on a checkout change and presented the result with confidence intervals.

Junior Analyst, Kestrel Systems (2022 to 2023)
- Maintained Excel models for monthly forecasting and reconciled them with finance.

EDUCATION
B.Sc Statistics, Sample University, 2022
""",
    "resume_03": """
KARTHIK IYER
karthik.iyer@example.com | +91 90000 00003

SUMMARY
Backend developer with 3 years of experience building Java microservices for payments and order processing.

SKILLS
Java, Spring Boot, REST APIs, microservices, PostgreSQL, SQL, Kafka, Docker, Git, JUnit, AWS

EXPERIENCE
Software Engineer, Orchid Software (2023 to present)
- Develop and maintain Spring Boot services that handle about 2 million requests a day.
- Design REST APIs and PostgreSQL schemas; tune slow queries and add indexes.
- Publish and consume Kafka events between order and payment services.
- Containerise services with Docker and deploy them to AWS ECS.
- Write JUnit tests and review pull requests in Git.

Associate Software Engineer, Orchid Software (2022 to 2023)
- Fixed defects in a legacy Java application and migrated it to Spring Boot.

EDUCATION
B.E. Information Technology, Example Institute of Technology, 2022
""",
    "resume_04": """
SNEHA KULKARNI
sneha.kulkarni@example.com | +91 90000 00004

SUMMARY
Frontend developer with 2 years of experience building responsive web applications in React and TypeScript.

SKILLS
JavaScript, TypeScript, React, HTML, CSS, Redux, REST APIs, Jest, Git, Figma basics

EXPERIENCE
Frontend Developer, Brightwave Labs (2024 to present)
- Build reusable React components and a design system shared by three product teams.
- Consume REST APIs, handle loading and error states, and manage state with Redux.
- Improve page load time by code splitting and lazy loading images.
- Write unit tests with Jest and React Testing Library.

Web Developer Intern, Kestrel Systems (2023)
- Converted Figma designs to HTML and CSS pages.

EDUCATION
B.Tech, Computer Science, Sample University, 2024
""",
    "resume_05": """
VIKRAM SINGH
vikram.singh@example.com | +91 90000 00005

SUMMARY
DevOps engineer with 4 years of experience automating builds, deployments and infrastructure for cloud-hosted services.

SKILLS
Linux, Docker, Kubernetes, AWS, Terraform, Jenkins, GitHub Actions, Python, Bash, Prometheus, Grafana, Git

EXPERIENCE
DevOps Engineer, Orchid Software (2022 to present)
- Run production workloads on Kubernetes (EKS) and manage the AWS infrastructure with Terraform.
- Build CI/CD pipelines in Jenkins and GitHub Actions that cut release time from hours to minutes.
- Set up Prometheus and Grafana monitoring and on-call alerting.
- Write Python and Bash scripts to automate backups and log clean-up.

System Administrator, Lumen Retail (2020 to 2022)
- Administered Linux servers and handled patching and incident response.

EDUCATION
B.Sc Computer Science, Sample University, 2020

CERTIFICATIONS
AWS Certified Solutions Architect Associate
""",
    "resume_06": """
DIVYA NAIR
divya.nair@example.com | +91 90000 00006

SUMMARY
Enthusiastic fresher seeking a challenging role in the IT industry where I can utilise my skills.

SKILLS
Python, Java, C++, JavaScript, SQL, Machine Learning, Deep Learning, NLP, Computer Vision, TensorFlow, PyTorch, Keras, scikit-learn, pandas, NumPy, Hugging Face Transformers, Docker, Kubernetes, AWS, Azure, REST API, Git, Linux, Tableau, Power BI, Excel

EXPERIENCE
Sales Associate (part time), Lumen Retail (2024)
- Helped customers at the billing counter and managed daily cash reports.

PROJECTS
- Followed an online tutorial to build a movie recommendation notebook.

EDUCATION
B.Com, Sample University, 2025

CERTIFICATIONS
Completed an online Python course
""",
    "resume_07": """
ARJUN PATEL
arjun.patel@example.com | +91 90000 00007

SUMMARY
Recent graduate who enjoys teaching computers to find patterns in data.

PROJECTS
- Taught a computer to recognise handwritten digits: built a small neural network, trained it on 60,000 images, and tuned it until about 97 percent of the test images were classified correctly.
- Predicted house prices from listings data: cleaned the data, tried several statistical models, and kept the one that made the smallest errors on unseen examples.
- Grouped news articles by topic without using any labels, then checked the groups by hand.

EXPERIENCE
Research Assistant, Example Institute of Technology (6 months)
- Collected and labelled data for a study on speech recognition and wrote scripts to check label quality.

EDUCATION
B.Tech, Electronics and Communication, Example Institute of Technology, 2025
""",
    "resume_08": """
MEERA JOSHI
meera.joshi@example.com | +91 90000 00008

SUMMARY
Marketing executive with 4 years of experience in content, social media and email campaigns for consumer brands.

SKILLS
Content strategy, SEO, social media marketing, email campaigns, Google Analytics, Canva, Excel, copywriting

EXPERIENCE
Marketing Executive, Lumen Retail (2022 to present)
- Plan a monthly content calendar and write blog posts that grew organic traffic by 40 percent.
- Run Instagram and email campaigns and report open rates and conversions in Google Analytics.
- Coordinate with designers on banners and product pages.
- Track the campaign budget in Excel.

Content Writer, Brightwave Labs (2021 to 2022)
- Wrote website copy and case studies.

EDUCATION
BBA, Marketing, Sample University, 2021
""",
    "resume_09": """
FARHAN QURESHI
farhan.qureshi@example.com | +91 90000 00009

SUMMARY
Senior data scientist with 6 years of experience building and shipping natural language processing systems. Led a team of four, owned models from research through production, and mentored junior engineers.

SKILLS
Python, PyTorch, Hugging Face Transformers, NLP, deep learning, machine learning, scikit-learn, pandas, NumPy, SQL, Docker, AWS, REST API, MLflow, Git, Spark

EXPERIENCE
Senior Data Scientist, Orchid Software (2022 to present)
- Lead a team of four building a document-understanding platform that extracts entities and answers questions from contracts.
- Fine-tuned BERT and T5 models with Hugging Face Transformers and PyTorch; improved F1 on the entity task from 0.78 to 0.89 over the previous system.
- Designed offline and online evaluation, including error analysis, annotation guidelines and A/B experiments.
- Packaged models as Docker images and served them through REST APIs on AWS; reduced median latency by 35 percent with model distillation.
- Built feature and training pipelines with pandas, SQL and Spark, and tracked experiments in MLflow.
- Mentor three junior engineers and run weekly paper-reading sessions.

Data Scientist, Brightwave Labs (2019 to 2022)
- Built a ticket classification and routing model for customer support using scikit-learn and later a transformer model, which removed most manual triage.
- Created a search relevance model with sentence embeddings and a vector index.
- Wrote the team's evaluation toolkit for precision, recall, F1 and calibration curves.
- Worked with product managers to define success metrics and presented results to leadership.

Data Analyst, Kestrel Systems (2018 to 2019)
- Produced SQL reports and dashboards and automated recurring analysis in Python.

PUBLICATIONS
- Co-author, workshop paper on weak supervision for contract entity extraction (2023).

EDUCATION
M.Tech, Computer Science, Example Institute of Technology, 2018
B.Tech, Computer Science, Sample University, 2016

CERTIFICATIONS
AWS Certified Machine Learning Specialty
""",
    "resume_10": """
ISHAAN VERMA
ishaan.verma@example.com | +91 90000 00010

SUMMARY
Self-taught developer with a year of freelance experience building small websites and web apps for local businesses.

SKILLS
JavaScript, React, Node.js, HTML, CSS, Python (basic), SQL (basic), Git

PROJECTS
- Restaurant ordering site: React front end with a Node.js and Express back end; menu data stored in a SQL database.
- Personal budget tracker: a single-page app using React hooks and local storage.

EXPERIENCE
Freelance Web Developer (2024 to present)
- Delivered four websites for local shops, including a booking form and a contact page.
- Fixed browser compatibility bugs and improved mobile layouts.

EDUCATION
B.A. Economics, Sample University, 2023
""",
}

# ----------------------------------------------------------- job descriptions
JDS = {
    "jd_01": """
Junior Machine Learning Engineer

About the role
We are a small product team building prediction features for an online marketplace. You will join the ML team to help train, evaluate and ship models.

Responsibilities
- Prepare and clean datasets and build features for model training.
- Train and evaluate classification and regression models.
- Work with senior engineers to put models into production.

Requirements
- Strong Python programming experience.
- Hands-on experience with pandas, NumPy and scikit-learn.
- Understanding of model evaluation: precision, recall, F1 and cross-validation.
- Knowledge of deep learning with TensorFlow or PyTorch.
- Familiarity with Docker and REST APIs.

Nice to have
- Experience with NLP or Hugging Face Transformers.
- Basic knowledge of AWS.

Benefits
Flexible hours, a learning budget and health insurance.
""",
    "jd_02": """
Data Analyst

About the role
Our operations team needs an analyst who can turn raw data into clear reports that managers use every week.

Responsibilities
- Write SQL queries to pull and combine data from our warehouse.
- Build and maintain dashboards and weekly reports.
- Analyse trends, run simple experiments and present findings to non-technical colleagues.
- Clean and validate data and document how our metrics are defined.

Requirements
- Strong SQL skills.
- Experience with Excel and a BI tool such as Power BI or Tableau.
- Working knowledge of Python and pandas for analysis.
- Good grasp of statistics and data visualization.
- Clear written and spoken communication.

Nice to have
- Experience with A/B testing.
- Familiarity with cloud data warehouses.

Benefits
Hybrid work, paid certifications and health insurance.
""",
    "jd_03": """
Backend Java Developer

About the role
Join the payments platform team to build and run the services that process our customers' orders.

Responsibilities
- Design, build and maintain Java microservices and REST APIs.
- Write clean, tested code and take part in code reviews.
- Model data and tune queries in a relational database.
- Help deploy and monitor services in the cloud.

Requirements
- 2+ years of professional Java experience.
- Hands-on experience with Spring Boot and REST APIs.
- Solid SQL skills, preferably PostgreSQL.
- Experience with Git and automated testing.
- Understanding of microservices.

Nice to have
- Experience with Docker, Kafka and AWS.

Benefits
Competitive salary, stock options and health insurance.
""",
    "jd_04": """
Frontend Developer

About the role
We are looking for a frontend developer to build the customer-facing web app for our booking platform.

Responsibilities
- Build responsive user interfaces with React.
- Integrate with REST APIs and manage application state.
- Turn designs into accessible, reusable components.
- Write tests and fix bugs found in code review.

Requirements
- Solid experience with JavaScript and TypeScript.
- Hands-on experience with React.
- Good knowledge of HTML and CSS.
- Experience consuming REST APIs.
- Comfortable using Git.

Nice to have
- Experience writing tests with Jest.
- An eye for design and accessibility.

Benefits
Remote-friendly, flexible hours and a home-office allowance.
""",
    "jd_05": """
DevOps Engineer

About the role
Help our engineering teams ship faster and more reliably by owning our cloud infrastructure and delivery pipelines.

Responsibilities
- Build and maintain CI/CD pipelines.
- Manage cloud infrastructure as code.
- Run and scale containerised services in production.
- Set up monitoring, logging and alerting, and take part in on-call.

Requirements
- Experience with Linux administration.
- Strong knowledge of Docker and Kubernetes.
- Hands-on experience with AWS.
- Experience with Terraform or similar infrastructure-as-code tools.
- Scripting in Python or Bash.

Nice to have
- Experience with Jenkins or GitHub Actions.
- Familiarity with Prometheus and Grafana.

Benefits
On-call allowance, learning budget and health insurance.
""",
    "jd_06": """
Senior NLP Engineer

About the role
We are building language-understanding features for our legal technology product and need an experienced NLP engineer to lead the modelling work.

Responsibilities
- Fine-tune and evaluate transformer models for classification, extraction and search.
- Design experiments, annotation guidelines and error analysis.
- Deploy models as services and monitor their quality in production.
- Mentor junior team members.

Requirements
- 4+ years of experience in machine learning or NLP.
- Strong Python and PyTorch skills.
- Hands-on experience with Hugging Face Transformers.
- Experience deploying models with Docker and REST APIs.
- Strong SQL and data handling with pandas.

Nice to have
- Experience with AWS and MLflow.
- Experience with vector search.

Benefits
Conference budget, flexible hours and health insurance.
""",
}

# (resume, jd) in a mixed order, so neighbouring rows are not alike.
PAIRS = [
    ("resume_06", "jd_01"), ("resume_03", "jd_03"), ("resume_08", "jd_01"),
    ("resume_01", "jd_01"), ("resume_10", "jd_04"), ("resume_09", "jd_01"),
    ("resume_02", "jd_06"), ("resume_05", "jd_05"), ("resume_07", "jd_01"),
    ("resume_04", "jd_02"), ("resume_02", "jd_02"), ("resume_01", "jd_06"),
    ("resume_06", "jd_06"), ("resume_10", "jd_03"), ("resume_09", "jd_06"),
    ("resume_03", "jd_01"), ("resume_04", "jd_04"), ("resume_02", "jd_01"),
    ("resume_05", "jd_03"), ("resume_08", "jd_02"), ("resume_01", "jd_02"),
    ("resume_07", "jd_06"), ("resume_03", "jd_05"), ("resume_10", "jd_02"),
]

CSV_COLUMNS = [
    "pair_id", "resume_file", "jd_file",
    "human_score_0_to_100", "human_verdict", "notes",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite data/labels/pairs.csv even if it exists (erases ratings!)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = PROJECT_ROOT
    pairs_csv = root / "data" / "labels" / "pairs.csv"

    if pairs_csv.exists() and not args.force:
        print(f"STOPPED: {pairs_csv} already exists and may contain your ratings.")
        print("Nothing was written. Use --force only if you really want to replace it.")
        return 1

    resume_dir = root / "data" / "synthetic" / "resumes"
    jd_dir = root / "data" / "synthetic" / "jds"
    for folder in (resume_dir, jd_dir, pairs_csv.parent):
        folder.mkdir(parents=True, exist_ok=True)

    for name, text in RESUMES.items():
        (resume_dir / f"{name}.txt").write_text(text.strip() + "\n", encoding="utf-8", newline="\n")
    for name, text in JDS.items():
        (jd_dir / f"{name}.txt").write_text(text.strip() + "\n", encoding="utf-8", newline="\n")

    with pairs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for number, (resume, jd) in enumerate(PAIRS, start=1):
            writer.writerow([
                number,
                f"data/synthetic/resumes/{resume}.txt",
                f"data/synthetic/jds/{jd}.txt",
                "", "", "",
            ])

    print(f"Wrote {len(RESUMES)} resumes to {resume_dir}")
    print(f"Wrote {len(JDS)} job descriptions to {jd_dir}")
    print(f"Wrote {len(PAIRS)} pairs to {pairs_csv} (ratings are EMPTY: fill them yourself)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
