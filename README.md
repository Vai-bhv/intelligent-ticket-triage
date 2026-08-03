# Intelligent Ticket Triage & SLA Risk Prediction

> An end-to-end machine learning application for predicting SLA breach risk in operational support tickets using text, workload, team, and ticket characteristics.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-success)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Docker](https://img.shields.io/badge/Container-Docker-blue)

---

## Overview

Support teams often discover high-risk tickets only after delays have already occurred.

This project predicts whether a newly created operational ticket is likely to breach its SLA and converts the model output into practical triage recommendations.

The application provides:

- SLA breach probability
- Business-facing risk level
- Suggested ticket priority
- Operational recommendations
- Tuned decision threshold
- Interactive Streamlit interface
- Dockerized deployment

The project uses synthetic, non-confidential enterprise-style ticket data.

---

## Business Problem

A standard ticket queue treats cases mainly according to manually assigned priority.

However, SLA risk also depends on factors such as:

- Ticket description
- Support category
- Current team backlog
- Team experience
- Previous reassignments
- Weekend creation
- SLA target
- Region and support channel

The goal is to identify likely SLA breaches early enough for teams to intervene.

---

## Machine Learning Workflow

```text
Synthetic Ticket Data
        |
        v
Feature Engineering
        |
        v
Text Processing with TF-IDF
        |
        v
Numerical and Categorical Preprocessing
        |
        v
Model Training
   ┌──────────────────────┐
   │ Logistic Regression  │
   │ XGBoost Classifier   │
   └──────────────────────┘
        |
        v
Model Selection using PR-AUC
        |
        v
Threshold Optimization using F2
        |
        v
Streamlit Risk Assessment App
```

---

## Dataset

The synthetic dataset contains 12,000 enterprise-style support tickets covering the period from 2023 to 2025.

Example ticket categories:

- Access Management
- Application Support
- Data Quality
- Infrastructure
- Workforce Operations

Main features include:

- Region
- Category
- Priority
- Assigned team
- Channel
- Ticket description
- Description length
- Backlog at creation
- Team experience
- Previous reassignments
- Weekend indicator
- SLA target hours
- Resolution hours
- SLA breach outcome

The generated SLA breach rate is approximately 12%.

---

## Feature Engineering

### Text features

Ticket descriptions are transformed using TF-IDF with:

- Unigrams and bigrams
- English stop-word removal
- Sublinear term frequency
- Maximum vocabulary size of 3,000 features

### Numerical features

- Description length
- Backlog at creation
- Team experience
- Previous reassignments
- Weekend creation
- SLA target hours
- Day of week
- Month
- Quarter
- Creation hour
- Month-end indicator

### Categorical features

- Region
- Category
- Priority
- Assigned team
- Channel
- Backlog bucket

---

## Models

Two classification pipelines were compared:

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline |
| XGBoost Classifier | Final nonlinear model |

The data was split chronologically:

- 80% older tickets for training
- 20% newer tickets for testing

This avoids evaluating the model on historical records after learning from newer tickets.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.789 | 0.340 | 0.821 | 0.481 | 0.876 | 0.500 |
| **XGBoost Classifier** | **0.820** | **0.374** | 0.758 | **0.501** | 0.869 | **0.507** |

XGBoost was selected using PR-AUC because SLA breaches represent only around 12% of the dataset.

Accuracy alone would be misleading for this imbalanced classification problem.

---

## Decision Threshold Optimization

The default classification threshold of 0.50 was compared with alternative thresholds.

The final threshold was selected using F2 score, which gives more importance to recall than precision.

### Final operating threshold

```text
0.45
```

### Performance at the selected threshold

- Precision: 0.344
- Recall: 0.800
- F1: 0.482
- F2: 0.633
- True positives: 228
- False negatives: 57

Compared with the default threshold, the selected threshold identifies 12 additional SLA breaches and reduces false negatives from 69 to 57.

This reflects the business assumption that missing a likely breach is more costly than reviewing an additional alert.

---

## Streamlit Application

The application allows users to enter ticket characteristics and receive an immediate SLA risk assessment.

The interface includes:

- Ticket description
- Region
- Category
- Current priority
- Assigned team
- Support channel
- SLA target
- Team backlog
- Team experience
- Previous reassignments
- Creation timing

The output includes:

- SLA breach probability
- Risk level
- Decision threshold
- Suggested operational priority
- Recommended action

---

## Example Output

A high-risk ticket may generate recommendations such as:

- Escalate the ticket for proactive review
- Move the ticket to a lower-load team
- Assign clear ownership
- Confirm weekend coverage
- Review whether the current priority should be increased

---

## Project Structure

```text
intelligent-ticket-triage/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_sla_risk_model.pkl
│
├── reports/
│   ├── model_metrics.json
│   ├── test_predictions.csv
│   ├── threshold_config.json
│   └── figures/
│
├── src/
│   ├── generate_data.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Vai-bhv/intelligent-ticket-triage.git
cd intelligent-ticket-triage
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Generate the synthetic ticket dataset:

```bash
python src/generate_data.py
```

Create engineered features:

```bash
python src/features.py
```

Train and compare the models:

```bash
python src/train.py
```

Optimize the decision threshold:

```bash
python src/evaluate.py
```

Launch the Streamlit application:

```bash
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

---

## Docker

Build the Docker image:

```bash
docker build -t intelligent-ticket-triage:1.0 .
```

Run the container:

```bash
docker run --rm \
  --name intelligent-ticket-triage \
  -p 8501:8501 \
  intelligent-ticket-triage:1.0
```

Open:

```text
http://localhost:8501
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TF-IDF
- Streamlit
- Docker
- Git

---

## Business Value

The project demonstrates how machine learning can support operational ticket management by:

- Identifying high-risk tickets earlier
- Prioritizing proactive intervention
- Reducing missed SLA deadlines
- Supporting workload balancing
- Improving ticket ownership
- Converting model probabilities into practical recommendations

---

## Limitations

- The dataset is synthetic and does not represent real production performance.
- Operational recommendations are rule-based.
- The model does not currently include live queue updates.
- The ticket text vocabulary is limited by the synthetic data generator.
- Model monitoring and drift detection are not implemented.

---

## Future Improvements

Potential extensions include:

- SHAP-based local explanations
- Real service-management data
- Cost-sensitive threshold optimization
- Automated retraining
- Model drift monitoring
- API integration
- Live ticket-system integration
- Resolution-time regression

---

## Author

**Vaibhav Gupta**

Data Analyst | Aspiring Data Scientist