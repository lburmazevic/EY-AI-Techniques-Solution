# AGENTS.md

## Purpose

This document defines how AI and human contributors should collaborate on the **Explainable Loan Default Propensity Model** project.

The project is Python-first and structured around three integrated components:
- predictive model (`P(default)`)
- decision engine (`approve/review/reject`)
- explainability layer (global + local reasoning)

---

## Core Principles

1. **XAI-first delivery**
   - Do not ship prediction-only outputs.
   - Every decision recommendation must include a reasoned explanation.

2. **Student-manageable complexity**
   - Favor simple, reproducible pipelines over overengineered solutions.
   - Keep dependencies minimal and well documented.

3. **Reproducibility by default**
   - Fix random seeds where relevant.
   - Separate raw data from processed artifacts.
   - Ensure notebooks and scripts can be rerun end-to-end.

4. **Fairness and accountability awareness**
   - Treat sensitive and proxy features carefully.
   - Document assumptions, limitations, and bias risks.

---

## Recommended Agent Roles

Use one role per task whenever possible.

### 1) Data Agent
**Scope**
- Dataset acquisition, data dictionary, schema checks
- Missing values, outliers, leakage checks
- Train/validation/test split strategy

**Outputs**
- `data/processed/` datasets
- Data quality report
- Feature readiness checklist

### 2) Modeling Agent
**Scope**
- Baseline model (Logistic Regression)
- Candidate stronger model (Random Forest/XGBoost/LightGBM)
- Metrics and threshold experiments

**Outputs**
- Trained model artifact(s)
- Metrics table and confusion matrix
- Threshold recommendation notes

### 3) Decision Engine Agent
**Scope**
- Convert default probability into `Approve`, `Review`, `Reject`
- Implement and justify threshold policy
- Produce decision outputs consumable by notebooks/demo

**Outputs**
- Decision policy module
- Example decision cases with rationale

### 4) Explainability Agent
**Scope**
- Global explainability: feature importance/coefficient/PDP
- Local explainability: SHAP (primary), LIME (optional)
- Align explanations with decision outcomes

**Outputs**
- Explainability plots in `outputs/figures/`
- Case-level explanation summaries

### 5) Evaluation and Governance Agent
**Scope**
- ROC-AUC, PR-AUC, precision, recall, F1, calibration
- Business interpretation of FP/FN trade-offs
- Fairness and limitations statement

**Outputs**
- Evaluation summary section for report
- Governance/fairness notes

### 6) Documentation and Presentation Agent
**Scope**
- Consolidate technical report and slides
- Ensure story coherence: risk prediction -> decision -> explanation
- Maintain glossary and assumptions list

**Outputs**
- Final report draft
- Final presentation deck content

---

## Python Project Standards

### Environment
- Python 3.10+ preferred
- Use a virtual environment (`venv` or `conda`)
- Track dependencies in `requirements.txt`
- Keep `requirements.txt` updated every time new code introduces or changes dependencies

### Coding conventions
- Follow PEP 8 and readable function naming
- Keep functions small and testable
- Add docstrings for non-trivial modules/functions
- Avoid notebook-only logic for core pipeline steps; move reusable code to `src/`

### Suggested libraries
- Data: `pandas`, `numpy`
- Modeling: `scikit-learn`, optional `xgboost` or `lightgbm`
- Explainability: `shap`, optional `lime`
- Visualization: `matplotlib`, `seaborn`

---

## Repository Workflow

### Folder intent
- `data/`: raw and processed data
- `notebooks/`: EDA, modeling, explainability walkthroughs
- `src/`: reusable pipeline/model/decision/XAI modules
- `outputs/`: metrics, plots, model artifacts
- `docs/`: plans, reports, governance notes
- `presentation/`: slide materials

### Branch and commit guidance
- Use short-lived feature branches per task
- Keep commits focused and descriptive
- Recommended commit style:
  - `feat: add decision threshold policy module`
  - `fix: prevent target leakage in preprocessing`
  - `docs: add fairness limitations section`

---

## Definition of Done (Per Task)

A task is done only if all checks pass:

1. Code/notebook runs without manual patching.
2. Outputs are saved in the correct folder.
3. Assumptions and limitations are documented.
4. If model logic changed, evaluation metrics are updated.
5. If decision logic changed, explanation examples are updated.

---

## Minimal End-to-End Acceptance Criteria

Before final submission, the integrated system must demonstrate:

1. Probability prediction for loan default.
2. Threshold-based recommendation (`Approve/Review/Reject`).
3. Clear explanation for each sample decision.
4. Performance metrics with business interpretation.
5. Fairness/accountability discussion proportional to project scope.

---

## Collaboration Protocol for AI-Assisted Work

- Always state which role/agent is currently acting.
- Keep prompts task-specific (one output objective at a time).
- Require each agent output to include:
  - files changed
  - assumptions made
  - quick validation steps
- Escalate to the team when trade-offs affect fairness, threshold policy, or model interpretability.

## Notebook Writing Style (Project Rule)

- Write notebook text from a student perspective.
- Keep markdown concise and practical.
- Avoid first-person references to AI/model/assistant authorship.
- Avoid numbered walkthroughs unless explicitly requested.
- Prefer short section headers with brief context paragraphs.

---

## Non-Goals

- Building a production banking platform
- Achieving perfect fairness guarantees
- Optimizing solely for leaderboard-style accuracy

The goal is a clear, credible, explainable decision-support prototype suitable for academic evaluation.
