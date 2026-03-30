# Explainable Loan Default Propensity Model  
## Project Operating Plan (XAI Decision-Support System)

## 1) Project Title and Goal

**Project Title:** Explainable Loan Default Propensity Model for Loan Approval Decision Support

This project develops an Explainable AI (XAI) system that estimates a borrower's probability of default and translates that prediction into a transparent loan approval recommendation. The core business question is: **"Should a loan be approved?"** The system is intentionally designed as a decision-support tool for human decision-makers, combining predictive analytics with interpretable reasoning so recommendations can be understood, challenged, and justified.

---

## 2) Delivery Phasing

### Phase A: Intermediate Phase (Model Foundation + Early Explainability)

**What gets done**
- Define problem scope, assumptions, and decision policy draft.
- Acquire and profile dataset; complete initial cleaning and preprocessing pipeline.
- Train baseline model (Logistic Regression) and one stronger candidate model.
- Produce first-pass explainability outputs (global + local examples).
- Draft initial fairness and risk review.

**Artifacts expected**
- Data dictionary and feature inventory.
- Data quality report (missingness, outliers, class imbalance, leakage checks).
- Baseline model notebook with metrics.
- Initial explainability visuals (feature importance, coefficient interpretation/SHAP samples).
- Interim slide deck (progress, issues, next decisions).

**What this phase proves**
- The project is technically feasible.
- The selected data can support default-risk prediction.
- Explanations can be generated and connected to lending logic.

### Phase B: Final Phase (Integrated XAI Decision System)

**What gets done**
- Finalize best-performing and explainable model.
- Implement decision engine (approve/review/reject logic based on probability thresholds).
- Finalize explainability layer for both global and case-level decisions.
- Perform fairness/accountability checks and articulate governance limits.
- Consolidate results into final report, presentation, and optional demo.

**Artifacts expected**
- Reproducible pipeline (cleaning -> training -> evaluation -> explanation).
- Final model outputs and calibration/threshold analysis.
- Decision logic module with documented policy rationale.
- Final XAI dashboard/notebook views for individual cases and portfolio-level insights.
- Technical report + presentation slides + optional lightweight demo.

**What this phase proves**
- The system works end-to-end as an integrated XAI decision-support solution.
- Recommendations are both quantitatively grounded and transparently justified.
- The submission is academically rigorous and practically relevant.

---

## 3) Objective and Business Framing

Loan default prediction matters because lending decisions directly affect financial risk, portfolio stability, and credit access. Pure prediction alone is insufficient in high-stakes domains like credit: the system must also support accountable decision-making.

This is a **decision-support** problem because:
- A probability score does not automatically define a policy action.
- Lending decisions require explicit threshold logic and human oversight.
- Different error types (wrong approvals vs wrong rejections) carry different business and social costs.

Explainability is critical in lending due to:
- **Fairness:** avoid opaque patterns that may produce discriminatory outcomes.
- **Trust:** enable credit officers and stakeholders to understand model behavior.
- **Accountability:** provide reasons for decisions and support internal auditability.
- **Regulatory relevance:** financial decisions often require reason-giving and defensibility.

---

## 4) Naming and System Framing

To avoid confusion, the project should use three explicit components:

1. **Loan Default Propensity Model (Predictive Core)**  
   - Estimates `P(default)` for each borrower.  
   - Output: probability score (e.g., 0.72).

2. **Loan Approval Decision Engine (Policy Layer)**  
   - Converts probability into recommendation via business thresholds.  
   - Output: `Approve`, `Review`, or `Reject`.

3. **Explainability Layer (Transparency Layer)**  
   - Explains why a score and recommendation were produced.  
   - Output: global drivers + local case-level reasons.

**Why this is not "just a model":**  
A standalone model predicts risk; an XAI decision-support system predicts risk, applies explicit decision policy, and provides transparent justification suitable for real decision contexts.

---

## 5) Challenge Requirements / Success Criteria

A successful submission should satisfy both **academic** and **technical** standards.

### Academic success criteria
- Clear problem framing tied to XAI, not only predictive performance.
- Explicit methodology for explainability and fairness-aware interpretation.
- Well-argued trade-offs (performance vs interpretability vs operational simplicity).
- Reproducible analysis and defensible conclusions.

### Technical success criteria
- Clean, documented data pipeline.
- Baseline and advanced model comparison.
- Robust evaluation with threshold-based business interpretation.
- Working decision recommendation logic.
- Credible explainability outputs (global + local).
- Transparent limitations and risk discussion.

---

## 6) Simplified Operating Design

The architecture should remain intentionally lean and student-manageable:

- **One predictive model** selected as final (with one baseline comparison).
- **One decision rule layer** with clear probability thresholds.
- **One explainability layer** using a small, consistent set of XAI methods.

### Proposed flow
1. Input borrower features.
2. Compute default probability.
3. Apply threshold policy to generate recommendation.
4. Generate explanation for score and recommendation.

### Anti-overengineering principles
- No complex microservice architecture required.
- No unnecessary model ensembles unless clearly justified.
- No excessive tooling stack; prioritize reproducibility and clarity.

---

## 7) Example System Logic

### Example input features
- Income
- Loan amount
- Debt-to-income ratio
- Credit history quality
- Employment length
- Home ownership status
- Loan purpose

### Example model output
- **Default probability:** `0.72`

### Example decision output (illustrative thresholds)
- `Approve` if probability < 0.30  
- `Review` if 0.30 <= probability < 0.60  
- `Reject` if probability >= 0.60

Given `0.72` -> **Recommendation: Reject**

### Example explanation output
- Main reasons:
  - High debt-to-income ratio
  - Poor/limited credit history
  - Large requested loan relative to income

This demonstrates integrated behavior: **prediction -> policy decision -> explanation**.

---

## 8) Dataset Strategy

### Recommended approach
Use a **public lending/credit dataset** with borrower attributes and repayment outcome labels (recommended for transparency and reproducibility in academic settings).

### Suitable dataset types
- Public loan performance datasets (e.g., Lending Club-style historical loans).
- Credit risk datasets with default/non-default outcomes.

### Target variable
- Binary target: `default` (1 = default, 0 = non-default), or a proxy such as charge-off/non-payment status.

### Common available features
- Applicant income, loan amount, term, interest rate
- Debt metrics (DTI), credit history indicators
- Employment length, home ownership, loan purpose
- Delinquency/bureau-related indicators (if available)

### Important note on approval labels
If there is no real `approved/rejected` variable, approval recommendation should be **derived** from predicted default risk via decision thresholds defined in the project policy layer.

---

## 9) Candidate Feature Inventory

| Feature Family | Example Variables | Notes |
|---|---|---|
| Financial Profile | annual income, monthly obligations, DTI | Core affordability indicators |
| Loan Characteristics | loan amount, term, interest rate, purpose | Captures exposure and product structure |
| Credit History | credit score band, delinquencies, revolving utilization, prior defaults | Strong predictors of repayment behavior |
| Applicant Stability | employment length, employment status, residential stability | Proxy signals for income continuity |
| Demographic / Contextual (with caution) | age band, region, macro context | Use carefully; justify inclusion/exclusion for fairness |
| Engineered Ratios | loan-to-income, installment-to-income, utilization buckets | Improves signal if leakage-free |

**Sensitive variables** (e.g., gender, ethnicity, protected proxies) should be excluded or tightly controlled depending on fairness framing and course expectations.

---

## 10) Data Quality and Preprocessing Plan

### Data quality checks
- Profile missingness by variable and segment.
- Identify outliers in monetary/ratio variables.
- Validate ranges and consistency (e.g., negative income anomalies).
- Detect duplicate records and impossible combinations.

### Preprocessing decisions
- **Missing values:** median/most-frequent imputation; optional missingness flags.
- **Outliers:** winsorization/capping or robust transformations where justified.
- **Categoricals:** one-hot encoding (or target-safe alternatives if needed).
- **Numericals:** scaling for linear models; optional for tree-based models.
- **Splits:** train/validation/test with stratification on default target.
- **Leakage prevention:** exclude post-decision/post-outcome variables.
- **Class imbalance:** class weights, threshold tuning, and/or resampling (applied on train only).

---

## 11) Modeling Strategy

### Recommended modeling path
1. **Baseline:** Logistic Regression  
   - Transparent coefficients; strong benchmark for interpretability.
2. **Stronger candidate:** Random Forest / XGBoost / LightGBM  
   - Better nonlinear capture, often higher predictive performance.
3. **Optional comparison:** keep 2-3 models total to avoid analysis sprawl.

### Model selection principle
The goal is **not** the most complex model; it is a model with good enough predictive performance that can be explained credibly and consistently for decision support.

### Interpretability trade-off
- Linear models: easier native interpretation, potentially lower performance.
- Tree boosting models: stronger performance, require model-agnostic/model-specific XAI tools.
- Final choice should balance performance, stability, and explainability quality.

---

## 12) Explainability Strategy (Core XAI Component)

### Explainability objectives
- Explain system behavior at portfolio level (**global explainability**).
- Explain individual borrower decisions (**local explainability**).
- Connect explanations to concrete approval/rejection recommendations.

### Recommended methods
| Scope | Method | Purpose |
|---|---|---|
| Global | Feature importance (model-specific or permutation) | Rank overall drivers of default risk |
| Global | Coefficients (for Logistic Regression) | Direction and magnitude of influence |
| Global | Partial Dependence / ICE | Show marginal effect patterns |
| Local | SHAP values (recommended) | Case-level contribution breakdown |
| Local | LIME (optional) | Alternative local surrogate explanation |

### Model-specific vs model-agnostic
- **Model-specific:** coefficients, tree-based importance (fast, direct but limited to model family).
- **Model-agnostic:** SHAP/LIME/permutation (comparable across models, often more flexible).

### Required explanation outputs in final deliverables
- Top global risk drivers with interpretation.
- At least 3-5 local case explanations (approve/review/reject examples).
- Decision rationale panel linking:
  - predicted probability,
  - threshold-based recommendation,
  - top contributing factors.

### Why this fulfills XAI goals
- Increases trust and auditability.
- Allows stakeholders to challenge and validate decisions.
- Demonstrates transparent AI usage in a high-stakes domain.

---

## 13) Fairness, Accountability, and Risk Considerations

### Key fairness concerns
- Historical bias embedded in training data.
- Proxy discrimination via non-sensitive variables correlated with protected traits.
- Unequal error rates across groups.

### Risk controls for project scope
- Document inclusion/exclusion rationale for sensitive/contextual features.
- Run basic subgroup performance checks where feasible.
- Compare explanation patterns for potential disparate impacts.
- Keep a "limitations and governance" section explicit in the report.

### Accountability position
Explainability is especially important in financial decisions because recommendations can materially affect people's access to credit. The project should show how transparent reasoning improves accountability without claiming full legal compliance certification.

---

## 14) Evaluation Plan

### Core classification metrics
- Accuracy (reported but not primary in imbalanced settings)
- Precision, Recall, F1-score
- ROC-AUC
- PR-AUC (important under class imbalance)

### Decision-oriented evaluation
- Confusion matrix at selected thresholds.
- Threshold tuning aligned to policy goals.
- Optional calibration check (Brier score, reliability curve) to ensure probabilities are meaningful.

### Business interpretation of errors
- **False Positive (predict default but borrower repays):** unnecessary rejection/opportunity loss.
- **False Negative (predict repay but borrower defaults):** credit loss/risk underestimation.
- Threshold choice should explicitly reflect the relative cost of these errors.

---

## 15) Deliverables

Minimum required outputs for final submission:

1. **Notebook(s):** EDA, preprocessing, modeling, explainability.
2. **Cleaned data pipeline:** reproducible scripts/notebooks for transformations.
3. **Trained model outputs:** saved metrics, selected model artifact, threshold rationale.
4. **Explainability visuals:** global and local interpretation figures/tables.
5. **Technical report:** methodology, results, trade-offs, fairness discussion, limitations.
6. **Presentation slides:** concise narrative for evaluation panel.
7. **Optional demo:** simple interactive inference + explanation view.

---

## 16) Suggested Folder Structure

```text
project-root/
|- data/
|  |- raw/
|  |- processed/
|  \- external/
|- notebooks/
|  |- 01_eda.ipynb
|  |- 02_preprocessing.ipynb
|  |- 03_modeling.ipynb
|  \- 04_explainability.ipynb
|- src/
|  |- data_pipeline/
|  |- modeling/
|  |- decision_engine/
|  \- explainability/
|- outputs/
|  |- metrics/
|  |- figures/
|  \- model_artifacts/
|- docs/
|  |- PROJECT_PLAN.md
|  |- technical_report.md
|  \- fairness_notes.md
\- presentation/
   \- final_slides.pptx
```

### Folder usage guidance
- `data/`: source and transformed datasets (never overwrite raw).
- `notebooks/`: analysis workflow in logical sequence.
- `src/`: reusable functions for pipeline, model, decision logic, XAI methods.
- `outputs/`: all generated evidence (figures, metrics, artifacts).
- `docs/`: planning and report documentation.
- `presentation/`: final communication material.

---

## 17) Final Recommended Project Framing

This project should be presented as an **Explainable AI decision-support system for loan approval**, not as a standalone predictive notebook. The core contribution is the integration of (1) default propensity estimation, (2) explicit approval recommendation logic, and (3) transparent, case-level and global explanations. Framed this way, the project demonstrates how AI can support **fairer, more accountable, and operationally usable** lending decisions by combining predictive performance with explainability and governance-aware reasoning.
