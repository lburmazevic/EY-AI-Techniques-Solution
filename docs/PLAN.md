# StartupEU - Explainable Funding Match Engine

**Acting role:** Documentation and Presentation Agent  
**Project option:** Option B - "StartupEU"  
**Goal:** Build an explainable AI system that matches Italian startups to the most relevant EU and PNRR funding instruments, with transparent rationale for every recommendation.

## 0. Delivery phasing

The project is delivered in two phases.

1. **Intermediate phase (current):** framing, dataset setup, baseline matcher, and initial XAI output.
   - Primary artifacts:
     - `src/` initial reproducible pipeline modules
     - `docs/PLAN.md` (this document)
     - `presentation/presentation.pptx` or `presentation/presentation.pdf` (<= 5 slides)
2. **Final phase:** validated matching engine, explainability layer, evaluation, and final communication package.
   - Primary artifacts:
     - `src.zip` (ordered, executable code)
     - `technical_report.pdf` (5-10 pages + appendices)
     - `presentation.pdf` or `presentation.pptx`

Submission deadline: **May 17, 2026, 23:59 (Rome time)**.

## 1. Objective and business framing

Local ecosystems (startups, incubators, and support teams) struggle to identify the right public funding opportunities because information is fragmented, terminology is complex, and screening is manual.

The proposed solution is an **XAI-first funding recommendation engine**:

- **Input:** startup profile (sector, maturity, revenue, innovation focus, optional sustainability tags)
- **Knowledge base:** open calls from EU and national sources
- **Output:** ranked funding opportunities with explanation of match drivers and mismatches

Example output style:

- "Startup Y matches EIC Accelerator because deep-tech focus (+0.71), early-stage profile (+0.55), and green transition alignment (+0.43), but has low revenue for some instruments (-0.22)."

This aligns with EY evaluation criteria: innovation, technical implementation, XAI quality, accountability/design, and communication.

## 2. Scope and assumptions

### 2.1 In-scope

- Build a working prototype for recommendation + explanation.
- Use only open, legally accessible data.
- Provide reproducible pipelines and documented assumptions.
- Prioritize explainability and user understandability for non-technical stakeholders.

### 2.2 Out-of-scope

- Full production deployment.
- Legal/compliance automation for grant eligibility.
- Blockchain layer (can be discussed as future extension only).

### 2.3 Simulation assumptions

- Geography: Italy-focused startup universe.
- Language handling: multilingual text normalization (Italian + English call descriptions).
- Recommendations are advisory, not deterministic eligibility decisions.
- Missing startup attributes are treated with explicit unknown flags where needed.

## 3. Data strategy (open sources only)

### 3.1 Primary data sources

1. **Italian Startup Registry** (`startup.registroimprese.it`)
   - Core startup attributes: sector, age, economic profile, innovation orientation.
2. **EU opportunities**
   - EIC Accelerator and Horizon Europe calls (public pages / APIs where available).
3. **Italian national opportunities**
   - PNRR Mission 4 (Research & Innovation) and related open program pages.

### 3.2 Optional enrichment sources

- EU Funding & Tenders opportunities API
- OpenCoesione open datasets
- Erasmus+ Project Results Platform
- DatiGov.it and ANAC open portals

### 3.3 Data governance principles

- Keep raw snapshots in `data/raw/` and cleaned versions in `data/processed/`.
- Store extraction date and source URL metadata.
- Add schema checks, missingness summaries, and basic quality diagnostics.
- Avoid scraping where official downloads/APIs exist.

## 4. Technical design

### 4.1 Pipeline overview

1. Ingest startup and funding call datasets.
2. Standardize schema and textual fields.
3. Build sentence embeddings for startups and calls.
4. Compute similarity matrix and produce top-k recommendations.
5. Add explainability layer (SHAP/LIME or feature-attribution proxy).
6. Generate stakeholder-friendly natural-language rationale.

### 4.2 Baseline and stronger models

- **Baseline:** semantic ranking via cosine similarity on multilingual sentence embeddings.
- **Candidate upgrade:** supervised reranker/classifier (e.g., gradient boosting or logistic model on engineered match features).
- **XAI methods:**
  - Global: feature importance / theme-level contribution summary
  - Local: SHAP (primary), LIME (optional backup)

### 4.3 Explainability output contract

For each recommended call, return:

- `match_score`
- `top_positive_drivers` (theme + contribution)
- `top_negative_drivers` (theme + penalty)
- `plain_language_explanation` (short text for non-technical users)
- `why_this_not_that` contrastive note for at least one near alternative

## 5. Repository and implementation plan

Planned structure:

- `data/raw/` - raw snapshots from open portals
- `data/processed/` - cleaned and normalized datasets
- `src/` - modular Python scripts (`ingest.py`, `preprocess.py`, `embed.py`, `match.py`, `explain.py`, `evaluate.py`)
- `outputs/` - recommendations, metrics, and figures
- `docs/` - plan, governance notes, report drafting material
- `presentation/` - slide deck assets

Execution order will be documented in `src/README.md` as required.

## 6. Evaluation framework

### 6.1 Technical metrics

- Ranking quality: Precision@k, Recall@k, nDCG@k
- Optional classification metrics for reranker: ROC-AUC, F1
- Robustness checks across startup segments (stage, sector, region)

### 6.2 XAI quality checks

- Explanation coherence with model signals
- Stability of top drivers under small text perturbations
- Contrastive explanation usefulness ("why A instead of B")

### 6.3 Business-value checks

- Relevance judged by expert-like rule checks or manual sample review
- Time-to-shortlist reduction vs manual baseline
- Interpretability score from user feedback rubric

## 7. Fairness, accountability, and limitations

- Sensitive/proxy features are reviewed before use in scoring.
- Recommendations are auditable through stored intermediate outputs.
- All assumptions and known limitations are documented in the report.
- Human oversight remains mandatory before final funding application decisions.

## 8. Deliverables checklist (course-compliant)

1. **Presentation** (`presentation.pdf` or `presentation.pptx`), max 5 slides.
2. **Code package** (`src.zip`) with executable, ordered, clean source files.
3. **Technical report** (`technical_report.pdf`), 5-10 pages, with:
   - Title page
   - Introduction
   - Methods
   - Results and Discussion
   - Conclusions
   - Appendix A: code description (<= 1 page)
   - Appendix B: CRediT contributions + Generative AI statement (<= 1 page)

Additional compliance notes:

- Include Python version and pinned dependencies in `requirements.txt`.
- Remove dead/commented-out legacy code.
- Ensure full reproducibility of results and figures.

## 9. Workplan milestones

1. Data acquisition and schema lock.
2. Baseline semantic matcher.
3. XAI layer and natural-language explanation prototype.
4. Evaluation and ablation checks.
5. Final packaging: report, deck, and runnable `src` bundle.

## 10. Definition of done

The task is considered done when:

1. The pipeline runs end-to-end with documented commands.
2. Top-k funding recommendations are generated for startup profiles.
3. Every recommendation includes human-readable explanation and driver breakdown.
4. Evaluation metrics and business interpretation are reported.
5. Deliverables match naming, format, and page constraints from project rules.
