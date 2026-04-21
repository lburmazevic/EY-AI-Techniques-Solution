# EY Challenge 2026 - EU Funding Call Matcher Plan

**Goal:** Build an explainable AI system that matches Italian municipal planning priorities to relevant EU/national funding calls, and clearly explains why each recommendation was made.

## 1. Project Requirements and Scoring Targets

From the EY brief and course rules, the solution must deliver:

- A working prototype (POC) with usable matching flow
- Explainable recommendations (not only ranked scores)
- Technical report (`technical_report.pdf`, 5-10 pages + appendices)
- Short presentation (`presentation.pdf` or `presentation.pptx`, max 5 slides)
- Reproducible source code in `src.zip`

Evaluation drivers to optimize:

| Driver | Score | What We Must Demonstrate |
| ------ | ----- | ------------------------ |
| Innovation | 10 | Differentiating element (XAI + understandable rationale) |
| Technical Implementation | 5 | End-to-end, manageable, reproducible pipeline |
| Accountability and Design | 5 | Reliable, traceable assumptions and decision logic |
| XAI Quality | 5 | Explanations are clear and credible for non-technical users |
| Communication | 5 | Strong business value and clear storytelling |

## 2. Problem Framing and Assumptions

### Core problem
- Municipalities (especially smaller ones) struggle to manually identify funding calls aligned with local priorities.
- Current process is fragmented, slow, and hard to justify transparently.

### Product objective
- Input: municipal planning text (PTCP, strategic plan excerpts, or simulated planning document).
- Output: top-N recommended calls + plain-language explanation of thematic alignment.

### Scope assumptions (declare explicitly in report)
- Focus on Italian municipalities and open calls available during project period.
- Start with one region or region cluster for simulation realism.
- Treat blockchain/certification as out of scope unless lightweight logging is added as optional extension.
- Human-in-the-loop decision support: tool recommends, officials decide.

## 3. Data Strategy (Open Sources Only)

| Source | Data Type | Usage in Pipeline |
| ------ | --------- | ----------------- |
| EU Funding and Tenders Portal API | Open call metadata and descriptions | Primary recommendation corpus |
| OpenCoesione | Historic Italian funded projects | Optional weak supervision, taxonomy enrichment, examples |
| Erasmus+ Results Platform | Programme/project descriptions | Expand call/project semantic coverage |
| Dati.gov.it / ANAC open portals | Local context indicators | Optional municipality-level context features |
| Municipal planning documents | Policy priorities and needs | Query side for matching |

Data actions:

- Build a unified call schema: `id`, `programme`, `title`, `description`, `deadline`, `eligibility`, `url`, `language`.
- Normalize and clean multilingual text (Italian + English), remove boilerplate, keep thematic content.
- Version datasets with extraction date to ensure reproducibility.

## 4. Solution Architecture

###+ Stage A - Knowledge Base Construction
- Ingest calls and planning texts.
- Preprocess text (cleaning, section chunking, language detection, deduplication).
- Store normalized artifacts in `data/processed/`.

###+ Stage B - Semantic Matching Engine
- Encode calls and municipal plans with `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
- Compute cosine similarity between plan embeddings and call embeddings.
- Return top-N ranked matches with similarity scores.

###+ Stage C - Explainability Layer (Core Differentiator)
- Base explainability: SHAP or LIME on a surrogate classifier/ranker features.
- Contrastive explanation: "why Call A over Call B" using feature/theme contribution deltas.
- User-friendly explanation: convert contribution signals into plain-language rationale.

###+ Stage D - Optional Innovation Layer
- LLM explanation synthesis (e.g., Anthropic API) grounded only on retrieved evidence + XAI outputs.
- Mandatory guardrail: no free-form unsupported claims; include citation snippets from source text.

## 5. Modeling and Evaluation Plan

### Baseline and candidate approaches

| Component | Baseline | Candidate/Primary |
| --------- | -------- | ----------------- |
| Retrieval | TF-IDF + cosine | Multilingual sentence embeddings + cosine |
| Re-ranking (optional) | None | Lightweight classifier/reranker on top-K calls |
| XAI | Keyword overlap and score decomposition | SHAP/LIME + contrastive explanation |

### Evaluation framework

- **Offline relevance:** Precision@K, Recall@K, nDCG@K, MRR (on expert-labeled or proxy-labeled pairs).
- **Explanation quality:** human rubric (clarity, faithfulness, usefulness, actionability).
- **System quality:** latency, reproducibility, failure rate on missing/short plans.
- **Business framing:** proportion of recommendations judged "actionable" by non-technical reviewer.

### Validation design
- Use time-aware splits if calls span multiple publication periods.
- Keep held-out municipal plans (or synthetic scenarios) for final blind evaluation.
- Track ablations: no-XAI vs XAI; baseline retrieval vs embedding retrieval.

## 6. Explainability and Accountability Design

Each recommendation should produce:

1. Match score and rank
2. Top thematic drivers (keywords/themes with contribution weight)
3. Evidence snippets from municipal plan and call text
4. Contrastive note (why this call outranks an alternative)
5. Confidence/uncertainty flag (high, medium, low)

Governance checks to include:

- Data provenance log (source, date, endpoint/file).
- Prompt and model version logging for generated explanations.
- Error handling policy: fallback to transparent lexical matching if embedding/XAI component fails.
- Limitation note: recommendation support, not legal/compliance advice.

## 7. Implementation Roadmap (Now -> May 17)

| Week | Focus | Deliverables |
| ---- | ----- | ------------ |
| W1 | Scope, assumptions, data contracts | Problem statement, source inventory, schema draft |
| W2 | Data ingestion and preprocessing | ETL scripts, cleaned corpora, data dictionary |
| W3 | Baseline + embedding retriever | Ranked output v1, baseline metrics |
| W4 | XAI implementation | SHAP/LIME outputs, explanation templates |
| W5 | POC interface + evaluation | End-to-end demo flow, metrics table, error analysis |
| W6 | Final packaging | Technical report, 5-slide deck, reproducibility checks |

## 8. Deliverables Mapping to Submission Rules

### Required files
- `presentation.pdf` or `presentation.pptx` (max 5 slides)
- `src.zip` (organized Python code + `README.md` execution guide + `requirements.txt`)
- `technical_report.pdf` (5-10 pages + appendices A/B)

### `src` package minimum structure

```text
src/
|- data_ingestion.py
|- preprocessing.py
|- embedding_matcher.py
|- explainability.py
|- evaluation.py
|- app_or_demo.py
|- config.py
|- README.md
\- requirements.txt
```

### Report structure checklist
- Section 1: Introduction
- Section 2: Methods
- Section 3: Results and Discussion
- Section 4: Conclusions
- Appendix A: Code description
- Appendix B: CRediT contributions + GenAI statement

## 9. Differentiators to Maximize EY Score

- **Explainability-first UX:** each recommendation includes human-readable rationale and evidence.
- **Contrastive reasoning:** explicit "why this call, not another" explanation.
- **Non-technical understandability:** municipal-officer style output, not data-science jargon.
- **Reproducibility discipline:** deterministic pipeline, versioned sources, clear run order.
- **Business value narrative:** democratize access to EU funding for smaller municipalities.

## 10. Risk Register and Mitigation

| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| Sparse or noisy municipal text | Weak matching quality | Text chunking, query expansion, minimum text checks |
| Limited labeled relevance data | Hard to evaluate objectively | Build small expert-labeled set + proxy labels from historical projects |
| Hallucinated LLM explanation | Trust/compliance issue | Grounded prompts + citation requirement + rule-based fallback |
| Over-complex architecture | Delivery risk | Keep MVP lean: embedding retrieval + one XAI pipeline |
| Language mismatch (IT/EN) | Reduced recall | Multilingual embeddings + language-aware preprocessing |

## 11. Final Operating Principle

The project succeeds if it clearly shows an end-to-end flow:

**Municipal plan -> ranked funding calls -> transparent explanation -> actionable decision support**

with reproducible code, understandable communication, and explicit assumptions from day one.
