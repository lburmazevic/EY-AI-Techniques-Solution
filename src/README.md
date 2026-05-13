# How-To Run Guide

This project runs a funding-call matching pipeline with retrieval, explainability, and optional LLM summaries.

## File map and run order

| Run order | File | What it does |
| --- | --- | --- |
| 1 | `ingest.ipynb` | Builds/updates the Chroma vector database from funding call PDFs. |
| 2 | `query.ipynb` | Runs retrieval, ranking, XAI, and Phase 3 summary experiments. |
| 3 | `app.py` | Launches the Streamlit interface for interactive end-to-end use. |
| support | `pipeline_core_methods.py` | Shared core methods for cleaning, chunking, scoring, and XAI logic. |
| support | `llm_explainer.py` | Ollama summary layer with prompt, parsing, and citation handling. |

## 1) Create and activate virtual environment

### Linux / macOS / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Start Ollama (required for LLM summaries)

Install Ollama from <https://ollama.com>, then run:

```bash
ollama serve
```

In another terminal, pull the model:

```bash
ollama pull qwen3:0.6b
```

## 4) Run the Streamlit app

From project root:

```bash
streamlit run app.py
```

Open the URL shown in terminal, usually:

```text
http://localhost:8501
```

## Notes

- Keep paths unchanged: `docs/`, `data/chroma/`, and `outputs/` are used directly by code.
- If `data/chroma/` is missing, run ingestion first.
- If LLM output is unavailable, verify `ollama serve` is running and `qwen3:0.6b` is installed.
