# Minimal Run Guide

This project runs a funding-call matching pipeline with retrieval, explainability, and optional LLM summaries.

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

## 5) Optional notebooks (rebuild / experiments)

- `ingest.ipynb` → builds/updates the Chroma database from `docs/fundingcalls/`
- `query.ipynb` → runs retrieval, XAI, and Phase 3 summary generation

## Notes

- Keep paths unchanged: `docs/`, `data/chroma/`, and `outputs/` are used directly by code.
- If `data/chroma/` is missing, run ingestion first.
- If LLM output is unavailable, verify `ollama serve` is running and `qwen3:0.6b` is installed.