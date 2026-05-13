# Project Notes

* `app.py`: Streamlit app entry point for the interactive funding-match workflow.
* `pipeline_core_methods.py`: shared core logic for PDF text cleaning, chunking, retrieval scoring, and XAI helpers.
* `llm_explainer.py`: Ollama-based summarization layer that turns retrieved evidence into structured stakeholder-facing output.
* `docs/fundingcalls/`: source funding call PDFs used to build the retrieval corpus.
* `docs/strategicplans/`: strategic plan PDFs used as query/demo inputs.
* `data/chroma/`: persistent ChromaDB storage for the `funding_calls` collection. Do not delete unless intentionally rebuilding the vector database.
* `ingest.ipynb`: one-time or occasional ingestion notebook that reads funding call PDFs, chunks them, embeds them, and stores them in ChromaDB.
* `query.ipynb`: main experimentation notebook for querying the vector database with strategic plans and inspecting ranked matches.
* `outputs/app_results.json`: saved app/demo retrieval results.
* `outputs/phase3/`: saved structured LLM/explanation outputs for downstream review or demos.
---

**Current runtime configuration**

* Embedding model: `Qwen/Qwen3-Embedding-0.6B`
* Chroma collection name: `funding_calls`
* Local LLM for summaries/explanations: `qwen3:0.6b` via Ollama

---
# Python Environment Rule

Use the project's bundled virtual environment (`.venv`) for every Python operation (commands, scripts, tools, linting, and type checking).

---

# Python & Markdown Change Rule

After every Python-related change (including `.py` and `.ipynb` files), run:

* `poe check-py`

After every documentation-related change (including `.md` files), run:

* `poe check-md`

Before committing, run the full formatting/check workflow:

* `poe fmt`
* `poe check`
