# Project Notes

* `data/`: contains local funding call PDFs and processed data.
* `data/chroma/`: persistent ChromaDB database (DO NOT delete unless rebuilding).
* `ingest.ipynb`: builds the vector database (chunking + embeddings + storage).
* `query.ipynb`: used to query the database and retrieve relevant chunks.

---

# RAG Workflow Rule

This project follows a strict two-phase workflow:

### 1. Ingestion (heavy, run rarely)

* Reads PDFs
* Extracts and cleans text
* Splits into chunks
* Generates embeddings
* Stores in ChromaDB

**Do NOT rerun ingestion unless:**

* new documents are added
* chunking logic changes
* embedding model changes

---

### 2. Querying (fast, run frequently)

* Takes user query
* Retrieves relevant chunks from ChromaDB
* (Optionally) sends context to LLM

---

# ChromaDB Rules

* Uses **persistent storage** (`data/chroma/`)
* Avoid running multiple ingestion processes simultaneously (can cause DB lock)
* Do not duplicate data (avoid repeated `add()` / `upsert()` without reset)

---

# Embedding Model Rule

* Default model: `paraphrase-multilingual-mpnet-base-v2`
* Model is downloaded automatically on first use
* Do not change model without rebuilding the database

---

# Python Environment Rule

* Always use the project virtual environment (`.venv`)
* Recommended setup (WSL):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

* Do not mix Windows Python and WSL Python

---

# Notebook Usage Rule

* Do NOT press "Run All" blindly
* Run ingestion cells only when needed
* Use query notebook for normal interaction

---

# Common Pitfalls

* Running ingestion multiple times → duplicated data
* Using wrong Python environment → missing packages / errors
* Mixing WSL and Windows paths → broken file access
* Expecting output before long embedding step finishes

---

# Goal

Build a reliable RAG system for funding calls that:

* retrieves relevant documents semantically
* supports multilingual queries (e.g. English + Italian)
* returns clean, usable context for downstream analysis
