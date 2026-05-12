# Project Notes

* `data/`: contains local funding call PDFs and processed data.
* `data/chroma/`: persistent ChromaDB database (DO NOT delete unless rebuilding).
* `ingest.ipynb`: builds the vector database (chunking + embeddings + storage).
* `query.ipynb`: used to query the database and retrieve relevant chunks.

---

# Python Environment Rule

Use the project's bundled virtual environment (`.venv`) for every Python operation (commands, scripts, tools, linting, and type checking).
