# Documents

This folder contains the PDF documents used as source material by the retrieval pipeline.

The project uses two main types of documents:

1. **University strategic plans**
   These are the input documents that describe a university's priorities, goals, and strategic direction.

1. **Funding call documents**
   These are the funding opportunities that the system searches, ranks, and compares against the strategic plan.

## How this folder is used

During ingestion, the pipeline reads the PDFs in this folder, extracts their text, chunks the content, creates embeddings, and stores the processed information for retrieval.

The Streamlit app then uses this processed document base to match a selected or uploaded strategic plan against relevant funding opportunities.

## Notes

- Keep only source PDFs in this folder.
- Do not store generated outputs here.
- Generated rankings, summaries, or intermediate files should go in `outputs/`.
- Final reports or presentation PDFs should go in `reports/`.
