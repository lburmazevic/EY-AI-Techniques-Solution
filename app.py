import time
import streamlit as st

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from pipeline_core_methods import (
    chunk_text,
    clean_text,
    extract_uploaded_sp,
    rank_calls_for_sp_query,
)

st.set_page_config(
    page_title="EU Funds Matcher",
    page_icon="https://flagcdn.com/eu.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#
#
# 
#
#


N_RESULTS = 30  # Good output and not overdoing it with computing

SP_CHUNK_SIZE = 1200
SP_CHUNK_OVERLAP = 200
SP_MIN_CHUNK_CHARS = 160

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "funding_calls"
CHROMA_DIR = Path("data/chroma")

TOP_K_CALLS = 3
STRONG_HIT_THRESHOLD = 0.55
EVIDENCE_PER_CALL = 3

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve_and_rank_calls_for_uploaded_sp(sp_obj: dict, n_results: int = N_RESULTS) -> dict:
    collection = get_collection()
    return rank_calls_for_sp_query(
        collection=collection,
        sp=sp_obj,
        n_results=n_results,
        strong_hit_threshold=STRONG_HIT_THRESHOLD,
        evidence_per_call=EVIDENCE_PER_CALL,
        top_k_calls=TOP_K_CALLS,
    )


# 
#
# INTERFACE
#
#

st.title("EU Funds AI Matcher")
st.caption("Demo for strategic plan -> funding call matching")

# Session States
if "uploaded_runs" not in st.session_state:
    st.session_state.uploaded_runs = {}  

if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


st.sidebar.header("Demo Strategic Plans")
sp_options = [
    "Bocconi_Piano_Strategico2021-2025&Vision2030.pdf",
    "LUM_PIANO-STRATEGICO-DATENEO-2021-2025.pdf",
    "Naples_Federico_Piano_strategico_2021_2026.pdf",
    "Piano Strategico Luiss 2021-2024 (per sito).pdf",
    "Politechnico_di_Milano_2023-2025.pdf",
    "Sapienza_2021_2027.pdf",
]
selected_sp = st.sidebar.selectbox("Choose one SP", sp_options)
st.session_state.selected_item_id = selected_sp



st.sidebar.divider()
st.sidebar.subheader("Uploaded (This Session)")
if st.session_state.uploaded_runs:
    for name in st.session_state.uploaded_runs.keys():
        st.sidebar.write(f"- {name}")
else:
    st.sidebar.caption("No uploaded analyses yet.")

# MAIN

# Upload
st.subheader("Upload")
uploaded_file = st.file_uploader("Upload a PDF strategic plan", type=["pdf"])

# Analyze
st.subheader("Analyze")
is_ready = uploaded_file is not None

if st.button("Run Analysis", type="primary", disabled=not is_ready):
    with st.spinner("Extracting, chunking, retrieving, and ranking..."):
        sp_obj = extract_uploaded_sp(
            uploaded_file,
            size=SP_CHUNK_SIZE,
            overlap=SP_CHUNK_OVERLAP,
            min_chars=SP_MIN_CHUNK_CHARS,
        )
        phase1_result = retrieve_and_rank_calls_for_uploaded_sp(sp_obj, n_results=30)

    st.success(
        f"Done. Raw hits: {phase1_result['raw_hits']} | "
        f"Top calls: {len(phase1_result['top_calls'])}"
    )

    # Save in session for Step 4 (XAI + rendering)
    st.session_state.latest_sp_obj = sp_obj
    st.session_state.latest_phase1 = phase1_result

if not is_ready:
    st.caption("Upload a PDF first to enable analysis.")
# Results
st.subheader("Results")
st.write("Results will be shown here.")

