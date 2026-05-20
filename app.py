"""
app.py

What this file contains:
- Streamlit user interface for the end-to-end funding-match workflow.
- Session-state logic for demo strategic plans and uploaded strategic plans.
- Orchestration of Phase 1 retrieval/ranking, Phase 2 explainability attachment, and Phase 3 LLM briefing generation.
- Rendering of scores, matched themes, key gaps, supporting evidence, and stakeholder-facing summaries.

Role in the whole project:
- This is the interactive execution layer used by end users.
- It connects the backend analytical modules (pipeline_core_methods.py and llm_explainer.py) to a reproducible UI workflow.
- It is the final delivery surface for exploration, interpretation, and communication of results.
"""

import json
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions

from src.llm_explainer import generate_summary
from src.pipeline_core_methods import build_call_xai, extract_uploaded_sp, rank_calls_for_sp_query

st.set_page_config(
    page_title="EU Funds Matcher",
    page_icon="https://flagcdn.com/eu.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#
#
# METHODS
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


LLM_MODEL_NAME = "qwen3:0.6b"
PHASE3_TEMPERATURE = 0.1
PHASE3_MIN_CITATIONS_PER_CALL = 2
PHASE3_LANGUAGE = "English"

XAI_TOP_THEMES = 4
XAI_EVIDENCE_PER_CALL = 3
XAI_MAX_GAPS = 4
GAP_MIN_CALL_SIGNAL = 0.2
GAP_MAX_SP_SIGNAL = 0.1

APP_RESULTS_PATH = Path("outputs/app_results.json")
PHASE3_STRUCTURED_PATH = Path("outputs/phase3/phase3_structured_results.json")

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve_and_rank_calls_for_uploaded_sp(
    sp_obj: dict,
    n_results: int = N_RESULTS,
    strong_hit_threshold: float = STRONG_HIT_THRESHOLD,
    evidence_per_call: int = EVIDENCE_PER_CALL,
    top_k_calls: int = TOP_K_CALLS,
    progress_callback=None,
) -> dict:
    collection = get_collection()
    return rank_calls_for_sp_query(
        collection=collection,
        sp=sp_obj,
        n_results=n_results,
        strong_hit_threshold=strong_hit_threshold,
        evidence_per_call=evidence_per_call,
        top_k_calls=top_k_calls,
        progress_callback=progress_callback,
    )


@st.cache_data
def load_demo_runs() -> dict:
    demo_runs = {}

    if APP_RESULTS_PATH.exists():
        with APP_RESULTS_PATH.open("r", encoding="utf-8") as fh:
            app_results = json.load(fh)

        for value in app_results.values():
            sp_file = value.get("sp_file")
            if not sp_file:
                continue
            demo_runs[sp_file] = {
                "phase2": value,
                "phase3": None,
                "source": "demo",
            }

    if PHASE3_STRUCTURED_PATH.exists():
        with PHASE3_STRUCTURED_PATH.open("r", encoding="utf-8") as fh:
            phase3_structured = json.load(fh)

        for row in phase3_structured:
            sp_file = row.get("sp_file")
            if not sp_file:
                continue
            if sp_file not in demo_runs:
                demo_runs[sp_file] = {
                    "phase2": {
                        "sp_id": row.get("sp_id"),
                        "sp_file": sp_file,
                        "raw_hits": None,
                        "top_calls": [],
                    },
                    "phase3": None,
                    "source": "demo",
                }
            demo_runs[sp_file]["phase3"] = {
                "sp_id": row.get("sp_id"),
                "sp_file": sp_file,
                "model_name": "precomputed",
                "temperature": None,
                "structured": row,
                "stakeholder_text": "",
                "prompt": "",
                "raw_response_text": "",
            }

    return demo_runs



#
#
# XAI
#
#

# We copy the lexicon part from the query.ipynb


THEME_LEXICON = {
    "sustainability_climate": [
        "sustainability", "sustainable", "climate", "green", "decarbonization",
        "circular economy", "energy transition", "biodiversity", "sdg",
        "sostenibilita", "sostenibile", "clima", "transizione energetica",
        "economia circolare", "decarbonizzazione", "biodiversita"
    ],
    "ai_data_digital": [
        "artificial intelligence", "ai", "machine learning", "data", "digital",
        "digitalization", "digitization", "big data", "cloud", "cybersecurity",
        "intelligenza artificiale", "apprendimento automatico", "dati", "digitale",
        "digitalizzazione", "transizione digitale"
    ],
    "internationalization_collaboration": [
        "international", "internationalization", "cross-border", "european", "consortium",
        "partnership", "collaboration", "mobility", "network", "joint",
        "internazionale", "internazionalizzazione", "europeo", "consorzio",
        "partenariato", "collaborazione", "mobilita", "rete", "congiunto"
    ],
    "innovation_transfer_industry": [
        "innovation", "technology transfer", "valorization", "startup", "spin-off",
        "industry", "industrial", "commercialization", "patent",
        "innovazione", "trasferimento tecnologico", "valorizzazione",
        "industria", "industriale", "brevett"
    ],
    "skills_education_capacity": [
        "skills", "competence", "training", "education", "lifelong learning",
        "upskilling", "reskilling", "capacity building", "doctoral", "curriculum",
        "competenze", "formazione", "istruzione", "apprendimento permanente", "dottorato"
    ],
    "inclusion_gender_social": [
        "inclusion", "equity", "equality", "gender", "diversity", "accessibility",
        "social impact", "vulnerable", "cohesion",
        "inclusione", "equita", "uguaglianza", "genere", "diversita",
        "accessibilita", "impatto sociale", "vulnerabili", "coesione"
    ],
    "research_infrastructure_excellence": [
        "research infrastructure", "infrastructure", "laboratory", "equipment",
        "excellence", "scientific excellence", "facility", "platform",
        "infrastruttura di ricerca", "infrastruttura", "laboratorio", "attrezzature",
        "eccellenza", "eccellenza scientifica", "piattaforma"
    ],
    "governance_policy_reform": [
        "governance", "reform", "policy", "regulation", "institutional",
        "coordination", "roadmap", "implementation", "monitoring", "evaluation",
        "riforma", "politica", "regolazione", "istituzionale",
        "coordinamento", "attuazione", "monitoraggio", "valutazione"
    ],
}



def attach_xai_to_phase1(
    sp_obj: dict,
    phase1_result: dict,
    xai_evidence_per_call: int = XAI_EVIDENCE_PER_CALL,
    xai_top_themes: int = XAI_TOP_THEMES,
    gap_min_call_signal: float = GAP_MIN_CALL_SIGNAL,
    gap_max_sp_signal: float = GAP_MAX_SP_SIGNAL,
    xai_max_gaps: int = XAI_MAX_GAPS,
) -> dict:
    calls_with_xai = []
    for call in phase1_result.get("top_calls", []):
        call_copy = dict(call)
        call_copy["xai"] = build_call_xai(
            sp_text=sp_obj.get("text", ""),
            call_row=call_copy,
            theme_lexicon=THEME_LEXICON,
            xai_evidence_per_call=xai_evidence_per_call,
            xai_top_themes=xai_top_themes,
            gap_min_call_signal=gap_min_call_signal,
            gap_max_sp_signal=gap_max_sp_signal,
            xai_max_gaps=xai_max_gaps,
        )
        calls_with_xai.append(call_copy)

    return {
        "sp_id": phase1_result.get("sp_id"),
        "sp_file": phase1_result.get("sp_file"),
        "raw_hits": phase1_result.get("raw_hits"),
        "non_empty_pages": phase1_result.get("non_empty_pages"),
        "top_calls": calls_with_xai,
    }



# 
#
# INTERFACE
#
#



st.title("Translating Italian University Strategy into Funding Opportunity")
st.caption("Demo for strategic plan -> funding call matching")

st.markdown(
    """
    <style>
    [class*="st-key-remove_uploaded_"] button {
        background-color: #c1121f;
        color: #ffffff;
        border: 1px solid #c1121f;
    }
    [class*="st-key-remove_uploaded_"] button:hover {
        background-color: #9b0d18;
        color: #ffffff;
        border: 1px solid #9b0d18;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("ui_busy", False):
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            pointer-events: none;
            opacity: 0.65;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Session States
if "uploaded_runs" not in st.session_state:
    st.session_state.uploaded_runs = {}  

if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None

if "upload_mode" not in st.session_state:
    st.session_state.upload_mode = True

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "selected_demo_sp" not in st.session_state:
    st.session_state.selected_demo_sp = None

if "reset_demo_sp" not in st.session_state:
    st.session_state.reset_demo_sp = False

if "ui_busy" not in st.session_state:
    st.session_state.ui_busy = False

if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

if "pending_analysis_file" not in st.session_state:
    st.session_state.pending_analysis_file = None

if "cfg_n_results" not in st.session_state:
    st.session_state.cfg_n_results = N_RESULTS
if "cfg_top_k_calls" not in st.session_state:
    st.session_state.cfg_top_k_calls = TOP_K_CALLS
if "cfg_strong_hit_threshold" not in st.session_state:
    st.session_state.cfg_strong_hit_threshold = STRONG_HIT_THRESHOLD
if "cfg_evidence_per_call" not in st.session_state:
    st.session_state.cfg_evidence_per_call = EVIDENCE_PER_CALL
if "cfg_xai_top_themes" not in st.session_state:
    st.session_state.cfg_xai_top_themes = XAI_TOP_THEMES
if "cfg_xai_evidence_per_call" not in st.session_state:
    st.session_state.cfg_xai_evidence_per_call = XAI_EVIDENCE_PER_CALL
if "cfg_xai_max_gaps" not in st.session_state:
    st.session_state.cfg_xai_max_gaps = XAI_MAX_GAPS
if "cfg_gap_min_call_signal" not in st.session_state:
    st.session_state.cfg_gap_min_call_signal = GAP_MIN_CALL_SIGNAL
if "cfg_gap_max_sp_signal" not in st.session_state:
    st.session_state.cfg_gap_max_sp_signal = GAP_MAX_SP_SIGNAL

if st.session_state.reset_demo_sp:
    st.session_state.selected_demo_sp = None
    st.session_state.reset_demo_sp = False


st.sidebar.header("Demo Strategic Plans")
demo_runs = load_demo_runs()
sp_options = [
    "Bocconi_Piano_Strategico2021-2025&Vision2030.pdf",
    "LUM_PIANO-STRATEGICO-DATENEO-2021-2025.pdf",
    "Naples_Federico_Piano_strategico_2021_2026.pdf",
    "Piano Strategico Luiss 2021-2024 (per sito).pdf",
    "Politechnico_di_Milano_2023-2025.pdf",
    "Sapienza_2021_2027.pdf",
]
selected_sp = st.sidebar.selectbox(
    "Choose one SP",
    sp_options,
    index=None,
    placeholder="Select a strategic plan:",
    key="selected_demo_sp",
    disabled=st.session_state.ui_busy,
)

if (
    not st.session_state.upload_mode
    and selected_sp is not None
    and selected_sp not in st.session_state.uploaded_runs
):
    st.session_state.selected_item_id = None
    st.session_state.upload_mode = False



st.sidebar.divider()
st.sidebar.subheader("Uploaded (This Session)")
if st.session_state.uploaded_runs:
    for name in list(st.session_state.uploaded_runs.keys()):
        open_col, remove_col = st.sidebar.columns([0.8, 0.2])
        with open_col:
            label = name if st.session_state.selected_item_id != name else f"> {name}"
            if st.button(
                label,
                key=f"open_uploaded_{name}",
                use_container_width=True,
                disabled=st.session_state.ui_busy,
            ):
                st.session_state.reset_demo_sp = True
                st.session_state.selected_item_id = name
                st.session_state.upload_mode = False
                st.rerun()
        with remove_col:
            if st.button(
                "X",
                key=f"remove_uploaded_{name}",
                use_container_width=True,
                disabled=st.session_state.ui_busy,
            ):
                del st.session_state.uploaded_runs[name]
                if st.session_state.selected_item_id == name:
                    remaining = list(st.session_state.uploaded_runs.keys())
                    st.session_state.selected_item_id = remaining[0] if remaining else None
                if not st.session_state.uploaded_runs and selected_sp is None:
                    st.session_state.upload_mode = True
                st.rerun()
else:
    st.sidebar.caption("No uploaded analyses yet.")

if st.sidebar.button("Upload new file", use_container_width=True, disabled=st.session_state.ui_busy):
    st.session_state.reset_demo_sp = True
    st.session_state.upload_mode = True
    st.session_state.selected_item_id = None
    st.rerun()

# MAIN





# Resolve active run first
active_item_id = st.session_state.selected_item_id
active_source = None
active_run = None

if active_item_id and active_item_id in st.session_state.uploaded_runs:
    active_run = st.session_state.uploaded_runs[active_item_id]
    active_source = "uploaded"
elif selected_sp and selected_sp in demo_runs:
    active_run = demo_runs[selected_sp]
    active_source = "demo"

analysis_ready = active_run is not None
show_entry_sections = st.session_state.upload_mode or not analysis_ready
is_demo_view = active_source == "demo"
show_input_controls = show_entry_sections and not is_demo_view


def _run_pending_action(active_item_id_value, active_run_value):
    action = st.session_state.pending_action
    if not st.session_state.ui_busy or action is None:
        return

    try:
        if action == "analysis":
            pending = st.session_state.pending_analysis_file or {}
            file_name = pending.get("name")
            file_bytes = pending.get("bytes")
            if not file_name or file_bytes is None:
                raise ValueError("No uploaded file payload found for pending analysis action")

            progress_text = st.empty()
            progress_bar = st.progress(0)

            def _on_progress(done: int, total: int) -> None:
                if total <= 0:
                    progress_bar.progress(0)
                    progress_text.info("Preparing retrieval...")
                    return

                ratio = min(1.0, max(0.0, done / total))
                progress_bar.progress(ratio)
                progress_text.info(f"Retrieving chunks: {done}/{total}")

            class _PendingUpload:
                def __init__(self, name: str, payload: bytes):
                    self.name = name
                    self._payload = payload

                def read(self):
                    return self._payload

            with st.spinner("Extracting, chunking, retrieving, and ranking..."):
                uploaded_obj = _PendingUpload(file_name, file_bytes)
                sp_obj = extract_uploaded_sp(
                    uploaded_obj,
                    size=SP_CHUNK_SIZE,
                    overlap=SP_CHUNK_OVERLAP,
                    min_chars=SP_MIN_CHUNK_CHARS,
                )
                phase1_result = retrieve_and_rank_calls_for_uploaded_sp(
                    sp_obj,
                    n_results=st.session_state.cfg_n_results,
                    strong_hit_threshold=st.session_state.cfg_strong_hit_threshold,
                    evidence_per_call=st.session_state.cfg_evidence_per_call,
                    top_k_calls=st.session_state.cfg_top_k_calls,
                    progress_callback=_on_progress,
                )

            progress_bar.progress(1.0)
            progress_text.success("Retrieval completed")
            st.success(
                f"Done. Raw hits: {phase1_result['raw_hits']} | "
                f"Top calls: {len(phase1_result['top_calls'])}"
            )

            st.session_state.latest_sp_obj = sp_obj
            st.session_state.latest_phase1 = phase1_result

            phase2_result = attach_xai_to_phase1(
                sp_obj,
                phase1_result,
                xai_evidence_per_call=st.session_state.cfg_xai_evidence_per_call,
                xai_top_themes=st.session_state.cfg_xai_top_themes,
                gap_min_call_signal=st.session_state.cfg_gap_min_call_signal,
                gap_max_sp_signal=st.session_state.cfg_gap_max_sp_signal,
                xai_max_gaps=st.session_state.cfg_xai_max_gaps,
            )
            item_id = sp_obj["source_file"]
            st.session_state.uploaded_runs[item_id] = {
                "sp_obj": sp_obj,
                "phase1": phase1_result,
                "phase2": phase2_result,
                "phase3": st.session_state.uploaded_runs.get(item_id, {}).get("phase3"),
            }
            st.session_state.reset_demo_sp = True
            st.session_state.selected_item_id = item_id
            st.session_state.upload_mode = False

        elif action == "summary":
            if active_item_id_value is None or active_run_value is None:
                raise ValueError("No active uploaded run found for pending summary action")

            phase2_result = active_run_value["phase2"]
            with st.spinner("Generating grounded executive summary..."):
                llm_out = generate_summary(
                    strategic_plan=phase2_result,
                    model=LLM_MODEL_NAME,
                    temperature=PHASE3_TEMPERATURE,
                    min_citations_per_call=PHASE3_MIN_CITATIONS_PER_CALL,
                    language=PHASE3_LANGUAGE,
                )

            phase3_result = {
                "sp_id": phase2_result.get("sp_id"),
                "sp_file": phase2_result.get("sp_file"),
                "model_name": LLM_MODEL_NAME,
                "temperature": PHASE3_TEMPERATURE,
                "structured": llm_out.get("structured", {}),
                "stakeholder_text": llm_out.get("stakeholder_text", ""),
                "prompt": llm_out.get("prompt", ""),
                "raw_response_text": llm_out.get("raw_response_text", ""),
            }
            st.session_state.uploaded_runs[active_item_id_value]["phase3"] = phase3_result
            st.session_state.latest_phase3 = phase3_result
            st.success("LLM briefing generated.")
    finally:
        st.session_state.pending_action = None
        st.session_state.pending_analysis_file = None
        st.session_state.ui_busy = False
        st.rerun()


if st.session_state.ui_busy and st.session_state.pending_action is not None:
    _run_pending_action(active_item_id, active_run)

# Upload
if show_input_controls:
    st.subheader("Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF strategic plan",
        type=["pdf"],
        disabled=st.session_state.ui_busy,
    )
else:
    uploaded_file = None


if show_input_controls:
    st.subheader("Configuration")
    st.caption("Tune ranking and explainability thresholds before running analysis.")

    h1, h2, h3 = st.columns([1.2, 3.2, 1.2])
    h1.markdown("**Threshold Name**")
    h2.markdown("**What it controls**")
    h3.markdown("**Value**")

    r1c1, r1c2, r1c3 = st.columns([1.2, 3.2, 1.2])
    r1c1.code("N_RESULTS")
    r1c2.write("How many candidate chunks are retrieved per SP chunk before call-level aggregation.")
    r1c3.number_input(
        "N_RESULTS",
        min_value=10,
        max_value=100,
        step=5,
        key="cfg_n_results",
        label_visibility="collapsed",
        disabled=st.session_state.ui_busy,
    )

    r2c1, r2c2, r2c3 = st.columns([1.2, 3.2, 1.2])
    r2c1.code("TOP_K_CALLS")
    r2c2.write("How many final calls are returned to the user after ranking.")
    r2c3.number_input(
        "TOP_K_CALLS",
        min_value=1,
        max_value=10,
        step=1,
        key="cfg_top_k_calls",
        label_visibility="collapsed",
        disabled=st.session_state.ui_busy,
    )

    r3c1, r3c2, r3c3 = st.columns([1.2, 3.2, 1.2])
    r3c1.code("STRONG_HIT_THRESHOLD")
    r3c2.write("Similarity threshold used to count strong hits in consistency scoring.")
    r3c3.slider(
        "STRONG_HIT_THRESHOLD",
        min_value=0.40,
        max_value=0.80,
        step=0.01,
        key="cfg_strong_hit_threshold",
        label_visibility="collapsed",
        disabled=st.session_state.ui_busy,
    )

    r4c1, r4c2, r4c3 = st.columns([1.2, 3.2, 1.2])
    r4c1.code("EVIDENCE_PER_CALL")
    r4c2.write("Number of top evidence chunks kept and displayed for each ranked call.")
    r4c3.number_input(
        "EVIDENCE_PER_CALL",
        min_value=1,
        max_value=8,
        step=1,
        key="cfg_evidence_per_call",
        label_visibility="collapsed",
        disabled=st.session_state.ui_busy,
    )

    r5c1, r5c2, r5c3 = st.columns([1.2, 3.2, 1.2])
    r5c1.code("XAI_TOP_THEMES")
    r5c2.write("Maximum number of matched themes shown per call in the explainability section.")
    r5c3.number_input(
        "XAI_TOP_THEMES",
        min_value=2,
        max_value=8,
        step=1,
        key="cfg_xai_top_themes",
        label_visibility="collapsed",
        disabled=st.session_state.ui_busy,
    )

    r6c1, r6c2, r6c3 = st.columns([1.2, 3.2, 1.2])
    r6c1.code("XAI_EVIDENCE_PER_CALL")
    r6c2.write("How many evidence chunks feed theme matching and gap extraction per call.")
    r6c3.number_input(
        "XAI_EVIDENCE_PER_CALL",
        min_value=1,
        max_value=6,
        step=1,
        key="cfg_xai_evidence_per_call",
        label_visibility="collapsed",
        disabled=st.session_state.ui_busy,
    )

    r7c1, r7c2, r7c3 = st.columns([1.2, 3.2, 1.2])
    r7c1.code("XAI_MAX_GAPS")
    r7c2.write("Maximum number of strategic gaps returned per call.")
    r7c3.number_input(
        "XAI_MAX_GAPS",
        min_value=1,
        max_value=8,
        step=1,
        key="cfg_xai_max_gaps",
        label_visibility="collapsed",
        disabled=st.session_state.ui_busy,
    )

    r8c1, r8c2, r8c3 = st.columns([1.2, 3.2, 1.2])
    r8c1.code("GAP_MIN_CALL_SIGNAL")
    r8c2.write("Minimum call-side theme signal required before a theme can be flagged as a gap.")
    r8c3.slider(
        "GAP_MIN_CALL_SIGNAL",
        min_value=0.10,
        max_value=0.40,
        step=0.01,
        key="cfg_gap_min_call_signal",
        label_visibility="collapsed",
        disabled=st.session_state.ui_busy,
    )

    r9c1, r9c2, r9c3 = st.columns([1.2, 3.2, 1.2])
    r9c1.code("GAP_MAX_SP_SIGNAL")
    r9c2.write("Maximum SP-side theme signal allowed when identifying underrepresented gap themes.")
    r9c3.slider(
        "GAP_MAX_SP_SIGNAL",
        min_value=0.00,
        max_value=0.25,
        step=0.01,
        key="cfg_gap_max_sp_signal",
        label_visibility="collapsed",
        disabled=st.session_state.ui_busy,
    )





# Analyze
if show_input_controls:
    st.subheader("Analyze")
    is_ready = uploaded_file is not None

    if st.button("Run Analysis", type="primary", disabled=st.session_state.ui_busy or not is_ready):
        st.session_state.pending_analysis_file = {
            "name": uploaded_file.name,
            "bytes": uploaded_file.read(),
        }
        st.session_state.ui_busy = True
        st.session_state.pending_action = "analysis"
        st.rerun()

    if not is_ready:
        st.caption("Upload a PDF first to enable analysis.")





# Results
st.subheader("Results")
if active_run is not None:
    phase2_result = active_run["phase2"]

    raw_hits_value = phase2_result.get("raw_hits")
    raw_hits_display = raw_hits_value if raw_hits_value is not None else "precomputed"
    st.write(
        f"SP: `{phase2_result['sp_file']}` | "
        f"Raw hits: `{raw_hits_display}` | "
        f"Top calls: `{len(phase2_result['top_calls'])}`"
    )

    for i, call in enumerate(phase2_result["top_calls"], start=1):
        xai = call.get("xai", {})
        st.markdown(f"### {i}) {call['call_name']}")
        st.write(
            f"Final score: `{call['final_score']}` | "
            f"Semantic: `{call['semantic_score']}` | "
            f"Coverage: `{call['coverage_score']}` | "
            f"Consistency: `{call['consistency_score']}`"
        )

        with st.expander("Matched themes", expanded=False):
            for row in xai.get("matched_themes", []):
                st.write(
                    f"- {row['theme']} | "
                    f"match={row['match_strength']} "
                    f"(sp={row['sp_score']}, call={row['call_score']})"
                )

        with st.expander("Key gaps / recommended actions", expanded=False):
            gaps = xai.get("key_gaps", [])
            if not gaps:
                st.write("- No major gaps detected.")
            else:
                for gap in gaps:
                    st.write(f"- {gap['action']} (gap={gap['gap_strength']})")

        with st.expander("Supporting evidence", expanded=False):
            for ev in xai.get("supporting_chunks", []):
                page = ev["page"] if ev["page"] is not None else "n/a"
                st.write(
                    f"- source={ev['source_file']} | page={page} | sim={ev['similarity']}"
                )
                st.caption(ev.get("excerpt", ""))

else:
    st.write("Results will be shown here.")



# Summary


st.subheader("Summary")

if active_run is not None:
    phase2_result = active_run["phase2"]

    if active_source == "uploaded":
        has_phase3 = active_run.get("phase3") is not None
        if not has_phase3 and st.button(
            "Generate LLM Briefing",
            type="secondary",
            disabled=st.session_state.ui_busy,
        ):
            st.session_state.ui_busy = True
            st.session_state.pending_action = "summary"
            st.rerun()

    if active_source == "demo" and active_run.get("phase3") is None:
        st.caption("No precomputed summary found for this demo strategic plan.")

    phase3 = active_run.get("phase3")
    if phase3 is not None:

        structured = phase3.get("structured", {})

        st.markdown("### Executive Briefing")

        executive_summary = structured.get("executive_summary", "").strip()
        if executive_summary:
            st.markdown(executive_summary)

        confidence = structured.get("confidence", "n/a")
        st.markdown(f"**Confidence:** {confidence}")

        st.markdown("#### Top 3 Calls")
        top_calls = structured.get("top_calls", [])
        if not top_calls:
            st.write("No calls returned.")
        else:
            for call in top_calls:
                rank = call.get("rank", "-")
                name = call.get("call_name", "Unknown call")
                why_match = call.get("why_match", "")
                call_confidence = call.get("confidence", "n/a")

                st.markdown(f"**{rank}. {name}**")
                if why_match:
                    st.write(why_match)
                st.caption(f"Call confidence: {call_confidence}")

                citations = call.get("citations", [])
                if citations:
                    citation_lines = []
                    for citation in citations:
                        source_file = citation.get("source_file", "unknown")
                        page = citation.get("page", "n/a")
                        citation_lines.append(f"- {source_file} (page {page})")
                    st.markdown("Citations:\n" + "\n".join(citation_lines))

                st.divider()

        st.markdown("#### Recommended Focus Areas")
        actions = structured.get("improvement_actions", [])
        if not actions:
            st.write("No recommended actions returned.")
        else:
            for action in actions:
                st.markdown(f"- {action}")
        st.caption("In order to shorten the running time, the output for the LLM summary has been limited.")
else:
    st.caption("Run analysis first to enable LLM briefing.")
