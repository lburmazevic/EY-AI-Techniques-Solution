import json
import re
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions

from llm_explainer import generate_summary
from pipeline_core_methods import build_call_xai, extract_uploaded_sp, rank_calls_for_sp_query

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
    progress_callback=None,
) -> dict:
    collection = get_collection()
    return rank_calls_for_sp_query(
        collection=collection,
        sp=sp_obj,
        n_results=n_results,
        strong_hit_threshold=STRONG_HIT_THRESHOLD,
        evidence_per_call=EVIDENCE_PER_CALL,
        top_k_calls=TOP_K_CALLS,
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



def attach_xai_to_phase1(sp_obj: dict, phase1_result: dict) -> dict:
    calls_with_xai = []
    for call in phase1_result.get("top_calls", []):
        call_copy = dict(call)
        call_copy["xai"] = build_call_xai(
            sp_text=sp_obj.get("text", ""),
            call_row=call_copy,
            theme_lexicon=THEME_LEXICON,
            xai_evidence_per_call=XAI_EVIDENCE_PER_CALL,
            xai_top_themes=XAI_TOP_THEMES,
            gap_min_call_signal=GAP_MIN_CALL_SIGNAL,
            gap_max_sp_signal=GAP_MAX_SP_SIGNAL,
            xai_max_gaps=XAI_MAX_GAPS,
        )
        calls_with_xai.append(call_copy)

    return {
        "sp_id": phase1_result.get("sp_id"),
        "sp_file": phase1_result.get("sp_file"),
        "raw_hits": phase1_result.get("raw_hits"),
        "non_empty_pages": phase1_result.get("non_empty_pages"),
        "top_calls": calls_with_xai,
    }


def plain_briefing(phase3: dict) -> str:
    text = (phase3.get("stakeholder_text") or "").strip()

    match = re.search(r"<TEXT>(.*?)</TEXT>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    text = re.sub(r"</?JSON>|</?TEXT>", "", text, flags=re.IGNORECASE).strip()

    if not text:
        structured = phase3.get("structured", {})
        lines = []
        executive_summary = structured.get("executive_summary")
        if executive_summary:
            lines.append(executive_summary)
            lines.append("")

        lines.append("Top 3 Calls")
        for call in structured.get("top_calls", []):
            lines.append(f"- {call.get('call_name', '')}: {call.get('why_match', '')}")

        lines.append("")
        lines.append("Recommended Focus Areas")
        for action in structured.get("improvement_actions", []):
            lines.append(f"- {action}")

        lines.append("")
        lines.append(f"Confidence: {structured.get('confidence', 'n/a')}")
        text = "\n".join(lines).strip()

    return text



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
)

if selected_sp is not None and selected_sp not in st.session_state.uploaded_runs:
    st.session_state.selected_item_id = None
    st.session_state.upload_mode = False



st.sidebar.divider()
st.sidebar.subheader("Uploaded (This Session)")
if st.session_state.uploaded_runs:
    for name in list(st.session_state.uploaded_runs.keys()):
        open_col, remove_col = st.sidebar.columns([0.8, 0.2])
        with open_col:
            label = name if st.session_state.selected_item_id != name else f"> {name}"
            if st.button(label, key=f"open_uploaded_{name}", use_container_width=True):
                st.session_state.selected_demo_sp = None
                st.session_state.selected_item_id = name
                st.session_state.upload_mode = False
        with remove_col:
            if st.button("X", key=f"remove_uploaded_{name}", use_container_width=True):
                del st.session_state.uploaded_runs[name]
                if st.session_state.selected_item_id == name:
                    remaining = list(st.session_state.uploaded_runs.keys())
                    st.session_state.selected_item_id = remaining[0] if remaining else None
                if not st.session_state.uploaded_runs and selected_sp is None:
                    st.session_state.upload_mode = True
                st.rerun()
else:
    st.sidebar.caption("No uploaded analyses yet.")

if st.sidebar.button("Upload new file", use_container_width=True):
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

# Upload
if show_entry_sections:
    st.subheader("Upload")
    uploaded_file = st.file_uploader("Upload a PDF strategic plan", type=["pdf"])
else:
    uploaded_file = None





# Analyze
if show_entry_sections:
    st.subheader("Analyze")
    is_ready = uploaded_file is not None

    if st.button("Run Analysis", type="primary", disabled=not is_ready):
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

        with st.spinner("Extracting, chunking, retrieving, and ranking..."):
            sp_obj = extract_uploaded_sp(
                uploaded_file,
                size=SP_CHUNK_SIZE,
                overlap=SP_CHUNK_OVERLAP,
                min_chars=SP_MIN_CHUNK_CHARS,
            )
            phase1_result = retrieve_and_rank_calls_for_uploaded_sp(
                sp_obj,
                n_results=30,
                progress_callback=_on_progress,
            )

        progress_bar.progress(1.0)
        progress_text.success("Retrieval completed")

        st.success(
            f"Done. Raw hits: {phase1_result['raw_hits']} | "
            f"Top calls: {len(phase1_result['top_calls'])}"
        )

    # Save in session for Step 4 (XAI + rendering)
        st.session_state.latest_sp_obj = sp_obj
        st.session_state.latest_phase1 = phase1_result

        phase2_result = attach_xai_to_phase1(sp_obj, phase1_result)
        item_id = sp_obj["source_file"]
        st.session_state.uploaded_runs[item_id] = {
            "sp_obj": sp_obj,
            "phase1": phase1_result,
            "phase2": phase2_result,
            "phase3": st.session_state.uploaded_runs.get(item_id, {}).get("phase3"),
        }
        st.session_state.selected_demo_sp = None
        st.session_state.selected_item_id = item_id
        st.session_state.upload_mode = False
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

    if active_source == "uploaded" and st.button("Generate LLM Briefing", type="secondary"):
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
        st.session_state.uploaded_runs[active_item_id]["phase3"] = phase3_result
        st.session_state.latest_phase3 = phase3_result
        st.success("LLM briefing generated.")

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
