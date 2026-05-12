from __future__ import annotations

from io import BytesIO
import re
from typing import Dict, List

from pypdf import PdfReader


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int, overlap: int, min_chars: int) -> List[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + size, text_len)
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)

        if end == text_len:
            break

        start += size - overlap

    return chunks


def extract_uploaded_sp(uploaded_file, size: int, overlap: int, min_chars: int) -> Dict:
    file_bytes = uploaded_file.read()
    reader = PdfReader(BytesIO(file_bytes))

    page_texts: List[str] = []
    for page in reader.pages:
        raw_text = page.extract_text() or ""
        text = clean_text(raw_text)
        if len(text) >= 40:
            page_texts.append(text)

    full_text = "\n".join(page_texts)
    chunks = chunk_text(full_text, size=size, overlap=overlap, min_chars=min_chars)

    return {
        "sp_id": uploaded_file.name.rsplit(".", 1)[0],
        "source_file": uploaded_file.name,
        "text": full_text,
        "chunks": chunks,
        "non_empty_pages": len(page_texts),
    }


def similarity_from_distance(distance: float) -> float:
    return 1.0 / (1.0 + float(distance))


def rank_calls_for_sp_query(
    collection,
    sp: Dict,
    n_results: int,
    strong_hit_threshold: float,
    evidence_per_call: int,
    top_k_calls: int,
) -> Dict:
    by_call: Dict[str, Dict] = {}
    raw_hits = 0

    query_texts = sp.get("chunks") or [sp["text"]]

    for chunk_idx, sp_chunk in enumerate(query_texts):
        result = collection.query(
            query_texts=[sp_chunk],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )

        metadatas = result["metadatas"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]
        raw_hits += len(metadatas)

        for meta, doc, dist in zip(metadatas, documents, distances):
            call = meta.get("source_file", "unknown")
            page = meta.get("page", None)
            sim = similarity_from_distance(dist)

            if call not in by_call:
                by_call[call] = {
                    "call_name": call,
                    "similarities": [],
                    "pages": set(),
                    "evidence": [],
                }

            by_call[call]["similarities"].append(sim)
            if page is not None:
                by_call[call]["pages"].add(page)

            by_call[call]["evidence"].append(
                {
                    "source_file": meta.get("source_file", call),
                    "source_path": meta.get("source_path", ""),
                    "page": page,
                    "similarity": round(sim, 4),
                    "sp_chunk_index": chunk_idx,
                    "chunk": doc[:700],
                }
            )

    ranked = []
    for call, payload in by_call.items():
        sims = sorted(payload["similarities"], reverse=True)
        top_sims = sims[:5]

        semantic_score = sum(top_sims) / max(1, len(top_sims))
        coverage_score = min(1.0, len(payload["pages"]) / 6.0)
        strong_hits = sum(1 for s in sims if s >= strong_hit_threshold)
        consistency_score = min(1.0, strong_hits / 5.0)
        final_score = 0.7 * semantic_score + 0.2 * coverage_score + 0.1 * consistency_score

        ranked.append(
            {
                "call_name": call,
                "semantic_score": round(semantic_score, 4),
                "coverage_score": round(coverage_score, 4),
                "consistency_score": round(consistency_score, 4),
                "final_score": round(final_score, 4),
                "evidence": sorted(payload["evidence"], key=lambda x: x["similarity"], reverse=True)[:evidence_per_call],
            }
        )

    ranked = sorted(ranked, key=lambda x: x["final_score"], reverse=True)

    return {
        "sp_id": sp["sp_id"],
        "sp_file": sp["source_file"],
        "sp_chunk_count": len(query_texts),
        "top_calls": ranked[:top_k_calls],
        "raw_hits": raw_hits,
    }


def compute_theme_signal_from_text(text: str, theme_lexicon: Dict[str, List[str]]) -> Dict[str, Dict]:
    normalized = clean_text(text).lower()
    raw_hits = {}
    total_hits = 0

    for theme, keywords in theme_lexicon.items():
        hits = 0
        matched = []
        for keyword in keywords:
            count = normalized.count(keyword.lower())
            if count > 0:
                hits += count
                matched.append(keyword)
        raw_hits[theme] = {"hits": hits, "matched_keywords": matched}
        total_hits += hits

    signal = {}
    for theme, payload in raw_hits.items():
        score = (payload["hits"] / total_hits) if total_hits > 0 else 0.0
        signal[theme] = {
            "hits": payload["hits"],
            "score": round(score, 4),
            "matched_keywords": payload["matched_keywords"][:8],
        }

    return signal


def derive_matched_themes(
    sp_signal: Dict,
    call_signal: Dict,
    theme_lexicon: Dict[str, List[str]],
    top_n: int,
) -> List[Dict]:
    rows = []
    for theme in theme_lexicon.keys():
        sp_score = sp_signal.get(theme, {}).get("score", 0.0)
        call_score = call_signal.get(theme, {}).get("score", 0.0)
        rows.append(
            {
                "theme": theme,
                "sp_score": round(sp_score, 4),
                "call_score": round(call_score, 4),
                "match_strength": round(min(sp_score, call_score), 4),
                "keywords_in_call": call_signal.get(theme, {}).get("matched_keywords", [])[:5],
            }
        )

    rows.sort(key=lambda x: (x["match_strength"], x["call_score"]), reverse=True)
    return rows[:top_n]


def derive_key_gaps(
    sp_signal: Dict,
    call_signal: Dict,
    theme_lexicon: Dict[str, List[str]],
    gap_min_call_signal: float,
    gap_max_sp_signal: float,
    max_gaps: int,
) -> List[Dict]:
    action_map = {
        "sustainability_climate": "SP should strengthen measurable sustainability and climate actions with clear KPIs.",
        "ai_data_digital": "SP should strengthen AI/data/digital transition actions with concrete implementation milestones.",
        "internationalization_collaboration": "SP should strengthen international collaboration targets (consortia, mobility, partnerships).",
        "innovation_transfer_industry": "SP should strengthen technology transfer and industry engagement pathways.",
        "skills_education_capacity": "SP should strengthen skills and capacity-building plans with measurable outcomes.",
        "inclusion_gender_social": "SP should strengthen inclusion and gender/social impact commitments with indicators.",
        "research_infrastructure_excellence": "SP should strengthen research infrastructure and excellence positioning.",
        "governance_policy_reform": "SP should strengthen governance and implementation roadmap clarity.",
    }

    candidates = []
    for theme in theme_lexicon.keys():
        sp_score = sp_signal.get(theme, {}).get("score", 0.0)
        call_score = call_signal.get(theme, {}).get("score", 0.0)
        if call_score >= gap_min_call_signal and sp_score <= gap_max_sp_signal:
            candidates.append((theme, call_score - sp_score, sp_score, call_score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    gaps = []
    for theme, gap_strength, sp_score, call_score in candidates[:max_gaps]:
        gaps.append(
            {
                "theme": theme,
                "gap_strength": round(gap_strength, 4),
                "sp_score": round(sp_score, 4),
                "call_score": round(call_score, 4),
                "action": action_map.get(theme, "SP should improve strategic alignment for this theme."),
            }
        )

    return gaps


def build_call_xai(
    sp_text: str,
    call_row: Dict,
    theme_lexicon: Dict[str, List[str]],
    xai_evidence_per_call: int,
    xai_top_themes: int,
    gap_min_call_signal: float,
    gap_max_sp_signal: float,
    xai_max_gaps: int,
) -> Dict:
    evidence = call_row.get("evidence", [])[:xai_evidence_per_call]
    evidence_text = " ".join([item.get("chunk", "") for item in evidence])

    sp_signal = compute_theme_signal_from_text(sp_text, theme_lexicon)
    call_signal = compute_theme_signal_from_text(evidence_text, theme_lexicon)

    supporting_chunks = []
    for item in evidence:
        supporting_chunks.append(
            {
                "source_file": item.get("source_file", call_row.get("call_name")),
                "source_path": item.get("source_path", ""),
                "page": item.get("page"),
                "similarity": item.get("similarity"),
                "excerpt": item.get("chunk", "")[:320],
            }
        )

    return {
        "call_name": call_row.get("call_name"),
        "final_score": call_row.get("final_score"),
        "score_breakdown": {
            "semantic_score": call_row.get("semantic_score"),
            "coverage_score": call_row.get("coverage_score"),
            "consistency_score": call_row.get("consistency_score"),
        },
        "supporting_chunks": supporting_chunks,
        "matched_themes": derive_matched_themes(
            sp_signal,
            call_signal,
            theme_lexicon,
            top_n=xai_top_themes,
        ),
        "key_gaps": derive_key_gaps(
            sp_signal,
            call_signal,
            theme_lexicon,
            gap_min_call_signal=gap_min_call_signal,
            gap_max_sp_signal=gap_max_sp_signal,
            max_gaps=xai_max_gaps,
        ),
    }
