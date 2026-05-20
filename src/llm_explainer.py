"""
llm_explainer.py

What this file contains:
- Phase 3 summarization utilities for transforming structured evidence into stakeholder-facing output.
- Prompt construction for grounded JSON + text responses with citation and confidence policy constraints.
- Response parsing, recovery/retry logic, and output normalization.
- Post-processing safeguards (including citation enforcement) before returning structured results.

Role in the whole project:
- This is the communication layer that converts analytical outputs into decision-ready briefings.
- It preserves traceability by tying narrative claims to retrieved evidence and citations.
- It enables reproducible, policy-constrained natural-language reporting on top of the deterministic retrieval/XAI pipeline.
"""

from __future__ import annotations

import json
from typing import Dict

from ollama import Client

DEFAULT_MODEL = "qwen3:0.6b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_RETRIES = 1
DEFAULT_NUM_PREDICT = 2000
DEFAULT_NUM_CTX = 8192


def _build_payload(strategic_plan: Dict) -> Dict:
    calls_payload = []
    for rank, call in enumerate(strategic_plan.get("top_calls", []), start=1):
        xai = call.get("xai", {})
        supporting = []
        for ev in xai.get("supporting_chunks", []):
            supporting.append(
                {
                    "source_file": ev.get("source_file"),
                    "page": ev.get("page"),
                    "similarity": ev.get("similarity"),
                    "excerpt": ev.get("excerpt", ""),
                }
            )

        calls_payload.append(
            {
                "rank": rank,
                "call_name": xai.get("call_name", call.get("call_name")),
                "scores": {
                    "final_score": xai.get("final_score", call.get("final_score")),
                    "semantic_score": xai.get("score_breakdown", {}).get("semantic_score", call.get("semantic_score")),
                    "coverage_score": xai.get("score_breakdown", {}).get("coverage_score", call.get("coverage_score")),
                    "consistency_score": xai.get("score_breakdown", {}).get("consistency_score", call.get("consistency_score")),
                },
                "matched_themes": xai.get("matched_themes", []),
                "key_gaps": xai.get("key_gaps", []),
                "supporting_chunks": supporting,
            }
        )

    return {
        "sp_id": strategic_plan.get("sp_id"),
        "sp_file": strategic_plan.get("sp_file"),
        "raw_hits": strategic_plan.get("raw_hits"),
        "top_calls": calls_payload,
    }


def _normalize_citations(citations) -> list[Dict]:
    if not isinstance(citations, list):
        return []

    normalized = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        source_file = item.get("source_file")
        if not source_file:
            continue
        normalized.append({"source_file": source_file, "page": item.get("page")})
    return normalized


def _enforce_min_citations(structured: Dict, strategic_plan: Dict, min_citations_per_call: int) -> Dict:
    calls = structured.get("top_calls")
    source_calls = strategic_plan.get("top_calls", [])
    if not isinstance(calls, list):
        return structured

    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            continue

        citations = _normalize_citations(call.get("citations", []))
        seen = {(c.get("source_file"), c.get("page")) for c in citations}

        source_evidence = []
        if idx < len(source_calls):
            source_evidence = source_calls[idx].get("xai", {}).get("supporting_chunks", [])

        for ev in source_evidence:
            source_file = ev.get("source_file")
            page = ev.get("page")
            key = (source_file, page)
            if not source_file or key in seen:
                continue
            citations.append({"source_file": source_file, "page": page})
            seen.add(key)
            if len(citations) >= min_citations_per_call:
                break

        if len(citations) < min_citations_per_call:
            call["confidence"] = "Low"

        call["citations"] = citations

    return structured


def _build_prompt(payload: Dict, min_citations_per_call: int = 2, language: str = "English") -> str:
    schema = {
        "sp_id": "string",
        "sp_file": "string",
        "confidence": "High | Medium | Low",
        "executive_summary": "2-4 sentences, stakeholder style",
        "top_calls": [
            {
                "rank": "integer",
                "call_name": "string",
                "why_match": "Exactly 3 sentences: (1) SP priority/theme, (2) matching call requirement/opportunity, (3) explicit alignment reasoning with evidence strength",
                "citations": [
                    {"source_file": "string", "page": "int|null"},
                    {"source_file": "string", "page": "int|null"},
                ],
                "confidence": "High | Medium | Low",
            }
        ],
        "improvement_actions": [
            "up to 5 action bullets based only on provided key_gaps/evidence"
        ],
    }

    instructions = f"""
You are an executive advisor. Write in {language}.
Use only the provided evidence. Do not invent facts.
For each top call, explain MATCHING, not call description.
Each top call must include at least {min_citations_per_call} citations from provided chunks.
Each top call should include at least one citation supporting the SP-side claim and one citation supporting the call-side claim when available.
why_match must follow this exact pattern:
- Sentence 1: what the SP prioritizes (theme/goal).
- Sentence 2: what the call supports/requires on that same theme.
- Sentence 3: why this is a strong/medium/weak fit for this SP specifically.
Do not summarize the call in isolation.
Use comparative language: "SP emphasizes...", "The call supports...", "Therefore alignment is ... because...".
Assign confidence as High/Medium/Low. If evidence is weak or citations are missing, downgrade confidence.
Output MUST start with <JSON> and end with </TEXT>. Do not add any preface, code fences, or extra sections.
Return two sections exactly:
<JSON>
{{valid JSON matching schema}}
</JSON>
<TEXT>
Readable stakeholder briefing with headings:
- Top 3 Calls
- Why They Match
- Recommended Focus Areas
- Confidence Note
</TEXT>
""".strip()

    strict_example = """
<JSON>
{"sp_id":"...","sp_file":"...","confidence":"Medium","executive_summary":"...","top_calls":[{"rank":1,"call_name":"...","why_match":"SP emphasizes digital transformation and AI-enabled teaching modernization. The call explicitly funds AI/data capability building and university-industry transfer activities. Therefore this call is a medium-to-strong fit because its funded actions directly map to the SP's AI and skills priorities, though governance deliverables are less explicit in the SP.","citations":[{"source_file":"...","page":1},{"source_file":"...","page":2}],"confidence":"Medium"}],"improvement_actions":["..."]}
</JSON>
<TEXT>
Top 3 Calls\n...\nWhy They Match\n...\nRecommended Focus Areas\n...\nConfidence Note\n...
</TEXT>
""".strip()

    return (
        "/no_think\n"
        + instructions
        + "\n\nRequired output template example:\n"
        + strict_example
        + "\n\nJSON schema:\n"
        + json.dumps(schema, ensure_ascii=True, indent=2)
        + "\n\nInput payload:\n"
        + json.dumps(payload, ensure_ascii=True, indent=2)
    )


def _extract_first_json_object(text: str) -> Dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in model response")

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                return json.loads(candidate)

    raise ValueError("Could not find a balanced JSON object in model response")


def _parse_llm_response(text: str) -> Dict:
    json_start = text.find("<JSON>")
    json_end = text.find("</JSON>")
    text_start = text.find("<TEXT>")
    text_end = text.find("</TEXT>")

    if json_start != -1 and json_end != -1 and text_start != -1 and text_end != -1:
        json_block = text[json_start + len("<JSON>") : json_end].strip()
        text_block = text[text_start + len("<TEXT>") : text_end].strip()
        parsed = json.loads(json_block)
        return {
            "json": parsed,
            "text": text_block,
            "json_block_raw": json_block,
        }

    # If wrappers are partially followed, still try to parse JSON block.
    if json_start != -1 and json_end != -1:
        json_block = text[json_start + len("<JSON>") : json_end].strip()
        parsed = json.loads(json_block)
        return {
            "json": parsed,
            "text": "",
            "json_block_raw": json_block,
        }

    # Fallback parser for local models that ignore wrappers
    parsed = _extract_first_json_object(text)
    text_block = text.strip()
    return {
        "json": parsed,
        "text": text_block,
        "json_block_raw": json.dumps(parsed, ensure_ascii=True),
    }


def _request_reformat_response(
    client: Client,
    model: str,
    prior_output: str,
    temperature: float,
    num_predict: int,
    num_ctx: int,
) -> str:
    reformat_prompt = (
        "Reformat your previous answer into EXACTLY two wrapper sections and do not change factual content.\n"
        "Output must be:\n<JSON>...valid JSON...</JSON>\n<TEXT>...readable briefing...</TEXT>\n\n"
        "Previous answer to reformat:\n"
        f"{prior_output}"
    )
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": reformat_prompt}],
        options={"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    )
    return response.get("message", {}).get("content", "")


def generate_summary(
    strategic_plan: Dict,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    min_citations_per_call: int = 2,
    language: str = "English",
    max_retries: int = DEFAULT_MAX_RETRIES,
    num_predict: int = DEFAULT_NUM_PREDICT,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> Dict:
    payload = _build_payload(strategic_plan)
    prompt = _build_prompt(payload, min_citations_per_call=min_citations_per_call, language=language)

    client = Client()
    def _chat_response_text(predict_limit: int) -> tuple[str, str]:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": predict_limit, "num_ctx": num_ctx},
        )
        message = getattr(response, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        thinking = getattr(message, "thinking", "") if message is not None else ""
        return content, thinking

    response_text, thinking_text = _chat_response_text(num_predict)
    if not response_text:
        if thinking_text:
            retry_predict = max(num_predict * 2, 1200)
            response_text, retry_thinking = _chat_response_text(retry_predict)
            if not response_text:
                raise ValueError(
                    "Empty response from Ollama model after retry. "
                    f"Thinking snippet: {str(retry_thinking)[:300]}"
                )
        else:
            raise ValueError("Empty response from Ollama model")

    try:
        parsed = _parse_llm_response(response_text)
    except Exception as first_err:
        # First, retry a fresh generation with a larger budget for likely truncation.
        grew_predict = max(num_predict * 2, 3000)
        retry_text, _ = _chat_response_text(grew_predict)
        if retry_text:
            try:
                parsed = _parse_llm_response(retry_text)
                response_text = retry_text
                return {
                    "prompt": prompt,
                    "raw_response_text": response_text,
                    "structured": parsed["json"],
                    "stakeholder_text": parsed["text"],
                }
            except Exception:
                pass

        if max_retries <= 0:
            snippet = response_text[:800]
            raise ValueError(f"Failed to parse model response: {first_err}. Raw snippet: {snippet}") from first_err

        reformatted_text = _request_reformat_response(
            client,
            model,
            response_text,
            temperature,
            num_predict,
            num_ctx,
        )
        if not reformatted_text:
            snippet = response_text[:800]
            raise ValueError(f"Model reformat retry returned empty text. Raw snippet: {snippet}") from first_err

        try:
            parsed = _parse_llm_response(reformatted_text)
            response_text = reformatted_text
        except Exception as second_err:
            snippet = reformatted_text[:800]
            raise ValueError(
                f"Failed to parse model response after retry: {second_err}. Raw snippet: {snippet}"
            ) from second_err

    return {
        "prompt": prompt,
        "raw_response_text": response_text,
        "structured": _enforce_min_citations(
            structured=parsed["json"],
            strategic_plan=strategic_plan,
            min_citations_per_call=min_citations_per_call,
        ),
        "stakeholder_text": parsed["text"],
    }
