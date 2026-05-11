from __future__ import annotations

import json
from typing import Dict, List

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
                "why_match": "2-3 sentences",
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
Each top call must include at least {min_citations_per_call} citations from provided chunks.
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
{"sp_id":"...","sp_file":"...","confidence":"Medium","executive_summary":"...","top_calls":[{"rank":1,"call_name":"...","why_match":"...","citations":[{"source_file":"...","page":1},{"source_file":"...","page":2}],"confidence":"Medium"}],"improvement_actions":["..."]}
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
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    )

    message = getattr(response, "message", None)
    response_text = getattr(message, "content", "") if message is not None else ""
    thinking_text = getattr(message, "thinking", "") if message is not None else ""
    if not response_text:
        if thinking_text:
            retry_predict = max(num_predict * 2, 1200)
            response_retry = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_predict": retry_predict, "num_ctx": num_ctx},
            )
            retry_message = getattr(response_retry, "message", None)
            response_text = getattr(retry_message, "content", "") if retry_message is not None else ""
            if not response_text:
                retry_thinking = getattr(retry_message, "thinking", "") if retry_message is not None else ""
                raise ValueError(
                    "Empty response from Ollama model after retry. "
                    f"Thinking snippet: {str(retry_thinking)[:300]}"
                )
        else:
            raise ValueError("Empty response from Ollama model")

    try:
        parsed = _parse_llm_response(response_text)
    except Exception as first_err:
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
        "structured": parsed["json"],
        "stakeholder_text": parsed["text"],
    }
