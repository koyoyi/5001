"""Offline checks for the RAG demo pipeline: diagnostics, held-out guard, generator normalisation."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

from ragstudy.claude_generation import (FALLBACK_BETA, MODEL, ClaudeGenerator, build_request,
                                        document_text, parse_response)
from ragstudy.pipeline import QuestionIndex, explain, normalize_qwen, retrieval_diagnostics

DEV = {"query_id": "DEV_Q1", "split": "dev", "question": "q", "gold": ["gold.txt"],
       "candidates": {"gold.txt", "a.txt", "b.txt"}, "answer_location": "head"}


def result_for(diagnostics, generation=None, evidence=()):
    return {"question": "q", "top_k": 3, "candidate_count": 150, "diagnostics": diagnostics,
            "evidence": list(evidence), "generation": generation or {"status": "not_run"}}


def test_held_out_question_shows_no_gold():
    test_entry = {"query_id": "TEST_Q1", "split": "test", "question": "q"}
    d = retrieval_diagnostics(test_entry, ["gold.txt"], 3)
    assert d["held_out"] and "gold_document_ids" not in d
    notes = explain(result_for(d))
    assert "Held-out" in notes[0][1] and not any("gold.txt" in note for _, note in notes)


def test_retrieval_ranking_and_hit_notes():
    miss = dict(DEV, candidates={"a.txt"}, answer_location="tail")
    note = explain(result_for(retrieval_diagnostics(miss, ["a.txt"], 3)))[0][1]
    assert note.startswith("Retrieval miss") and "after word 256" in note
    ranked = retrieval_diagnostics(DEV, ["a.txt", "b.txt", "c.txt", "gold.txt"], 3)
    assert ranked["gold_rank"] == 4 and not ranked["gold_in_evidence"]
    assert "at rank 4" in explain(result_for(ranked))[0][1]
    hit = retrieval_diagnostics(DEV, ["gold.txt"], 3)
    assert hit["gold_in_evidence"] and "rank 1" in explain(result_for(hit))[0][1]


def test_index_prefers_test_match_for_duplicated_text(tmp_path):
    manifest = [{"query_id": "DEV_Q1", "split": "dev", "question": "Same  question", "positive_document_ids": ["g"]},
                {"query_id": "TEST_Q1", "split": "test", "question": "same question", "positive_document_ids": ["g"]}]
    runs = [{"query_id": q["query_id"], "candidate_ids": ["g"]} for q in manifest]
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "candidate_runs.json").write_text(json.dumps(runs), encoding="utf-8")
    index = QuestionIndex(tmp_path, tmp_path)
    assert index.lookup("SAME question")["split"] == "test"
    assert "gold" not in index.queries["TEST_Q1"] and index.lookup("other") is None

QWEN_ANSWERED = {"status": "two_stage_citation_proposed_semantics_unverified", "raw_output": "Use version 8.",
                 "raw_citation_output": "[D2]", "citation_map": {"[D1]": "a.txt", "[D2]": "gold.txt"},
                 "question_truncated": True, "evidence": [{"label": "[D1]", "excerpt_truncated": True}]}


def test_normalize_qwen_answer_label_only_and_abstention():
    answered = normalize_qwen(QWEN_ANSWERED)
    assert answered["status"] == "answered" and answered["citations"][0]["document_id"] == "gold.txt"
    assert answered["truncated_labels"] == ["[D1]"] and answered["question_truncated"]
    # A bare label passed the old check in the saved run; it must now count as no answer.
    assert normalize_qwen({**QWEN_ANSWERED, "raw_output": "D3"})["status"] == "no_answer_content"
    abstained = normalize_qwen({**QWEN_ANSWERED, "status": "abstained", "raw_output": "Insufficient evidence."})
    assert abstained["status"] == "abstained" and abstained["answer"] is None
    notes = explain(result_for(retrieval_diagnostics(DEV, ["gold.txt"], 3), abstained))
    assert any("over-cautious" in note for _, note in notes)


EVIDENCE = [{"label": "[D1]", "document_id": "a.txt", "text": "Port 8443 is used by the console."},
            {"label": "[D2]", "document_id": "gold.txt", "text": "Snapshots are kept for seven days."}]


def span(text, index, start, cited):
    return NS(type="char_location", cited_text=cited, document_index=index, document_title=None,
              start_char_index=start, end_char_index=start + len(cited))


def response(*blocks, stop="end_turn", model=MODEL, iterations=None):
    return NS(content=list(blocks), stop_reason=stop, stop_details=None, model=model,
              usage=NS(input_tokens=900, output_tokens=40, iterations=iterations))


def test_claude_parse_checks_every_cited_span():
    parsed = parse_response(response(
        NS(type="thinking", thinking=""),
        NS(type="text", text="Snapshots are kept for seven days", citations=[span("", 1, 0, "Snapshots are kept for seven days")]),
        NS(type="text", text=" and the console uses port 9999", citations=[span("", 0, 0, "Port 9999")]),
        NS(type="text", text=", per policy.", citations=None)), EVIDENCE)
    assert parsed["status"] == "answered"
    assert [c["verified"] for c in parsed["citations"]] == [True, False]
    assert parsed["citations"][0]["document_id"] == "gold.txt" and parsed["uncited_characters"] == len("per policy")
    notes = explain(result_for(retrieval_diagnostics(DEV, ["gold.txt"], 3), {"generator": MODEL, **parsed}))
    assert any("1 of 2 cited spans" in note for _, note in notes)


def test_claude_parse_abstention_refusal_and_fallback():
    assert parse_response(response(NS(type="text", text="INSUFFICIENT_EVIDENCE", citations=None)), EVIDENCE)[
        "status"] == "abstained"
    refused = parse_response(response(stop="refusal"), EVIDENCE)
    assert refused["status"] == "refused" and refused["answer"] is None
    served = parse_response(response(NS(type="text", text="ok", citations=None), model="claude-opus-4-8",
                                     iterations=[NS(type="fallback_message")]), EVIDENCE)
    assert served["fallback_served"] and served["served_by"] == "claude-opus-4-8"

def test_document_text_keeps_an_exact_prefix():
    source = "alpha  beta\ngamma delta"
    text, cut = document_text(source, max_words=3)
    assert cut and source.startswith(text) and text.split() == ["alpha", "beta", "gamma"]
    assert document_text(source, max_words=10) == (source, False)


VENDOR = Path(__file__).resolve().parents[1] / ".vendor-claude"
REPLY = {"id": "msg_test", "type": "message", "role": "assistant", "model": MODEL, "stop_reason": "end_turn",
         "stop_sequence": None, "usage": {"input_tokens": 900, "output_tokens": 20},
         "content": [{"type": "text", "text": "Snapshots are kept for seven days.", "citations": [
             {"type": "char_location", "cited_text": "Snapshots are kept for seven days.", "document_index": 1,
              "document_title": "[D2] gold.txt", "start_char_index": 0, "end_char_index": 34}]}]}


def sdk_generator(status, body, seen):
    if VENDOR.is_dir() and str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")

    def handler(request):
        seen.update(headers=request.headers, body=json.loads(request.content))
        return httpx2.Response(status, json=body)
    client = anthropic.Anthropic(api_key="test-key", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    return ClaudeGenerator(client)


def test_claude_generator_round_trip_through_sdk():
    seen = {}
    out = sdk_generator(200, REPLY, seen).generate("How long are snapshots kept?", EVIDENCE)
    assert out["status"] == "answered" and out["citations"][0]["verified"]
    assert seen["body"]["model"] == MODEL and seen["body"]["fallbacks"] == "default"
    assert FALLBACK_BETA in seen["headers"]["anthropic-beta"]
    documents = [block for block in seen["body"]["messages"][0]["content"] if block["type"] == "document"]
    assert [d["source"] for d in documents] == [{"type": "text", "media_type": "text/plain", "data": e["text"]}
                                                for e in EVIDENCE]
    assert all(d["citations"] == {"enabled": True} for d in documents)
    assert build_request("q", EVIDENCE)["system"] == seen["body"]["system"]


def test_claude_generator_reports_auth_errors():
    error = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    out = sdk_generator(401, error, {}).generate("q", EVIDENCE)
    assert out["status"] == "error" and "ANTHROPIC_API_KEY" in out["error"]
