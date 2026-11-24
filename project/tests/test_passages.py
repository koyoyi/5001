"""Small source-only fixtures for deterministic, bounded passage extraction."""

import inspect
import math
import re

import pytest

from ragstudy.passages import select_source_passage


REQUIRED = {"start", "end", "excerpt", "word_count", "section", "matched_terms", "score",
            "selection_basis", "policy", "title", "max_words", "max_characters"}


def assert_valid(result, source, max_words):
    assert REQUIRED <= result.keys()
    assert isinstance(result["start"], int) and not isinstance(result["start"], bool)
    assert isinstance(result["end"], int) and not isinstance(result["end"], bool)
    assert 0 <= result["start"] <= result["end"] <= len(source)
    assert result["excerpt"] == source[result["start"]:result["end"]]
    assert result["word_count"] == len(re.findall(r"\S+", result["excerpt"]))
    assert result["word_count"] <= max_words
    assert result["max_words"] == max_words
    assert result["max_characters"] == min(6000, max_words * 40)
    assert len(result["excerpt"]) <= result["max_characters"]
    assert isinstance(result["matched_terms"], list)
    assert all(isinstance(term, str) for term in result["matched_terms"])
    assert isinstance(result["score"], (int, float)) and math.isfinite(result["score"])
    assert isinstance(result["selection_basis"], str) and result["selection_basis"]
    assert isinstance(result["policy"], str) and result["policy"]
    return result


def test_exact_source_offsets_word_limit_and_determinism():
    source = ("Title: Widget maintenance\n\nBackground notes.\n\nANSWER\n"
              "Restart the amber widget service after clearing the stale lock. "
              "Preserve its diagnostic log before the restart.\n\nREFERENCES\nExternal links.")
    question = "How can I restart the amber widget service?"
    result = assert_valid(select_source_passage(question, source, max_words=12), source, 12)
    assert result == select_source_passage(question, source, max_words=12)
    assert "amber widget" in result["excerpt"]
    assert result["start"] > 0
    assert "ANSWER" in str(result["section"]).upper()
    assert result["matched_terms"]


@pytest.mark.parametrize("heading", ["ANSWER", "RESOLVING THE PROBLEM", "SOLUTION", "RESOLUTION"])
def test_explicit_solution_sections_beat_question_and_reference_repetition(heading):
    question = "How do I restart the blue widget service?"
    repeated = (question + " ") * 15
    source = (f"Title: {question}\n\nQUESTION\n{repeated}\n\n{heading}\n"
              "Stop the blue widget service, remove the stale lock, and start it again.\n\n"
              f"REFERENCES\n{repeated}")
    result = assert_valid(select_source_passage(question, source, max_words=25), source, 25)
    assert "remove the stale lock" in result["excerpt"]
    assert "How do I restart" not in result["excerpt"]
    assert not any(label in str(result["section"]).upper() for label in ("QUESTION", "TITLE", "REFERENCE"))


def test_why_question_can_select_cause_instead_of_repeated_question():
    question = "Why does the amber widget timeout?"
    source = ("Title: Amber widget notes\n\nQUESTION\n" + (question + " ") * 10
              + "\n\nCAUSE\nThe amber widget timeout occurs because a stale lock blocks the worker.\n\n"
                "REFERENCES\nGeneral vendor documentation.")
    result = assert_valid(select_source_passage(question, source, max_words=20), source, 20)
    assert "stale lock" in result["excerpt"]
    assert "CAUSE" in str(result["section"]).upper()


def test_plain_document_finds_query_words_in_later_window():
    source = ("ordinary unrelated background archive material " * 60
              + "Install the amber widget driver and restart the device. "
              + "ordinary unrelated background archive material " * 20)
    result = assert_valid(select_source_passage("amber widget driver", source, max_words=12), source, 12)
    assert result["start"] > 0
    assert "amber widget driver" in result["excerpt"]


def test_no_overlap_tie_uses_earliest_original_offset():
    source = "alpha beta gamma delta epsilon zeta eta theta"
    result = assert_valid(select_source_passage("unicorn quartz", source, max_words=3), source, 3)
    assert result["start"] == 0
    assert result["matched_terms"] == []


def test_empty_source_has_empty_exact_interval():
    result = assert_valid(select_source_passage("How can I repair a widget?", "", max_words=7), "", 7)
    assert (result["start"], result["end"], result["excerpt"], result["word_count"]) == (0, 0, "", 0)


@pytest.mark.parametrize("max_words", [1, 5, 200])
def test_very_long_unbroken_word_respects_character_bound(max_words):
    source = "x" * 12000
    result = assert_valid(select_source_passage("unrelated", source, max_words=max_words), source, max_words)
    assert result["start"] == 0
    assert result["excerpt"]
    assert result["word_count"] == 1


def test_unicode_and_crlf_offsets_stay_in_original_source():
    source = ("Title: 蓝色服务诊断 résumé\r\n\r\nQUESTION\r\n蓝色服务为什么超时？\r\n\r\n"
              "ANSWER\r\n蓝色服务修复步骤：先清理锁，再重新启动。\r\n保留诊断日志。\r\n\r\n"
              "REFERENCES\r\n外部说明。\r\n")
    result = assert_valid(select_source_passage("蓝色服务超时", source, max_words=8), source, 8)
    assert "先清理锁" in result["excerpt"]
    assert result["start"] > source.index("ANSWER")
    assert result == select_source_passage("蓝色服务超时", source, max_words=8)


@pytest.mark.parametrize("max_words", [0, -1, True, False, 1.5, "12", None])
def test_invalid_word_limits_are_rejected(max_words):
    with pytest.raises((TypeError, ValueError)):
        select_source_passage("widget", "widget source text", max_words=max_words)


@pytest.mark.parametrize("question,source", [(None, "text"), (23, "text"), (["widget"], "text"),
                                          ("widget", None), ("widget", 23), ("widget", ["text"])])
def test_question_and_full_source_must_be_strings(question, source):
    with pytest.raises(TypeError):
        select_source_passage(question, source)


def test_public_interface_accepts_no_qrels_or_reference_answer():
    assert list(inspect.signature(select_source_passage).parameters) == [
        "question", "full_source_text", "max_words"]
