"""Select one bounded, continuous evidence window from an original source.

This is deterministic evidence presentation, not retrieval, answer generation,
or a relevance judgment. It takes no QA-row answer or qrels. Returned offsets
always refer to the unchanged Python source string: ``source[start:end]``.
"""
import re


POLICY = "lexical_section_contiguous_window_v1"
_WORDS = re.compile(r"\S+")
_TERMS = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*|[\u3400-\u9fff]|[^\W\d_]+", re.UNICODE)
_STOP = frozenset("a an and are as at be been but by can could did do does for from has have how i if in into is it its may me my of on or our should that the their there these this to was we were what when where which why will with would you your".split())
_STRONG_ANSWER = frozenset(("ANSWER", "RESOLVING THE PROBLEM", "SOLUTION", "RESOLUTION"))
_QUESTION = frozenset(("QUESTION", "PROBLEM", "PROBLEM(ABSTRACT)"))
_METADATA = frozenset(("TITLE", "REFERENCES", "RELATED INFORMATION", "CHANGE HISTORY",
                       "GET NOTIFIED ABOUT FUTURE SECURITY BULLETINS"))
_HEADINGS = (*_STRONG_ANSWER, *_QUESTION, *_METADATA, "TEXT", "SYMPTOM", "CAUSE",
             "DIAGNOSING THE PROBLEM", "ENVIRONMENT", "SUMMARY", "VULNERABILITY DETAILS",
             "AFFECTED PRODUCTS AND VERSIONS", "REMEDIATION/FIXES", "WORKAROUNDS AND MITIGATIONS")
_HEADING_PATTERN = "|".join(re.escape(name).replace(r"\ ", r"[ \t]+")
                            for name in sorted(_HEADINGS, key=lambda item: (-len(item), item)))
_HEAD = re.compile(r"^[ \t]*(?P<label>" + _HEADING_PATTERN + r")(?:[ \t]*:[ \t]*|[ \t]*\r?$)",
                   re.MULTILINE | re.IGNORECASE)


def _terms(text):
    return {term for term in (match.group().casefold() for match in _TERMS.finditer(text))
            if term not in _STOP}


def _sections(source):
    matches = list(_HEAD.finditer(source))
    sections = []
    if not matches:
        return [("BODY", 0, len(source))]
    if source[:matches[0].start()].strip():
        sections.append(("BODY", 0, matches[0].start()))
    for index, match in enumerate(matches):
        label = re.sub(r"\s+", " ", match["label"]).upper()
        if label == "TEXT":
            label = "BODY"
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        if source[match.end():end].strip():
            sections.append((label, match.end(), end))
    return sections


def _windows(source, start, end, max_words, max_characters):
    words = list(_WORDS.finditer(source, start, end))
    if not words:
        return
    stride = max(1, max_words // 2)
    anchors = set(range(0, len(words), stride))
    anchors.add(max(0, len(words) - max_words))
    seen = set()
    for anchor in sorted(anchors):
        left = words[anchor].start()
        last = min(len(words), anchor + max_words)
        right = min(words[last - 1].end(), left + max_characters)
        if right < words[last - 1].end() and words[anchor].end() <= right:
            # Prefer an intact word boundary when a normal window hits its
            # character budget; a single overlong token must be sliced.
            right = max(word.end() for word in words[anchor:last] if word.end() <= right)
        if (left, right) not in seen:
            seen.add((left, right))
            yield left, right
    for word in words:
        if word.end() - word.start() <= max_characters:
            continue
        # A source token can be arbitrarily long. Character windows keep the
        # output bounded and still allow evidence late in that exact token.
        step = max(1, max_characters // 2)
        starts = list(range(word.start(), word.end(), step))
        starts.append(max(word.start(), word.end() - max_characters))
        for left in sorted(set(starts)):
            right = min(word.end(), left + max_characters)
            if (left, right) not in seen:
                seen.add((left, right))
                yield left, right


def select_source_passage(question, full_source_text, max_words=120):
    """Return a traceable excerpt selected using fixed lexical/section rules.

    Explicit ANSWER / RESOLVING THE PROBLEM / SOLUTION / RESOLUTION sections
    are preferred over QUESTION restatements and introductions. For a question
    beginning with "why", CAUSE can also compete. Otherwise overlapping fixed
    word windows are used outside question/reference metadata when possible.
    Coverage of non-stopword query terms determines the score; exact score ties
    use the earliest original offset. The score is a heuristic, not confidence.

    Word count uses whitespace-delimited words. A separate character ceiling
    ``min(6000, 40 * max_words)`` handles huge unbroken tokens. Such a token may
    be sliced, but the returned text is always a single unchanged substring.
    """
    if not isinstance(question, str) or not isinstance(full_source_text, str):
        raise TypeError("question and full_source_text must be strings")
    if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words < 1:
        raise ValueError("max_words must be a positive integer")
    source = full_source_text
    max_characters = min(6000, max_words * 40)
    title = re.search(r"^[ \t]*Title:[ \t]*([^\r\n]+)", source, re.IGNORECASE | re.MULTILINE)
    result = {"start": 0, "end": 0, "excerpt": "", "word_count": 0, "section": None,
              "matched_terms": [], "score": 0., "selection_basis": "empty_or_metadata_only",
              "policy": POLICY, "title": title.group(1).strip() if title else None,
              "max_words": max_words, "max_characters": max_characters}
    if not source.strip():
        return result
    sections = _sections(source)
    preferred = [section for section in sections if section[0] in _STRONG_ANSWER]
    wants_cause = re.match(r"^\s*why\b", question, re.IGNORECASE) is not None
    if preferred:
        pool = preferred + ([section for section in sections if section[0] == "CAUSE"] if wants_cause else [])
        basis = "answer_section_lexical_overlap"
    else:
        pool = [section for section in sections if section[0] not in _QUESTION | _METADATA]
        basis = "non_question_fixed_window"
        if not pool:
            pool = [section for section in sections if section[0] in _QUESTION]
            basis = "question_only_fallback"
    query_terms = _terms(question)
    best = None
    for label, start, end in pool:
        for left, right in _windows(source, start, end, max_words, max_characters):
            excerpt = source[left:right]
            matched = sorted(query_terms.intersection(_terms(excerpt)))
            coverage = len(matched) / max(1, len(query_terms))
            bonus = .20 if label in _STRONG_ANSWER else .25 if label == "CAUSE" and wants_cause else 0.
            score = coverage + bonus
            key = (-score, left, right)
            if best is None or key < best[0]:
                best = (key, label, left, right, excerpt, matched, score)
    if best is not None:
        _, label, left, right, excerpt, matched, score = best
        result.update(start=left, end=right, excerpt=excerpt, word_count=len(_WORDS.findall(excerpt)),
                      section=label, matched_terms=matched, score=score, selection_basis=basis)
    return result
