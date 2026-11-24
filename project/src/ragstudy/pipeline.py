"""Question -> retrieval -> evidence -> generated answer -> citations, with layer-by-layer failure notes.

Train/dev questions get gold-document diagnostics. Test questions never do, so the held-out split
stays frozen. Nothing here trains, tunes or reads test metrics; gold labels never reach a generator.
"""
from pathlib import Path
import csv
import json
import re
import time
import zipfile

from .generation import check_citations, digest
from .passages import select_source_passage

DEFAULT_METHOD = "weighted_rrf"
METHODS = ("bm25", "semantic", "rrf", "weighted_rrf", "logistic", "random_forest")
WINDOW_MODE = "120-word source window per document (frozen protocol)"
RANKING_DEPTH = 20


def normalize_question(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


class QuestionIndex:
    """Light metadata for diagnostics; loads no models and keeps no test labels."""

    def __init__(self, artifact, root):
        artifact, root = Path(artifact), Path(root)
        manifest = json.loads((artifact / "split_manifest.json").read_text(encoding="utf-8"))
        runs = {row["query_id"]: row for row in
                json.loads((artifact / "candidate_runs.json").read_text(encoding="utf-8"))}
        path = root / "project/results/iteration2_diagnostics/per_query_diagnostics.csv"
        locations = ({row["query_id"]: row["answer_location"] for row in csv.DictReader(path.open(encoding="utf-8"))}
                     if path.is_file() else {})
        self.queries, self.by_question = {}, {}
        for query in manifest:
            entry = {"query_id": query["query_id"], "split": query["split"], "question": query["question"]}
            if query["split"] in ("train", "dev"):
                entry.update(gold=list(query["positive_document_ids"]),
                             candidates=set(runs[query["query_id"]]["candidate_ids"]),
                             answer_location=locations.get(query["query_id"]))
            self.queries[query["query_id"]] = entry
            self.by_question.setdefault(normalize_question(query["question"]), []).append(entry)

    def lookup(self, question):
        """Known question entry; any test match wins so held-out labels are never shown."""
        matches = self.by_question.get(normalize_question(question), [])
        return next((m for m in matches if m["split"] == "test"), matches[0] if matches else None)

def retrieval_diagnostics(known, ranking, top_k):
    """`ranking` is the retrieved document-ID order (top RANKING_DEPTH)."""
    if known is None:
        return {"known_question": False}
    if known["split"] == "test":
        return {"known_question": True, "query_id": known["query_id"], "split": "test", "held_out": True}
    rank = next((position for position, doc in enumerate(ranking, 1) if doc in known["gold"]), None)
    return {"known_question": True, "query_id": known["query_id"], "split": known["split"], "held_out": False,
            "gold_document_ids": known["gold"],
            "gold_in_candidates": any(doc in known["candidates"] for doc in known["gold"]),
            "gold_rank": rank, "ranking_depth": len(ranking),
            "gold_in_evidence": rank is not None and rank <= top_k,
            "answer_location": known.get("answer_location")}


def window_evidence(question, retrieved, sources):
    evidence = []
    for index, entry in enumerate(retrieved["evidence"], 1):
        source = sources[entry["document_id"]]
        window = select_source_passage(question, source, max_words=120)
        assert source[window["start"]:window["end"]] == window["excerpt"]
        evidence.append({"label": f"[D{index}]", "document_id": entry["document_id"], "text": window["excerpt"],
                         "source_start": window["start"], "source_end": window["end"], "truncated": False})
    return evidence


def normalize_qwen(raw):
    """Map a FrozenGenerator.generate_two_stage output onto the shared result shape."""
    content, source_status = raw.get("raw_output"), raw["status"]
    labels = list(raw.get("citation_map", {}))
    empty = not (content or "").strip() or check_citations(content, labels)["citation_only_output"]
    if source_status == "abstained":
        status = "abstained"
    elif source_status == "not_generated_no_evidence":
        status = "no_evidence"
    elif empty or source_status == "failed_content_generation":
        status = "no_answer_content"
    elif source_status == "two_stage_citation_proposed_semantics_unverified":
        status = "answered"
    elif source_status.startswith("withheld"):
        status = "citation_withheld"
    else:
        status = "error"
    proposed = dict.fromkeys(re.findall(r"\[D\d+\]", raw.get("raw_citation_output") or ""))
    citations = [{"label": label, "document_id": raw["citation_map"].get(label), "cited_text": None,
                  "source_start": None, "source_end": None, "verified": None}
                 for label in proposed] if status == "answered" else []
    return {"status": status, "source_status": source_status,
            "answer": content if status in ("answered", "citation_withheld") else None,
            "raw_answer": content, "raw_citation_output": raw.get("raw_citation_output"), "citations": citations,
            "question_truncated": bool(raw.get("question_truncated")),
            "truncated_labels": [item["label"] for item in raw.get("evidence", []) if item.get("excerpt_truncated")],
            "skipped_evidence": raw.get("skipped_evidence") or []}


class QwenGenerator:
    evidence_mode = WINDOW_MODE

    def __init__(self, target, threads=2):
        from .generation import FrozenGenerator
        self.model = FrozenGenerator.from_local(target, threads=threads)
        self.name = self.model.metadata["model_id"]

    def prepare(self, question, retrieved, sources):
        return window_evidence(question, retrieved, sources)

    def generate(self, question, evidence):
        raw = self.model.generate_two_stage({"question": question, "evidence": [
            {"document_id": item["document_id"], "excerpt": item["text"]} for item in evidence]})
        return normalize_qwen(raw)

LOCATION_NOTES = {"tail": " Its answer sits after word 256 of the gold document, outside the indexed head.",
                  "crosses_boundary": " Its answer crosses the 256-word boundary of the indexed head."}


def _retrieval_notes(d, top_k, candidate_count):
    if not d.get("known_question"):
        return ["Free-text question: there is no gold label, so retrieval success cannot be checked."]
    if d.get("held_out"):
        return ["Held-out test question: gold diagnostics are hidden to keep the test split frozen."]
    gold = ", ".join(d["gold_document_ids"])
    if not d["gold_in_candidates"]:
        return [f"Retrieval miss: gold {gold} is not among the {candidate_count} candidates, so no reranker "
                f"or generator can recover it." + LOCATION_NOTES.get(d.get("answer_location"), "")]
    if d["gold_in_evidence"]:
        return [f"Gold {gold} was retrieved at rank {d['gold_rank']} and given to the generator."]
    where = f"at rank {d['gold_rank']}" if d["gold_rank"] else f"below rank {d['ranking_depth']}"
    return [f"Ranking miss: gold {gold} is in the candidate set but {where}, outside the top {top_k} "
            f"given to the generator."]


def _evidence_notes(result, generation):
    cut = [item["label"] for item in result["evidence"] if item.get("truncated")]
    notes = [f"{', '.join(cut)} cut to the first words of the document; later text was not seen."] if cut else []
    short = generation.get("truncated_labels", [])
    if short:
        notes.append(f"{', '.join(short)} shortened to fit the 512-token prompt; the answer may have been cut off.")
    if generation.get("question_truncated"):
        notes.append("The question was cut to 96 tokens; constraints at its end were not seen by the generator.")
    notes += [f"{item.get('document_id', 'an item')} was dropped ({item['reason']})."
              for item in generation.get("skipped_evidence", [])]
    return notes

STATUS_NOTES = {
    "not_run": "Generation skipped; only retrieval and evidence are shown.",
    "no_evidence": "Nothing was retrieved, so there was nothing to answer from.",
    "abstained": "The generator reported insufficient evidence.",
    "no_answer_content": "The generator returned no answer text (for example only a source label).",
    "citation_withheld": "The answer has no confirmed source: the separate citation step did not propose a valid label.",
    "refused": "The API declined the request.",
    "truncated": "The output hit the token limit and may be incomplete.",
}


def _generation_notes(d, generation):
    status = generation.get("status", "not_run")
    if status == "error":
        return [f"Generation failed: {generation.get('error', 'unknown error')}."]
    notes = [STATUS_NOTES[status]] if status in STATUS_NOTES else []
    known = d.get("known_question") and not d.get("held_out")
    if status == "abstained" and known:
        notes.append("The gold document was in the evidence, so this is an over-cautious abstention or the "
                     "evidence shown missed the answer." if d["gold_in_evidence"] else
                     "This matches the retrieval result: the gold document was not in the evidence.")
    if status == "refused" and generation.get("refusal_category"):
        notes[-1] += f" Category: {generation['refusal_category']}."
    if status != "answered":
        return notes
    citations = generation.get("citations", [])
    if not citations:
        notes.append("The answer carries no citation.")
    elif any(c["verified"] is None for c in citations):
        notes.append("Answer and citation come from two separate calls of the local model; the label is a "
                     "proposal, not proof of support.")
    else:
        verified = sum(c["verified"] for c in citations)
        notes.append(f"{verified} of {len(citations)} cited spans match the source text exactly.")
    if generation.get("uncited_characters"):
        notes.append(f"{generation['uncited_characters']} characters of the answer carry no citation.")
    if known and citations and not {c["document_id"] for c in citations} & set(d["gold_document_ids"]):
        notes.append("No cited document is the labelled gold; the answer may rely on a related technote or be wrong.")
    if generation.get("fallback_served"):
        notes.append(f"A fallback model served this answer ({generation.get('served_by')}).")
    notes.append("Correctness is not verified automatically; read the cited text.")
    return notes


def explain(result):
    generation = result.get("generation") or {"status": "not_run"}
    d = result["diagnostics"]
    return ([("retrieval", note) for note in _retrieval_notes(d, result["top_k"], result.get("candidate_count"))]
            + [("evidence", note) for note in _evidence_notes(result, generation)]
            + [("generation", note) for note in _generation_notes(d, generation)])

class RAGPipeline:
    """Loads one verified retrieval run once, then answers any number of questions."""

    def __init__(self, artifact, root, generator=None):
        from .search import load_bundle
        self.artifact, self.root, self.generator = Path(artifact), Path(root), generator
        config = json.loads((self.artifact / "configuration.json").read_text(encoding="utf-8"))
        audit = json.loads((self.artifact / "data_audit.json").read_text(encoding="utf-8"))
        corpus = self.root / config["source"] / "corpus.zip"
        if digest(corpus) != audit["source_hashes"]["corpus.zip"]:
            raise ValueError("Original evidence corpus differs from the verified retrieval source.")
        self.archive = zipfile.ZipFile(corpus)
        self.members = {Path(name).name: name for name in self.archive.namelist() if not name.endswith("/")}
        self.index = QuestionIndex(self.artifact, self.root)
        self.bundle = load_bundle(self.artifact)

    def source(self, document_id):
        return self.archive.read(self.members[document_id]).decode("utf-8", errors="replace")

    def run(self, question, method=DEFAULT_METHOD, top_k=3):
        from .search import search
        if method not in METHODS:
            raise ValueError("Unknown retrieval method: " + method)
        if not 1 <= top_k <= 5:
            raise ValueError("top_k must be between 1 and 5.")
        retrieved = search(self.bundle, question, method=method, limit=RANKING_DEPTH)
        ranking = [entry["document_id"] for entry in retrieved["evidence"]]
        shown = {**retrieved, "evidence": retrieved["evidence"][:top_k]}
        sources = {doc: self.source(doc) for doc in ranking[:top_k]}
        prepare = self.generator.prepare if self.generator else window_evidence
        result = {"question": question, "method": method, "top_k": top_k,
                  "candidate_count": retrieved["candidate_count"],
                  "ranking": [{"rank": rank, "document_id": entry["document_id"], "score": entry["ranking_score"]}
                              for rank, entry in enumerate(retrieved["evidence"][:max(top_k, 10)], 1)],
                  "diagnostics": retrieval_diagnostics(self.index.lookup(question), ranking, top_k),
                  "evidence": prepare(question, shown, sources),
                  "evidence_mode": self.generator.evidence_mode if self.generator else WINDOW_MODE,
                  "timing_ms": {"retrieval": retrieved["timing_ms"]["total"]},
                  "generation": {"status": "not_run"}}
        if self.generator is not None:
            started = time.perf_counter()
            result["generation"] = {"generator": self.generator.name,
                                    **self.generator.generate(question, result["evidence"])}
            result["timing_ms"]["generation"] = (time.perf_counter() - started) * 1000
        result["explanation"] = explain(result)
        return result
