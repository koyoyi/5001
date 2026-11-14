from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re
import statistics
import zipfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "nvidia__TechQA-RAG-Eval"
normalize = lambda text: re.sub(r"\s+", " ", text.strip().casefold())

def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

rows = json.loads((DATA / "train.json").read_text(encoding="utf-8"))
groups = defaultdict(list)
for row in rows:
    groups[normalize(row["question"])].append(row)
conflicts = {key: group for key, group in groups.items()
             if len({row["is_impossible"] for row in group}) > 1}
cross_split = [group for group in groups.values()
               if len({row["id"].split("_Q")[0] for row in group}) > 1]

lengths = []
hash_groups = defaultdict(list)
by_base = {}
missing = []
content_matches = 0
content_mismatches = []
empty_docs = 0
decode_replacements = 0
with zipfile.ZipFile(DATA / "corpus.zip") as archive:
    members = [member for member in archive.infolist() if not member.is_dir()]
    for member in members:
        basename = Path(member.filename).name
        if basename in by_base:
            raise RuntimeError("Duplicate archive basename: " + basename)
        by_base[basename] = member.filename
        raw = archive.read(member.filename)
        text = raw.decode("utf-8", errors="replace")
        lengths.append(len(text))
        empty_docs += not bool(text.strip())
        decode_replacements += "\ufffd" in text
        hash_groups[hashlib.sha256(text.strip().encode("utf-8")).hexdigest()].append(basename)
    for row in rows:
        for context in row["contexts"]:
            filename = context["filename"]
            if filename not in by_base:
                missing.append({"id": row["id"], "filename": filename})
                continue
            original = archive.read(by_base[filename]).decode("utf-8", errors="replace")
            if original.strip() == context["text"].strip():
                content_matches += 1
            else:
                content_mismatches.append({"id": row["id"], "filename": filename})
    uncompressed_bytes = sum(member.file_size for member in members)

canonical = []
for key, group in groups.items():
    if key in conflicts or group[0]["is_impossible"]:
        continue
    docs = sorted({context["filename"] for row in group for context in row["contexts"]})
    if not docs or any(doc not in by_base for doc in docs):
        continue
    canonical.append({
        "canonical_query_id": min(row["id"] for row in group),
        "source_record_ids": sorted(row["id"] for row in group),
        "normalized_question_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
        "positive_document_ids": docs
    })

parent = list(range(len(canonical)))
def find(index):
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index
def union(left, right):
    parent[find(left)] = find(right)
owner = {}
for index, query in enumerate(canonical):
    for doc in query["positive_document_ids"]:
        if doc in owner:
            union(index, owner[doc])
        else:
            owner[doc] = index
components = defaultdict(list)
for index, query in enumerate(canonical):
    components[find(index)].append(query["canonical_query_id"])
sorted_lengths = sorted(lengths)
percentile = lambda p: sorted_lengths[round((len(sorted_lengths)-1)*p)]
duplicate_doc_groups = [names for names in hash_groups.values() if len(names) > 1]

audit = {
    "audit_date": "2026-11-12",
    "dataset": "nvidia/TechQA-RAG-Eval",
    "revision": "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a",
    "source_card": "https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval",
    "verification": "Downloaded full JSON and ZIP; LFS SHA-256 checked in download_manifest.json.",
    "corpus": {
        "documents": len(by_base),
        "uncompressed_bytes": uncompressed_bytes,
        "empty_text_documents": empty_docs,
        "documents_with_utf8_replacement_characters": decode_replacements,
        "exact_text_duplicate_groups": len(duplicate_doc_groups),
        "documents_in_exact_text_duplicate_groups": sum(map(len, duplicate_doc_groups)),
        "character_length": {"min": min(lengths), "median": statistics.median(lengths),
                             "p90": percentile(.9), "p95": percentile(.95),
                             "p99": percentile(.99), "max": max(lengths)}
    },
    "records": {
        "total": len(rows),
        "unique_record_ids": len({row["id"] for row in rows}),
        "original_id_prefix_counts": dict(Counter(row["id"].split("_Q")[0] for row in rows)),
        "answerable_records": sum(not row["is_impossible"] for row in rows),
        "unanswerable_records": sum(row["is_impossible"] for row in rows),
        "normalized_unique_questions": len(groups),
        "exact_duplicate_question_groups": sum(len(group) > 1 for group in groups.values()),
        "cross_original_split_duplicate_groups": len(cross_split),
        "mixed_answerability_question_groups": len(conflicts),
        "reference_context_content_matches": content_matches,
        "missing_reference_documents": missing,
        "reference_content_mismatches": content_mismatches
    },
    "conservative_ranking_cohort_proposal": {
        "status": "Candidate audit only; not a frozen experimental split.",
        "rule": "Exclude mixed-answerability normalized-question groups pending review; merge exact duplicate answerable questions and union their gold document IDs.",
        "canonical_answerable_queries": len(canonical),
        "positive_query_document_pairs": sum(len(query["positive_document_ids"]) for query in canonical),
        "unique_gold_documents": len({doc for query in canonical for doc in query["positive_document_ids"]}),
        "shared_gold_document_components": len(components),
        "max_queries_per_component": max(map(len, components.values())),
        "remaining_checks": ["Semantic near-duplicate questions", "Exact corpus duplicate aliases", "Long-document retrieval policy", "Human sample review"]
    },
    "interpretation": [
        "The source card says 908 QA pairs; the pinned JSON actually contains 910 records.",
        "The Hugging Face split named train contains original TRAIN and DEV IDs; it is not a ready-made new train/dev/test protocol.",
        "Original TechQA unanswerability is relative to its candidate setting and cannot be assumed to mean no evidence exists anywhere in this corpus.",
        "Unjudged corpus documents are not confirmed negatives.",
        "Questions, answers, contexts links and labels must not be indexed as additional corpus text.",
        "This is a subset/archive study, not the full original TechQA 800k-document search task.",
        "No retrieval, model training, latency benchmarking, or performance evaluation was run."
    ]
}
dump("techqa_audit.json", audit)
gold_names = {context["filename"] for row in rows for context in row["contexts"]}
dump("duplicate_document_aliases.json", {
    "groups": [{"filenames": names, "gold_filenames": sorted(set(names) & gold_names)} for names in duplicate_doc_groups],
    "note": "Review exact-text aliases before freezing retrieval IDs and group assignments."
})
dump("canonical_query_groups_proposed.json", {
    "status": "Proposed cohort, not adopted/frozen; semantic duplicate review pending.",
    "queries": canonical,
    "components": [{"query_ids": sorted(ids)} for ids in components.values()],
    "mixed_answerability_groups_pending_review": [
        {"record_ids": [row["id"] for row in group],
         "labels": [row["is_impossible"] for row in group]} for group in conflicts.values()],
    "cross_original_split_duplicate_groups": [[row["id"] for row in group] for group in cross_split]
})

plan = {
    "status": "Recommended plan after candidate data audit; original submitted proposal and active experiment config unchanged.",
    "date": "2026-11-12",
    "recommended_dataset": "NVIDIA TechQA-RAG-Eval pinned snapshot",
    "submitted_title_preserved": "Can Learned Reranking Improve Hybrid Retrieval for RAG?",
    "application_subtitle": "Software technical-support evidence retrieval",
    "user": "Software support engineers searching a fixed archive of IBM product documentation.",
    "task": "Given a technical question or error description, retrieve relevant Technotes and show supporting source text.",
    "preserved_from_submitted_proposal": [
        "Three research questions about retrieval complementarity, learned reranking, and quality/latency trade-offs",
        "BM25, frozen dense retrieval, RRF, L2 logistic regression, constrained random forest",
        "Identical candidate union for RRF and learned rerankers",
        "nDCG@10, Recall@20, candidate recall, median/p95 online latency",
        "Fold-local preprocessing, group-aware validation, frozen final test, fixed-model citation demo"
    ],
    "changes_required": [
        "Data section: replace FiQA corpus/query/qrels with this fixed technical-support archive.",
        "Document-level qrels: any chunking must map back to original Technote IDs before evaluation.",
        "Define one fixed long-document segmentation/aggregation policy shared by every method.",
        "Custom group-disjoint train/dev/test; do not report this as official TechQA held-out benchmark.",
        "Treat answerability as a separate audit, not a reliable global no-answer gold label.",
        "Update sources, dataset counts, limitations and implementation dates."
    ],
    "split_proposal": {
        "status": "Not frozen",
        "group_unit": "Connected component of duplicate/near-duplicate cases and shared gold document IDs",
        "approximate_group_ratios": {"train": .7, "dev": .15, "test": .15},
        "seed": 20261112,
        "cv": "Three-fold grouped CV on training groups only",
        "uncertainty": "Paired resampling by independent case/document component; report actual group/query counts.",
        "shared_corpus": "Same archive can be indexed by all splits; no QA answers or test labels in features."
    },
    "stages": [
        "Review sample cases, duplicate document aliases and long-document policy; freeze cohort and group splits.",
        "Build BM25/dense/RRF baselines and ID-aligned candidate runs.",
        "Fit logistic/RF rerankers using training evidence links and explicitly described proxy negatives.",
        "Select on development data, freeze pipeline, run held-out evaluation and error/latency analysis.",
        "Build a small citation-search demonstration and course presentation/code appendix."
    ],
    "risks": [
        "Around 581 canonical answerable queries before remaining review: suitable for a small exploratory reranking study, not a large-scale generalisation claim.",
        "Historic IBM support documents do not directly evaluate current vLLM, CUDA, GPU troubleshooting or incident resolution.",
        "Each retained query currently has one labelled positive document; Recall@20 is known-gold hit@20 here and does not measure exhaustive evidence recall.",
        "The number of training positives actually retrieved by the shared candidate union is not known until retrieval is implemented.",
        "No promised improvement over RRF; negative results remain valid."
    ],
    "other_candidates": [
        {"name": "RAGBench techqa", "verification": "Pinned test parquet downloaded and inspected",
         "test_rows": 314, "test_unique_normalized_questions": 157,
         "reason_not_primary": "Multiple generated answers per question and model-derived sentence labels; same TechQA source, not independent validation."},
        {"name": "FreshStack", "verification": "Official cards and API samples checked by collaborating agent",
         "langchain_test_queries": 203, "all_five_domains_test_queries": 672,
         "reason_not_primary": "Official query sets are test-only; relevance labels use model-assisted nugget annotation. Custom supervised splitting changes protocol."},
        {"name": "IBM watsonxDocsQA", "verification": "Official card and train/test API samples checked by collaborating agents",
         "corpus_documents": 1144, "train_queries": 45, "test_queries": 30,
         "reason_not_primary": "Small independent-query count; useful as a supplementary technical-document evaluation domain."},
        {"name": "PrimeQA original TechQA archive", "verification": "HF metadata and partial HTTP range verified; not downloaded",
         "archive_bytes": 2959973525,
         "reason_not_primary": "Much heavier corpus; no additional independent query source compared with the small derivative."}
    ]
}
dump("recommended_experiment_plan.json", plan)
print(json.dumps({
    "audit_file": str(ROOT / "techqa_audit.json"),
    "plan_file": str(ROOT / "recommended_experiment_plan.json"),
    "documents": len(by_base),
    "canonical_queries_proposed": len(canonical),
    "independent_document_components": len(components),
    "empty_documents": empty_docs,
    "exact_duplicate_document_groups": len(duplicate_doc_groups),
    "doc_char_lengths": audit["corpus"]["character_length"],
    "performance_experiments_run": False
}, ensure_ascii=True, indent=2))
