"""Pinned TechQA input audit and deterministic, label-isolated exploratory splits."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(text):
    return re.sub(r"\s+", " ", text.strip().casefold())


def assign_splits(queries, seed=20261114):
    """Split entire connected components; never use retrieval outcomes."""
    groups = sorted({query["group_id"] for query in queries})
    if len(groups) < 6:
        raise ValueError("At least six independent components are needed for train/dev/test.")
    order = np.random.RandomState(seed).permutation(len(groups))
    train_end = max(3, int(len(groups) * .70))
    dev_end = min(len(groups) - 1, max(train_end + 1, int(len(groups) * .85)))
    assignment = {groups[index]: ("train" if rank < train_end else "dev" if rank < dev_end else "test")
                  for rank, index in enumerate(order)}
    return [dict(query, split=assignment[query["group_id"]]) for query in queries]


def load_techqa(source, seed=20261114, head_words=256, near_duplicate_threshold=.90,
                document_policy="head", additional_query_groups=None, split_lock=None):
    source = Path(source)
    if not isinstance(head_words, int) or head_words < 1:
        raise ValueError("head_words must be a positive integer.")
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0,1].")
    if document_policy not in ("head", "full"):
        raise ValueError("document_policy must be head or full.")
    manifest_path = source.parent / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = {}
    for name in ("train.json", "corpus.zip"):
        matches = [entry for entry in manifest["files"]
                   if entry["local_file"].replace("\\", "/") in (name, source.name + "/" + name)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one pinned manifest entry for {source.name}/{name}.")
        entry = matches[0]
        inputs[name] = entry
        if sha256(source / name) != entry["sha256"]:
            raise ValueError(f"Pinned input SHA-256 mismatch: {name}")
    if inputs["train.json"]["revision"] != inputs["corpus.zip"]["revision"]:
        raise ValueError("Corpus and query files must use the same pinned revision.")
    if inputs["train.json"].get("repo") != inputs["corpus.zip"].get("repo"):
        raise ValueError("Corpus and query files must come from the same repository.")
    rows = json.loads((source / "train.json").read_text(encoding="utf-8"))
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Source record IDs must be unique.")
    originals, hash_groups = {}, defaultdict(list)
    with zipfile.ZipFile(source / "corpus.zip") as archive:
        for member in sorted(archive.infolist(), key=lambda member: member.filename):
            if member.is_dir():
                continue
            doc_id = Path(member.filename).name
            if doc_id in originals:
                raise ValueError("Duplicate archive basename: " + doc_id)
            text = archive.read(member).decode("utf-8", errors="replace")
            if not text.strip():
                raise ValueError("Empty corpus document: " + doc_id)
            originals[doc_id] = text
            hash_groups[hashlib.sha256(text.strip().encode("utf-8")).hexdigest()].append(doc_id)
    aliases = {doc: min(group) for group in hash_groups.values() for doc in group}
    document_ids = sorted(set(aliases.values()))
    # Preserve the policy in the audit; encoders separately record token limits.
    documents = [originals[doc] if document_policy == "full" else " ".join(originals[doc].split()[:head_words])
                 for doc in document_ids]
    grouped = defaultdict(list)
    for row in rows:
        grouped[normalize_question(row["question"])].append(row)
    queries, exclusions = [], []
    for key, group in sorted(grouped.items()):
        if len({row["is_impossible"] for row in group}) != 1:
            exclusions.append({"reason": "mixed_answerability", "ids": sorted(row["id"] for row in group)})
            continue
        if group[0]["is_impossible"]:
            continue
        positive_ids = set()
        for row in group:
            for context in row["contexts"]:
                doc_id = context["filename"]
                if doc_id not in originals:
                    raise ValueError(f"Missing reference document: {doc_id}")
                if originals[doc_id].strip() != context["text"].strip():
                    raise ValueError(f"Reference text mismatch: {row['id']} / {doc_id}")
                positive_ids.add(aliases[doc_id])
        if not positive_ids:
            continue
        representative = min(group, key=lambda row: row["id"])
        queries.append({"query_id": representative["id"], "question": representative["question"],
                        "source_record_ids": sorted(row["id"] for row in group),
                        "positive_document_ids": sorted(positive_ids)})
    parent = list(range(len(queries)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        parent[find(left)] = find(right)

    owner = {}
    for index, query in enumerate(queries):
        for doc in query["positive_document_ids"]:
            if doc in owner:
                union(index, owner[doc])
            else:
                owner[doc] = index
    character_vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                        dtype=np.float32).fit_transform([q["question"] for q in queries])
    similarity = (character_vectors @ character_vectors.T).toarray()
    near_pairs = []
    for left, right in zip(*np.where(np.triu(similarity >= near_duplicate_threshold, k=1))):
        union(int(left), int(right))
        near_pairs.append({"left": queries[left]["query_id"], "right": queries[right]["query_id"],
                           "char_cosine": float(similarity[left, right])})
    query_positions = {query["query_id"]: index for index, query in enumerate(queries)}
    for group in additional_query_groups or []:
        if len(group) < 2 or any(qid not in query_positions for qid in group):
            raise ValueError("Reviewed groups require at least two known canonical query IDs.")
        for qid in group[1:]:
            union(query_positions[group[0]], query_positions[qid])
    components = defaultdict(list)
    for index, query in enumerate(queries):
        components[find(index)].append(query["query_id"])
    group_ids = {index: min(ids) for index, ids in components.items()}
    for index, query in enumerate(queries):
        query["group_id"] = group_ids[find(index)]
    initial_query_count = len(queries)
    quarantined = []
    if split_lock is not None:
        if set(split_lock) != {query["query_id"] for query in queries}:
            raise ValueError("Split lock must cover exactly the original canonical query cohort.")
        component_splits = defaultdict(set)
        for query in queries:
            component_splits[query["group_id"]].add(split_lock[query["query_id"]])
        kept = []
        for query in queries:
            old_split = split_lock[query["query_id"]]
            if old_split not in ("train", "dev", "test"):
                raise ValueError("Unknown split in historical lock.")
            if len(component_splits[query["group_id"]]) > 1:
                quarantined.append({"query_id": query["query_id"], "previous_split": old_split,
                                    "group_id": query["group_id"], "reason": "reviewed_component_crosses_historical_splits"})
            else:
                kept.append(dict(query, split=old_split))
        queries = kept
    else:
        queries = assign_splits(queries, seed)
    split_audit = {split: {"queries": sum(q["split"] == split for q in queries),
                           "groups": len({q["group_id"] for q in queries if q["split"] == split})}
                   for split in ("train", "dev", "test")}
    positive_split = {}
    for query in queries:
        for doc in query["positive_document_ids"]:
            if doc in positive_split and positive_split[doc] != query["split"]:
                raise AssertionError("Shared gold document crossed split boundaries.")
            positive_split[doc] = query["split"]
    audit = {"dataset": "nvidia/TechQA-RAG-Eval", "revision": inputs["train.json"]["revision"],
             "protocol": "Custom exploratory split; not the official held-out TechQA benchmark",
             "raw_documents": len(originals), "unique_documents": len(document_ids),
             "source_records": len(rows), "canonical_answerable_queries": len(queries),
             "independent_components": len({query["group_id"] for query in queries}), "splits": split_audit,
             "canonical_queries_before_quarantine": initial_query_count,
             "quarantined_queries": quarantined,
             "split_policy": "historical_split_lock_with_component_quarantine" if split_lock is not None else "seeded_component_split",
             "seed": seed, "head_words": head_words, "near_duplicate_char_cosine": near_duplicate_threshold,
             "document_policy": document_policy, "additional_query_groups": additional_query_groups or [],
             "near_duplicate_pairs": near_pairs, "excluded_conflicts": exclusions,
             "duplicate_aliases": {doc: alias for doc, alias in aliases.items() if doc != alias},
             "semantic_duplicate_human_review": "pending; automated grouping is not exhaustive",
             "source_hashes": {name: sha256(source / name) for name in ("train.json", "corpus.zip")}}
    return {"document_ids": document_ids, "documents": documents, "originals": originals,
            "queries": queries, "audit": audit}
