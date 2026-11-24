"""Generate outcome-blind lexical duplicate candidates; never change cohorts.

Reads only pinned source queries/corpus and the experiment configuration, not
retrieval artifacts or metrics. Thresholds are review screens, not merge rules.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
for directory in (".vendor", ".vendor-hf"):
    vendor = ROOT / "project" / directory
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))
sys.path.insert(0, str(ROOT / "project" / "src"))
from ragstudy.data import load_techqa

OUTPUT = ROOT / "project" / "results" / "group_review"
SCREEN = {"full_char_cosine": .65, "full_word_cosine": .45,
          "de_template_word_cosine": .55}


def strip_template(text):
    text = re.sub(r"\s+", " ", text.casefold()).strip()
    for pattern in (r"\bhelp with\s*", r"\bsecurity bulletin\s*:?\s*",
                    r"\bi need to understand details regarding\s*",
                    r"\bwhere can i find (?:this|additional) information\s*\?",
                    r"\bwhere can i find information (?:on|for)\s*"):
        text = re.sub(pattern, "", text)
    return re.sub(r"^(?:how (?:do|can|should) (?:i|we)|how to|what (?:is|are)|"
                  r"why (?:is|are|does|do)|is (?:it|there)|can (?:i|we)|"
                  r"where (?:can|do) (?:i|we))\s+", "", text)


def write_high_risk_review():
    """Serialize the bounded reviewer judgments without new encoding or scoring."""
    candidate_path = OUTPUT / "candidate_pairs.json"
    pairs = json.loads(candidate_path.read_text(encoding="utf-8"))
    with (OUTPUT / "queries.csv").open(encoding="utf-8", newline="") as stream:
        queries = {query["query_id"]: query for query in csv.DictReader(stream)}
    target = [pair for pair in pairs if pair["cross_split"] and not pair["same_existing_group"]
              and pair["bge_cosine"] >= .90]
    ids = {name for pair in target for name in (pair["left_id"], pair["right_id"])}
    source = ROOT / "project/data/candidates/2026-11-12_technical_qa_audit/nvidia__TechQA-RAG-Eval/corpus.zip"
    screen = json.loads((OUTPUT / "screen_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != screen["source_audit"]["source_hashes"]["corpus.zip"]:
        raise ValueError("Source corpus differs from the screened corpus")
    cves = lambda text: sorted(set(re.findall(r"CVE-\d{4}-\d+", text, flags=re.I)))
    evidence = {}
    with zipfile.ZipFile(source) as archive:
        lookup = {Path(member).name: member for member in archive.namelist()}
        for query_id in sorted(ids):
            query = queries[query_id]
            docs = []
            for doc_id in json.loads(query["positive_document_ids"]):
                raw = archive.read(lookup[doc_id])
                text = raw.decode("utf-8", errors="replace")
                summary = re.search(r"SUMMARY\s+(.*?)VULNERABILITY DETAILS", text, flags=re.S)
                affected = re.search(r"AFFECTED PRODUCTS AND VERSIONS\s+(.*?)REMEDIATION/FIXES", text, flags=re.S)
                docs.append({"document_id": doc_id, "sha256": hashlib.sha256(raw).hexdigest(),
                             "title": text.splitlines()[0], "cves": cves(text),
                             "summary_excerpt": summary.group(1).strip() if summary else text[:1800],
                             "affected_products_excerpt": affected.group(1).strip()[:1800] if affected else None})
            evidence[query_id] = {"question": query["question"], "old_split": query["split"],
                                  "old_group": query["group_id"], "question_cves": cves(query["question"]),
                                  "gold_documents": docs}
    special = {
        frozenset(("TRAIN_Q312", "DEV_Q129")): ("different_affected_product", "Same January2015 Java CVEs, but ICCT upgrade/build fixes versus Java emergency fixes applied to deployed RedHat/AIX OS images; gold affected-products/remediation text distinguishes them."),
        frozenset(("TRAIN_Q386", "DEV_Q129")): ("different_affected_product", "Same January2015 Java CVEs, but ICCT versus deployed RedHat/AIX OS images; selecting CVE-2014-6593 does not erase the affected-product boundary."),
        frozenset(("DEV_Q187", "DEV_Q182")): ("different_affected_product", "WSRR/WAS July2017 Java CPU versus IIB/WMB July2017 Java fixes. CVE-2017-10115 overlaps, but packaged products, release ranges and remediation differ in gold originals."),
        frozenset(("DEV_Q187", "DEV_Q072")): ("different_affected_product_and_advisory_cycle", "WSRR/WAS July2017 CPU versus IIB/WMB April2017 CPU. The generic IIB question omits CVEs, but gold swg22005345 specifies 2017-3511/1289 and zlib CVEs, not the July2017 WAS advisory."),
        frozenset(("DEV_Q187", "TRAIN_Q321")): ("different_affected_product", "Shared Java CVEs, but WSRR/WAS July2017 versus API Connect V5 release-specific APAR LI79805. Same vulnerability family is not the same packaged-product support question."),
        frozenset(("TRAIN_Q493", "TRAIN_Q584")): ("unresolved_source_label_conflict", "Both questions name CVE-2017-1741 in different packaged-product contexts. TRAIN_Q584 gold swg22012345 is explicitly CVE-2017-1731/PI89498; TRAIN_Q493 gold swg22012342 is CVE-2017-1741/PI90042. Cannot close as an exact duplicate or clean distinct-issue pair without source-label correction. Keep current v2 unchanged."),
        frozenset(("TRAIN_Q493", "TRAIN_Q049")): ("unresolved_source_label_conflict", "TRAIN_Q049 title names CVE-2017-1731 but its body names CVE-2017-1741; its gold is 1731/PI89498. TRAIN_Q493 requests1741 with gold1741/PI90042. Conflicting source intent prevents a firm same-issue verdict; do not alter current v2."),
        frozenset(("TRAIN_Q450", "TRAIN_Q007")): ("same_issue_already_quarantined_in_v2", "Post-upgrade MQ client default-user propagation, JMS reason2035. Confirmed in prior targeted question/gold review; existing TRAIN_Q272 closure was already quarantined by root. No new v3 edge.")}
    target_keys = {frozenset((pair["left_id"], pair["right_id"])) for pair in target}
    ledger = []
    for pair in pairs:
        key = frozenset((pair["left_id"], pair["right_id"]))
        row = {name: pair[name] for name in ("left_id", "right_id", "left_split", "right_split",
                                           "left_group", "right_group", "cross_split", "same_existing_group",
                                           "full_char_cosine", "bge_cosine")}
        row.update(reviewed_this_round=key in target_keys, decision="not_reviewed_in_bounded_round",
                   reason="Outside the prespecified cross-old-split, different-component, BGE>=.90 scope.")
        if key in target_keys:
            a, b = cves(pair["left_text"]), cves(pair["right_text"])
            if key in special:
                row["decision"], row["reason"] = special[key]
            elif a and b and not set(a).intersection(b):
                row["decision"] = "different_explicit_requested_cves"
                row["reason"] = f"Question A explicitly requests {', '.join(a)}; question B requests {', '.join(b)}. Requested vulnerabilities differ, even when bulletin template/product/fix family overlaps. Gold title/summary/CVE inventory cross-checked."
            else:
                raise ValueError(f"Unadjudicated in-scope pair: {sorted(key)}")
            row["left_question_cves"], row["right_question_cves"] = a, b
            row["left_gold_ids"] = json.loads(queries[pair["left_id"]]["positive_document_ids"])
            row["right_gold_ids"] = json.loads(queries[pair["right_id"]]["positive_document_ids"])
        ledger.append(row)
    counts = {}
    for row in ledger:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    result = {"date": "2026-11-14", "scope_fixed_before_adjudication": "cross old split AND different existing component AND BGE cosine>=0.90",
              "old_split_manifest_sha256": "8094c45eeb5468678350fb283a5face880cccd1a3b8f82478373bcce0d6e8e35",
              "candidate_file_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
              "total_screened_pairs": len(pairs), "bounded_pairs_reviewed": len(target),
              "cross_split_candidates_total": sum(pair["cross_split"] for pair in pairs),
              "cross_split_candidates_outside_this_round": sum(pair["cross_split"] for pair in pairs) - len(target),
              "unique_queries_in_bounded_review": len(ids),
              "low_lexical_pairs_char_below_0_5": sum(pair["full_char_cosine"] < .5 for pair in target),
              "decision_counts": counts, "new_confirmed_same_issue_groups": [],
              "test_metrics_read": False, "test_unlock_authorized": False,
              "reviewer_scope": "Assistant semantic adjudication from question titles/explicit identities and gold title/summary/CVE inventories; ambiguous same-CVE cases additionally checked against full affected-product/remediation text. Human-verifiable evidence is retained; this is not a proof of zero semantic leakage.",
              "source_quality_warnings": [
                  {"query_id": "TRAIN_Q584", "warning": "Question requests CVE-2017-1741; gold swg22012345 names CVE-2017-1731 and PI89498, whereas swg22012342 names1741 and PI90042. Do not silently relabel or discard based on retrieval performance."},
                  {"query_id": "TRAIN_Q049", "warning": "Question title says1731, body says1741; gold is1731. Source intent is internally inconsistent."}],
              "pairs": ledger}
    (OUTPUT / "high_risk_pair_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "gold_evidence_high_risk.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: result[name] for name in ("bounded_pairs_reviewed", "unique_queries_in_bounded_review", "low_lexical_pairs_char_below_0_5", "decision_counts")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", action="store_true", help="Also screen frozen local BGE cosine >=.90 or per-query top two")
    parser.add_argument("--adjudicate-high-risk", action="store_true", help="Export bounded manual review decisions from existing screening files; no model encoding")
    args = parser.parse_args()
    if args.adjudicate_high_risk:
        write_high_risk_review()
        return
    config = json.loads((ROOT / "project/configs/techqa_exploratory.json").read_text(encoding="utf-8"))
    dataset = load_techqa(ROOT / config["source"], seed=config["seed"],
                          head_words=config["head_words"],
                          near_duplicate_threshold=config["near_duplicate_threshold"])
    queries = dataset["queries"]
    texts = [q["question"] for q in queries]
    matrices = []
    for analyzer, grams, documents in (("char_wb", (3, 5), texts),
                                        ("word", (1, 2), texts),
                                        ("word", (1, 2), [strip_template(t) for t in texts])):
        vectors = TfidfVectorizer(analyzer=analyzer, ngram_range=grams,
                                  stop_words="english" if analyzer == "word" else None,
                                  dtype=np.float64).fit_transform(documents)
        matrices.append((vectors @ vectors.T).toarray())
    mask = np.logical_or.reduce([matrix >= value for matrix, value in zip(matrices, SCREEN.values())])
    neural, nearest = None, None
    encoder_metadata = None
    if args.neural:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(8)
        model_path = ROOT / "project/models/bge-small-en-v1.5"
        model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
        model.max_seq_length = 512
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                                  normalize_embeddings=True, convert_to_numpy=True)
        neural = embeddings @ embeddings.T
        nearest = np.zeros_like(mask)
        for index, row in enumerate(neural):
            order = sorted((j for j in range(len(queries)) if j != index),
                           key=lambda j: (-float(row[j]), queries[j]["query_id"]))[:2]
            nearest[index, order] = True
        nearest = np.logical_or(nearest, nearest.T)
        mask |= (neural >= .90) | nearest
        encoder_metadata = {"model": "BAAI/bge-small-en-v1.5", "raw_query_text_no_prefix": True,
                            "revision": json.loads((model_path / "download_manifest.json").read_text(encoding="utf-8"))["revision"],
                            "max_seq_length": 512, "normalization": "l2", "cosine_threshold": .90,
                            "additional_screen": "per-query top two, symmetrized, deterministic query-ID tie break"}
    pairs = []
    for left, right in zip(*np.where(np.triu(mask, k=1))):
        a, b = queries[left], queries[right]
        pairs.append({"left_id": a["query_id"], "right_id": b["query_id"],
                      "left_split": a["split"], "right_split": b["split"],
                      "left_group": a["group_id"], "right_group": b["group_id"],
                      "same_existing_group": a["group_id"] == b["group_id"],
                      "cross_split": a["split"] != b["split"],
                      "shared_gold": sorted(set(a["positive_document_ids"]) & set(b["positive_document_ids"])),
                      **{name: float(matrix[left, right]) for name, matrix in zip(SCREEN, matrices)},
                      "bge_cosine": float(neural[left, right]) if neural is not None else None,
                      "neural_top_two": bool(nearest[left, right]) if nearest is not None else None,
                      "left_text": a["question"], "right_text": b["question"]})
    pairs.sort(key=lambda p: (not p["cross_split"], p["same_existing_group"],
                             -max([p[name] for name in SCREEN] + [p["bge_cosine"] or 0]),
                             p["left_id"], p["right_id"]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "queries.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("query_id", "split", "group_id", "positive_document_ids", "question"))
        writer.writeheader()
        for query in queries:
            writer.writerow({k: json.dumps(query[k], ensure_ascii=False) if isinstance(query[k], list) else query[k]
                             for k in writer.fieldnames})
    manifest = {"protocol": "outcome-blind-human-query-review-v1", "date": "2026-11-14",
                "scope": "All canonical queries; train/dev/test membership and query text only; no retrieval metrics read",
                "screening_thresholds_fixed_before_review": SCREEN,
                "thresholds_are_not_automatic_merge_rules": True,
                "neural_screening_fixed_before_encoding": encoder_metadata,
                "source_audit": dataset["audit"], "queries": len(queries),
                "pairs": len(pairs), "cross_split_pairs": sum(p["cross_split"] for p in pairs),
                "cross_split_different_gold_pairs": sum(p["cross_split"] and not p["shared_gold"] for p in pairs),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUTPUT / "candidate_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "screen_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("queries", "pairs", "cross_split_pairs")}, indent=2))
    print(f"Review files: {OUTPUT}")


if __name__ == "__main__":
    main()
