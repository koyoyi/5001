"""Download a pinned local generator or run preserved smoke/development demos."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
for directory in (".vendor", ".vendor-hf", ".vendor-onnx-compat", ".vendor-onnx-dml"):
    if (ROOT / "project" / directory).is_dir():
        sys.path.insert(0, str(ROOT / "project" / directory))
sys.path.insert(0, str(ROOT / "project/src"))
from ragstudy.generation import MODEL_SPECS, FrozenGenerator, digest, download_model


def materialize_source_evidence(retrieved, archive):
    """Choose windows only inside the documents already returned by retrieval."""
    from ragstudy.passages import select_source_passage
    members = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
    evidence = []
    for entry in retrieved["evidence"]:
        doc_id = entry["document_id"]
        raw = archive.read(members[doc_id])
        source = raw.decode("utf-8", errors="replace")
        window = select_source_passage(retrieved["question"], source, max_words=120)
        assert source[window["start"]:window["end"]] == window["excerpt"]
        evidence.append({**entry, "retrieved_head_excerpt": entry["excerpt"],
                         "excerpt": window["excerpt"], "source_window": window,
                         "original_document_sha256": hashlib.sha256(raw).hexdigest()})
    assert [entry["document_id"] for entry in evidence] == [entry["document_id"] for entry in retrieved["evidence"]]
    return {**retrieved, "evidence": evidence,
            "evidence_window_scope": "unlabelled source windows inside retrieved document IDs only; ranking unchanged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download-model", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="One self-authored evidence case, no dataset queries")
    mode.add_argument("--artifact", type=Path, help="Completed retrieval run; only development queries are selected")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--generator", choices=tuple(MODEL_SPECS), default="qwen2.5-0.5b-instruct")
    parser.add_argument("--method", choices=("rrf", "weighted_rrf", "logistic", "random_forest"), default="logistic")
    args = parser.parse_args()
    target = ROOT / "project/models" / args.generator
    if args.download_model:
        download_model(target, args.generator)
        return
    if not 1 <= args.limit <= 10:
        parser.error("--limit must be between 1 and 10")
    if args.smoke:
        generator = FrozenGenerator.from_local(target, threads=args.threads)
        generate = generator.generate_two_stage if generator.metadata.get("family") == "causal" else generator.generate
        retrieved = {"question": "How long is the retention period for the example backup service?",
                     "evidence": [{"document_id": "self_authored_backup_policy",
                                   "excerpt": "The example backup service retains snapshots for seven days. "
                                              "After seven days, expired snapshots are deleted."}]}
        demos = [{"case_id": "self_authored_smoke", **generate(retrieved)}]
        output = ROOT / "project/results" / ("generation_smoke_" + args.generator + ".json")
        selection = "one self-authored policy; not a QA benchmark or selected dataset success"
    else:
        from ragstudy.search import load_bundle, search
        artifact = args.artifact.resolve()
        if artifact.parent != (ROOT / "project/artifacts").resolve():
            parser.error("--artifact must identify a completed run inside project/artifacts")
        config = json.loads((artifact / "configuration.json").read_text(encoding="utf-8"))
        if config.get("final_test_enabled", False):
            parser.error("Use an exploration artifact with the final-test lock enabled")
        verification = json.loads((artifact / "verification.json").read_text(encoding="utf-8"))
        if verification.get("status") != "passed" or verification.get("counts", {}).get("test_evaluated", False):
            parser.error("Retrieval source verification must have passed without final-test evaluation")
        manifest = json.loads((artifact / "split_manifest.json").read_text(encoding="utf-8"))
        dev = {query["query_id"]: {"question": query["question"]} for query in manifest if query["split"] == "dev"}
        rows = {row["query_id"]: row for row in json.loads((artifact / "per_query_metrics.json").read_text(encoding="utf-8"))
                if row["split"] == "dev"}
        misses = sorted(qid for qid in dev if rows[qid]["candidate_recall"] == 0)
        selected = misses[:1]
        selected.extend(qid for qid in sorted(dev) if qid not in selected)
        selected = selected[:args.limit]
        audit = json.loads((artifact / "data_audit.json").read_text(encoding="utf-8"))
        corpus = ROOT / config["source"] / "corpus.zip"
        if digest(corpus) != audit["source_hashes"]["corpus.zip"]:
            parser.error("Original evidence corpus differs from the verified retrieval source")
        generator = FrozenGenerator.from_local(target, threads=args.threads)
        generate = generator.generate_two_stage if generator.metadata.get("family") == "causal" else generator.generate
        bundle = load_bundle(artifact)
        demos = []
        with zipfile.ZipFile(corpus) as archive:
            for qid in selected:
                question = dev[qid]["question"]
                retrieved = search(bundle, question, method=args.method, limit=3)
                retrieved = materialize_source_evidence(retrieved, archive)
                result = generate(retrieved)
                demos.append({"query_id": qid, "split": "dev", "retrieval_method": args.method,
                              "retrieval_diagnostic_known_gold_in_candidate": bool(rows[qid]["candidate_recall"]),
                              "retrieval_diagnostic_scope": "case stratification only; not passed to generator",
                              "retrieval_result": retrieved, **result})
                print(f"Generation case {qid}: {result['status']}", flush=True)
        output = ROOT / "project/results" / artifact.name / "generation_demo.json"
        selection = "one lowest-ID development candidate miss if available, then lowest remaining development IDs; no success filtering"
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    report = {"created_at_utc": now.isoformat(), "generation_run_id": stamp, "selection": selection,
              "scope": "local frozen evidence-conditioned generation demonstration; no QA accuracy claim",
              "test_queries_used": 0, "llm_training_performed": False,
              "retrieval_artifact_run_id": args.artifact.name if args.artifact else None,
              "retrieval_verification_sha256": digest(args.artifact / "verification.json") if args.artifact else None,
              "checks_scope": "citation syntax, citation-only outputs, missing citations and missing evidence; semantic support requires human review",
              "cases": demos}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    history = (args.artifact / "generation_demo_history" if args.artifact
               else ROOT / "project/artifacts/probe/generation_smoke_history")
    history.mkdir(parents=True, exist_ok=True)
    (history / (stamp + ".json")).write_bytes(output.read_bytes())
    if args.artifact:
        (args.artifact / "generation_demo.json").write_bytes(output.read_bytes())
    print(json.dumps({"output": str(output), "statuses": [case["status"] for case in demos],
                      "raw_outputs": [case["raw_output"] for case in demos],
                      "raw_citation_outputs": [case.get("raw_citation_output") for case in demos]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
