"""Command-line RAG demo: question -> retrieval -> evidence -> answer -> citations -> diagnosis.

Examples (run from the 5001 repository root):
  python project/scripts/rag_demo.py --list-examples
  python project/scripts/rag_demo.py --example DEV_Q014
  python project/scripts/rag_demo.py --question "How do I ...?" --generator claude
  python project/scripts/rag_demo.py --replay            # saved failure cases, no model loaded
  python project/scripts/rag_demo.py                     # interactive prompt
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
for directory in (".vendor-hf", ".vendor-onnx-compat", ".vendor-onnx-dml"):
    if (ROOT / "project" / directory).is_dir():
        sys.path.insert(0, str(ROOT / "project" / directory))
sys.path.insert(0, str(ROOT / "project/src"))
from ragstudy.claude_generation import MODEL as CLAUDE_MODEL
from ragstudy.pipeline import (DEFAULT_METHOD, METHODS, QuestionIndex, explain, normalize_qwen,
                               retrieval_diagnostics)

ARTIFACT = ROOT / "project/artifacts/20261114T144053296846Z_onnx"
QWEN = ROOT / "project/models/qwen2.5-0.5b-instruct"
WIDTH = 100


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--question", help="Free-text technical question")
    mode.add_argument("--example", metavar="QUERY_ID", help="A train or dev question ID, shown with gold diagnostics")
    mode.add_argument("--list-examples", action="store_true", help="List dev questions by retrieval outcome")
    mode.add_argument("--replay", action="store_true", help="Explain the saved five-case Qwen demo without models")
    parser.add_argument("--generator", choices=("qwen", "claude", "none"), default="qwen",
                        help="qwen: frozen local Qwen2.5-0.5B; claude: Claude API (sends text off this machine)")
    parser.add_argument("--claude-model", default=CLAUDE_MODEL, help="Model ID for --generator claude")
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    parser.add_argument("--top-k", type=int, default=3, help="Documents given to the generator (1-5)")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for the local generator")
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--save", action="store_true", help="Also save each result under <artifact>/rag_demo_history/")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    return parser.parse_args()

def wrap(text, indent="    "):
    return textwrap.fill(" ".join(text.split()), WIDTH, initial_indent=indent, subsequent_indent=indent)


def render(result, preview_chars=700, question_chars=500):
    d, generation = result["diagnostics"], result.get("generation") or {}
    gold = set(d.get("gold_document_ids", []))
    known = f"{d['query_id']} ({d['split']})" if d.get("known_question") else "free text"
    question = " ".join(result["question"].split())
    if len(question) > question_chars:
        question = question[:question_chars] + f" ... [{len(question) - question_chars} more characters]"
    out = ["", "=" * WIDTH, "QUESTION  " + known, wrap(question)]
    out.append(f"\nRETRIEVAL  {result['method']}, {result['candidate_count']} candidates, "
               f"top {result['top_k']} go to the generator   (* = labelled gold)")
    for row in result["ranking"]:
        mark = "->" if row["rank"] <= result["top_k"] else "  "
        star = "*" if row["document_id"] in gold else " "
        out.append(f"  {mark} {row['rank']:>2}. {star} {row['document_id']:<22} score {row['score']:.4f}")
    out.append(f"\nEVIDENCE  {result['evidence_mode']}")
    for item in result["evidence"]:
        text = item["text"] if len(item["text"]) <= preview_chars else (
            item["text"][:preview_chars] + f" ... [{len(item['text']) - preview_chars} more characters]")
        out += [f"  {item['label']} {item['document_id']}  source chars {item['source_start']}-{item['source_end']}",
                wrap(text)]
    out.append(f"\nANSWER  {generation.get('generator', 'none')}   status: {generation.get('status')}")
    shown = generation.get("answer") or generation.get("raw_answer")
    out.append(wrap(shown) if shown else "    (no answer text)")
    if generation.get("citations"):
        out.append("\nCITATIONS")
        for c in generation["citations"]:
            out.append(f"  {c['label']} -> {c['document_id']}")
            if c["cited_text"] is not None:
                check = "matches source" if c["verified"] else "DOES NOT match source"
                out += [f"     chars {c['source_start']}-{c['source_end']}, {check}:", wrap('"' + c["cited_text"] + '"', "     ")]
    out.append("\nDIAGNOSIS")
    out += [wrap(f"[{layer}] {note}", "  ") for layer, note in result["explanation"]]
    timing = ", ".join(f"{key} {value / 1000:.1f} s" for key, value in result.get("timing_ms", {}).items())
    if timing:
        out.append("\nTIME  " + timing)
    return "\n".join(out)

def stored_rows(artifact):
    """Saved train/dev rankings of the verified run (the run never scored test questions)."""
    rows = json.loads((artifact / "per_query_metrics.json").read_text(encoding="utf-8"))
    return {row["query_id"]: row for row in rows if row["split"] in ("train", "dev")}


def list_examples(index, artifact, method, top_k, per_group=3):
    groups = {"retrieval miss (gold not in candidates)": [], f"ranking miss (gold below top {top_k})": [],
              f"hit (gold in top {top_k})": []}
    names = list(groups)
    for qid, row in sorted(stored_rows(artifact).items()):
        if row["split"] != "dev":
            continue
        d = retrieval_diagnostics(index.queries[qid], row["rankings"][method], top_k)
        group = names[0] if not d["gold_in_candidates"] else names[2] if d["gold_in_evidence"] else names[1]
        groups[group].append(qid)
    print(f"Dev questions by outcome for {method}, top {top_k} (saved rankings of the verified run):")
    for name, ids in groups.items():
        print(f"\n{name}: {len(ids)} questions")
        for qid in ids[:per_group]:
            print(f"  {qid:<11} {' '.join(index.queries[qid]['question'].split())[:84]}")
    print("\nRun one with: python project/scripts/rag_demo.py --example <QUERY_ID>")


def replay(index, artifact):
    """Re-explain the saved Qwen demo cases from their stored outputs; no model or index is loaded."""
    path = ROOT / "project/results" / artifact.name / "generation_demo.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    rows = stored_rows(artifact)
    print(f"Replaying {len(saved['cases'])} saved cases from {path.relative_to(ROOT)} "
          f"(generation run {saved['generation_run_id']}); outputs are shown exactly as stored.")
    for case in saved["cases"]:
        retrieved = case["retrieval_result"]
        method, top_k = case["retrieval_method"], len(retrieved["evidence"])
        ranking = rows[case["query_id"]]["rankings"][method]
        result = {"question": retrieved["question"], "method": method, "top_k": top_k,
                  "candidate_count": retrieved["candidate_count"],
                  "ranking": [{"rank": rank, "document_id": entry["document_id"], "score": entry["ranking_score"]}
                              for rank, entry in enumerate(retrieved["evidence"], 1)],
                  "diagnostics": retrieval_diagnostics(index.queries[case["query_id"]], ranking, top_k),
                  "evidence": [{"label": f"[D{n}]", "document_id": e["document_id"], "text": e["excerpt"],
                                "source_start": e["source_window"]["start"], "source_end": e["source_window"]["end"],
                                "truncated": False} for n, e in enumerate(retrieved["evidence"], 1)],
                  "evidence_mode": "120-word source window per document (frozen protocol)",
                  "generation": {"generator": case["model"]["model_id"], **normalize_qwen(case)}}
        result["explanation"] = explain(result)
        print(render(result))

def build_generator(args):
    if args.generator == "none":
        return None
    if args.generator == "claude":
        # The SDK and its transport live in an ignored project folder, loaded only for this option.
        vendor = ROOT / "project/.vendor-claude"
        if vendor.is_dir():
            sys.path.insert(0, str(vendor))
        try:
            from ragstudy.claude_generation import ClaudeGenerator
            generator = ClaudeGenerator(model=args.claude_model)
        except ImportError:
            sys.exit("The Claude option needs the anthropic SDK: see project/requirements-claude.txt.")
        print(f"Claude generator: {generator.name} via {generator.endpoint}. Each question and its retrieved "
              f"documents are sent to that endpoint.", flush=True)
        return generator
    from ragstudy.pipeline import QwenGenerator
    print("Loading frozen Qwen2.5-0.5B on CPU ...", flush=True)
    return QwenGenerator(QWEN, threads=args.threads)


def answer(pipeline, question, args):
    result = pipeline.run(question, method=args.method, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else render(result), flush=True)
    if args.save:
        folder = args.artifact / "rag_demo_history"
        folder.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (folder / f"{stamp}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {folder / (stamp + '.json')}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    args.artifact = args.artifact.resolve()
    verification = json.loads((args.artifact / "verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "passed":
        sys.exit("The retrieval run has not passed verification.")
    index = QuestionIndex(args.artifact, ROOT)
    if args.list_examples:
        return list_examples(index, args.artifact, args.method, args.top_k)
    if args.replay:
        return replay(index, args.artifact)
    question = args.question
    if args.example:
        entry = index.queries.get(args.example)
        if entry is None or entry["split"] == "test":
            sys.exit("Use a train or dev question ID; test questions stay frozen.")
        question = entry["question"]
    from ragstudy.pipeline import RAGPipeline
    generator = build_generator(args)
    print("Loading the verified retrieval run ...", flush=True)
    pipeline = RAGPipeline(args.artifact, ROOT, generator=generator)
    if question:
        return answer(pipeline, question, args)
    print("Type a technical question (empty line to quit).")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            break
        if not question:
            break
        answer(pipeline, question, args)


if __name__ == "__main__":
    main()
