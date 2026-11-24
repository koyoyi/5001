"""Build report figures and model-interpretation JSON from the frozen main run.

Reads the deployed logistic/RF models and candidate features from the main
neural artifact (20261114T144053296846Z_onnx); does not refit or retrain and
never reads held-out test metrics. Figures are written to report/assets/.
"""
from pathlib import Path
import json
import sqlite3

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]          # project/
ART = ROOT / "artifacts" / "20261114T144053296846Z_onnx"
OUT = ROOT.parent / "report" / "assets"             # 5001/report/assets
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, RED, GRAY, INK, AXIS = "#2a78d6", "#eb6834", "#e34948", "#898781", "#0b0b0b", "#c3c2b7"
FEATURES = ["bm25_score", "semantic_score", "bm25_recip_rank", "semantic_recip_rank",
            "missing_bm25", "missing_semantic", "term_coverage", "log_query_len", "log_doc_len"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load_models():
    import joblib
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    bundle = joblib.load(ART / "search_bundle.joblib")
    lg, rf = bundle["models"]["logistic"], bundle["models"]["random_forest"]
    return lg, rf, bundle


def dev_feature_matrix():
    con = sqlite3.connect(ART / "experiments.sqlite")
    rows = con.execute("SELECT c.features FROM candidates c JOIN queries q "
                       "ON c.query_id=q.query_id WHERE q.split='dev'").fetchall()
    con.close()
    return np.asarray([json.loads(r[0]) for r in rows], dtype=np.float64)

def hbar(values, labels, colors, title, xlabel, fname, fmt="{:+.2f}"):
    order = np.argsort(np.abs(values))
    v, lab = np.asarray(values)[order], [labels[i] for i in order]
    col = [colors[i] for i in order] if isinstance(colors, list) else colors
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.barh(range(len(v)), v, color=col, height=0.66, zorder=3)
    ax.set_yticks(range(len(v)), lab, fontsize=10.5)
    ax.axvline(0, color=AXIS, lw=0.8)
    for i, val in enumerate(v):
        ax.text(val + (0.01 if val >= 0 else -0.01) * max(abs(v)) * 2, i, fmt.format(val),
                va="center", ha="left" if val >= 0 else "right", fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=12.5, color=INK, pad=8, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=GRAY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.22)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_logistic(lg):
    coef = lg["model"].coef_[0]
    colors = [BLUE if c >= 0 else RED for c in coef]
    hbar(coef, FEATURES, colors,
         "Logistic reranker — standardised coefficients (C=0.1)",
         "coefficient per standard deviation  (blue +, red -)",
         "fig_logistic_coef.png")
    return coef


def fig_rf_importance(rf):
    imp = rf["model"].feature_importances_
    hbar(imp, FEATURES, BLUE,
         "Random forest — impurity feature importance",
         "share of total importance", "fig_rf_importance.png", fmt="{:.2f}")
    return imp

def fig_dev_methods():
    summ = json.loads((ART / "results.json").read_text(encoding="utf-8"))
    methods = summ["summary"]["dev"]["methods"]
    labels = ["BM25", "Semantic", "RRF", "Weighted\nRRF", "Logistic", "Random\nforest"]
    keys = ["bm25", "semantic", "rrf", "weighted_rrf", "logistic", "random_forest"]
    vals = [methods[k]["ndcg@10"] for k in keys]
    learned = {"weighted_rrf", "logistic", "random_forest"}
    colors = [BLUE if k in learned else GRAY for k in keys]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(range(len(vals)), vals, color=colors, width=0.68, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(vals)), labels, fontsize=10)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("nDCG@10 (dev)", fontsize=10.5, color=GRAY)
    ax.set_title("BGE head pipeline — development nDCG@10 by method",
                 fontsize=12.5, color=INK, pad=8, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=GRAY)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dev_methods.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_tail_recall():
    diag = json.loads((ROOT / "results" / "iteration2_diagnostics" / "summary.json").read_text(encoding="utf-8"))
    reg = diag["summary"]["train"]["answer_region_subgroups"]
    groups = [("head", "Answer in head\n(n=323)"), ("tail", "Answer in tail\n(n=63)"),
              ("crosses_boundary", "Crosses 256-word\nboundary (n=25)")]
    head = [reg[k]["head_bm25_recall100"] for k, _ in groups]
    full = [reg[k]["full_bm25_recall100"] for k, _ in groups]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(x - 0.2, head, 0.38, color=BLUE, label="Head index (256 words)", zorder=3)
    ax.bar(x + 0.2, full, 0.38, color=ORANGE, label="Full-text index", zorder=3)
    for xi, (h, f) in enumerate(zip(head, full)):
        ax.text(xi - 0.2, h + 0.012, f"{h:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(xi + 0.2, f + 0.012, f"{f:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [g[1] for g in groups], fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("BM25 recall@100 (train)", fontsize=10.5, color=GRAY)
    ax.set_title("BM25 on a 256-word head misses tail answers", fontsize=12.5, color=INK, pad=8, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=GRAY)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_tail_recall.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def fig_rf_pdp(rf, X):
    from sklearn.inspection import PartialDependenceDisplay
    targets = [3, 2]  # semantic_recip_rank, bm25_recip_rank
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    disp = PartialDependenceDisplay.from_estimator(
        rf["model"], X, targets, feature_names=FEATURES, ax=ax,
        line_kw={"color": BLUE, "lw": 2.0})
    for a in np.ravel(disp.axes_):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=GRAY, labelsize=9)
        a.set_xlabel(a.get_xlabel(), fontsize=9.5, color=GRAY)
        if a.get_ylabel():
            a.set_ylabel("partial dependence\n(P relevant)", fontsize=9.5, color=GRAY)
    fig.suptitle("Random forest — how reciprocal ranks drive the score",
                 fontsize=12.5, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rf_pdp.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def retrieval_diagnostics():
    """Train/dev only: per-retriever recall@100 and gold-in-candidates-but-not-top10 counts."""
    import csv
    runs = {r["query_id"]: r for r in json.loads((ART / "candidate_runs.json").read_text(encoding="utf-8"))}
    queries = json.loads((ART / "split_manifest.json").read_text(encoding="utf-8"))
    recall = {}
    for split in ("train", "dev"):
        cohort = [q for q in queries if q["split"] == split]
        recall[split] = {key: float(np.mean([len(set(q["positive_document_ids"]) & set(runs[q["query_id"]][field]))
                                             / len(q["positive_document_ids"]) for q in cohort]))
                         for key, field in (("bm25", "bm25_order"), ("semantic", "semantic_order"),
                                            ("union", "candidate_ids"))}
    path = ROOT / "results" / "20261114T144053296846Z_onnx" / "per_query_metrics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    misranked = {}
    for split in ("train", "dev"):
        hit = [r for r in rows if r["split"] == split and float(r["candidate_recall"]) == 1.0]
        misranked[split] = {"queries_with_gold_in_candidates": sum(r["method"] == "rrf" for r in hit),
                            **{m: sum(r["method"] == m and float(r["ndcg@10"]) == 0.0 for r in hit)
                               for m in ("bm25", "semantic", "rrf", "weighted_rrf", "logistic", "random_forest")}}
    return {"recall@100": recall, "gold_in_candidates_not_top10": misranked}


def answer_location_subgroups():
    """Dev only: learned minus weighted RRF nDCG@10 by where the answer sits in the source document."""
    import csv
    from ragstudy.metrics import paired_group_bootstrap
    path = ROOT / "results" / "iteration2_diagnostics" / "per_query_diagnostics.csv"
    location = {r["query_id"]: r["answer_location"] for r in csv.DictReader(path.open(encoding="utf-8"))}
    rows = [r for r in json.loads((ART / "per_query_metrics.json").read_text(encoding="utf-8"))
            if r["split"] == "dev"]
    out = {}
    for name, keep in (("answer_in_head", lambda v: v == "head"),
                       ("answer_after_or_across_word_256", lambda v: v in ("tail", "crosses_boundary"))):
        cohort = [r for r in rows if keep(location.get(r["query_id"]))]
        out[name] = {challenger: paired_group_bootstrap(cohort, "weighted_rrf", challenger, "ndcg@10")
                     for challenger in ("logistic", "random_forest")}
    return out


def main():
    lg, rf, bundle = load_models()
    coef = fig_logistic(lg)
    imp = fig_rf_importance(rf)
    fig_dev_methods()
    fig_tail_recall()
    fig_rf_pdp(rf, dev_feature_matrix())
    summ = json.loads((ART / "results.json").read_text(encoding="utf-8"))
    interp = {
        "source_run": "20261114T144053296846Z_onnx",
        "note": "Deployed models read from search_bundle.joblib; no refit; test metrics not read.",
        "feature_names": FEATURES,
        "logistic": {"C": lg["parameters"]["C"], "intercept": float(lg["model"].intercept_[0]),
                     "standardized_coefficients": dict(zip(FEATURES, [float(c) for c in coef])),
                     "cv_ndcg@10": lg["cv_ndcg@10"], "training_pairs": lg["training_pairs"]},
        "random_forest": {"params": rf["parameters"],
                          "impurity_importance": dict(zip(FEATURES, [float(i) for i in imp])),
                          "cv_ndcg@10": rf["cv_ndcg@10"]},
        "weighted_rrf": bundle["weighted_rrf_selection"], "rrf_constant": bundle["rrf_constant"],
        "dev_methods_ndcg10": {k: v["ndcg@10"] for k, v in summ["summary"]["dev"]["methods"].items()},
        "retrieval_diagnostics": retrieval_diagnostics(),
        "dev_answer_location_subgroups": answer_location_subgroups()}
    (OUT / "model_interpretation.json").write_text(
        json.dumps(interp, ensure_ascii=False, indent=2), encoding="utf-8")
    print("coef:", interp["logistic"]["standardized_coefficients"])
    print("importance:", interp["random_forest"]["impurity_importance"])
    print("assets ->", OUT)


if __name__ == "__main__":
    main()
# END



