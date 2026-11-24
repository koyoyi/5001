"""Render verified iteration2 results without training or opening an unscored test.

Usage: python project/scripts/summarize_iteration2.py --main MAIN_ARTIFACT
Add --include-holdout only to include an already completed, registered and
independently verified frozen evaluation. This script never runs evaluation.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/src"))
from ragstudy.metrics import paired_group_bootstrap

HEAD_LSA = "20261114T135003226733Z_lsa"
FULL_LSA = "20261114T135456567288Z_lsa"
LABELS = {"head_lsa": "Head LSA", "full_lsa": "Full LSA", "head_onnx": "BGE head FP32"}
METHOD_LABELS = {"bm25": "BM25", "semantic": "Semantic", "rrf": "RRF",
                 "weighted_rrf": "Weighted RRF", "logistic": "Logistic", "random_forest": "Random forest"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12), message)


def artifact_path(root, value):
    value = Path(value)
    path = (root / value).resolve() if not value.is_absolute() else value.resolve()
    require(path.parent == (root / "project/artifacts").resolve() and path.is_dir(),
            "Use a completed artifact directly in this workspace's project/artifacts")
    return path


def checked_run(root, artifact, *, heldout=False):
    """Read matching independent verification evidence; never load a model."""
    proof = read_json(artifact / "verification.json")
    require(proof.get("status") == "passed" and proof.get("run_id") == artifact.name,
            "Run has no matching passed independent verification: " + artifact.name)
    required = {"results.json", "configuration.json", "per_query_metrics.json", "split_manifest.json",
                "data_audit.json", "experiments.sqlite", "candidate_runs.json"}
    require(required <= set(proof.get("evidence_sha256", {})), "Verification lacks required evidence hashes")
    for name, expected in proof["evidence_sha256"].items():
        path = artifact / name
        require(path.parent == artifact and path.is_file() and sha256(path) == expected,
                "Independent verification is stale or evidence changed: " + name)
    require(proof.get("checks", {}).get("exact_candidate_union") in
            ("saved_retrieval_runs", "recomputed_from_saved_local_index"), "Candidate union provenance is unverified")
    result, config = read_json(artifact / "results.json"), read_json(artifact / "configuration.json")
    require(result["run_id"] == artifact.name and result["configuration_sha256"] == sha256(artifact / "configuration.json"),
            "Run/configuration identity mismatch")
    require(bool(result["final_test_evaluated"]) == heldout, "Run test scope differs from the requested table")
    # Main runs contain only train/dev metrics. Do not access any raw qrel,
    # source-query file or SQLite table to compile this report.
    rows = read_json(artifact / "per_query_metrics.json")
    require(heldout or all(row["split"] != "test" for row in rows), "Primary development artifact contains test metrics")
    tracked = root / "project/results" / artifact.name
    require(tracked.is_dir() and read_json(tracked / "results.json") == result, "Tracked result differs from local evidence")
    csv_path = tracked / "per_query_metrics.csv"
    if proof.get("tracked_csv_sha256"):
        require(csv_path.is_file() and sha256(csv_path) == proof["tracked_csv_sha256"], "Tracked CSV changed after verification")
    evidence = {"artifact": artifact.relative_to(root).as_posix(), "run_id": artifact.name,
                "result": (tracked / "results.json").relative_to(root).as_posix(),
                "configuration": ((root / "project/results" / artifact.name / "configuration.json")
                                  if (root / "project/results" / artifact.name / "configuration.json").is_file()
                                  else artifact / "configuration.json").relative_to(root).as_posix(),
                "verification": (tracked / "verification.json").relative_to(root).as_posix(),
                "configuration_sha256": result["configuration_sha256"],
                "verification_sha256": sha256(artifact / "verification.json"),
                "result_sha256": sha256(artifact / "results.json"),
                "per_query_metrics_sha256": sha256(artifact / "per_query_metrics.json"),
                "split_manifest_sha256": sha256(artifact / "split_manifest.json")}
    return {"path": artifact, "result": result, "config": config, "rows": rows, "evidence": evidence}


def checked_cohort(run, split):
    rows = [row for row in run["rows"] if row["split"] == split]
    require(bool(rows) and len({row["query_id"] for row in rows}) == len(rows), "Empty or duplicate metric cohort")
    summary = run["result"]["summary"][split]
    require(len(rows) == summary["queries"] and len({row["group_id"] for row in rows}) == summary["groups"], "Metric cohort count differs")
    for method, metrics in summary["methods"].items():
        for metric in ("ndcg@10", "recall@20"):
            close(math.fsum(row[method][metric] for row in rows) / len(rows), metrics[metric], "Aggregate differs from raw per-query metrics")
    close(math.fsum(row["candidate_recall"] for row in rows) / len(rows), summary["candidate_recall"], "Candidate recall aggregate differs")
    return rows, summary


def contrast_rows(run, split, pipeline):
    rows, summary = checked_cohort(run, split)
    config = run["config"]
    require({"bm25", "weighted_rrf", "logistic", "random_forest"} <= set(summary["methods"]), "Required comparison method missing")
    comparisons = []
    for baseline in ("bm25", "weighted_rrf"):
        for challenger in ("logistic", "random_forest"):
            for metric in ("ndcg@10", "recall@20"):
                value = paired_group_bootstrap(rows, baseline, challenger, metric,
                    iterations=config["bootstrap_iterations"], seed=config["seed"])
                comparisons.append({"pipeline": pipeline, "split": split, **value})
    return comparisons


def method_rows(run, split, pipeline):
    _, summary = checked_cohort(run, split)
    return [{"pipeline": pipeline, "split": split, "method": method, **metrics,
             "candidate_recall": summary["candidate_recall"], "queries": summary["queries"],
             "groups": summary["groups"], "source_run": run["result"]["run_id"]}
            for method, metrics in summary["methods"].items()]


def raw_metric_rows(run, split, pipeline):
    return [{"pipeline": pipeline, "split": split, "query_id": row["query_id"],
             "group_id": row["group_id"], "method": method, **row[method],
             "candidate_recall": row["candidate_recall"]}
            for row in run["rows"] if row["split"] == split for method in row["rankings"]]


def collect_ablations(root, main):
    path = root / "project/results/iteration2_ablations.json"
    if not path.is_file():
        return {"status": "not_present", "methods": [], "contrasts_vs_main": []}
    payload = read_json(path)
    variants = payload.get("variants", {})
    if payload.get("status") != "complete" or not variants or any(row.get("state") != "complete" for row in variants.values()):
        return {"status": "incomplete", "variant_states": {name: row.get("state") for name, row in variants.items()},
                "methods": [], "contrasts_vs_main": []}
    expected = read_json(root / "project/results/iteration2_plan.json")["robustness_one_factor_at_a_time"]
    require(set(variants) == set(expected) and len(variants) == 6, "Completed sensitivity suite differs from the declared six variants")
    require(payload["provenance"]["source_artifact"] == main["evidence"]["artifact"], "Ablation suite belongs to a different primary source")
    baseline = {row["query_id"]: row for row in main["rows"] if row["split"] == "dev"}
    methods, comparisons, sources = [], [], {}
    for name in expected:
        entry = variants[name]
        current = checked_run(root, artifact_path(root, entry["artifact"]))
        require(current["path"].joinpath("split_manifest.json").read_bytes() == main["path"].joinpath("split_manifest.json").read_bytes(), "Sensitivity run changed the query partition")
        require(current["result"]["configuration_sha256"] == entry["artifact_configuration_sha256"], "Sensitivity configuration hash changed")
        require(current["result"]["summary"]["dev"] == entry["development"], "Sensitivity summary differs from its verified run")
        require(current["config"].get("model_source_run") == main["evidence"]["artifact"]
                and current["config"].get("fixed_model_parameters") is not None
                and not current["result"]["cross_validation"], "Sensitivity run lacks fixed main-CV parameters")
        methods += [row for row in method_rows(current, "dev", name) if row["method"] in ("logistic", "random_forest")]
        dev = [row for row in current["rows"] if row["split"] == "dev"]
        require({row["query_id"] for row in dev} == set(baseline), "Sensitivity development queries differ")
        for model in ("logistic", "random_forest"):
            rows = [{"query_id": row["query_id"], "group_id": row["group_id"],
                     "main": baseline[row["query_id"]][model], "variant": row[model]} for row in dev]
            require(all(row["group_id"] == baseline[row["query_id"]]["group_id"] for row in dev), "Sensitivity group changed")
            for metric in ("ndcg@10", "recall@20"):
                value = paired_group_bootstrap(rows, "main", "variant", metric,
                    iterations=current["config"]["bootstrap_iterations"], seed=current["config"]["seed"])
                comparisons.append({"variant": name, "model": model, "split": "dev", **value})
        sources[name] = current["evidence"]
    return {"status": "complete", "methods": methods, "contrasts_vs_main": comparisons,
            "sources": sources, "summary_sha256": sha256(path),
            "selection_policy": "Development sensitivities do not replace the preregistered main pipeline"}


def collect_holdout(root, main, include):
    if not include:
        return {"status": "not_requested", "test_metrics_read": False, "methods": [], "contrasts": []}
    registry_path = root / "project/artifacts/frozen_holdout_registry.json"
    if not registry_path.is_file():
        return {"status": "not_registered", "test_metrics_read": False, "methods": [], "contrasts": []}
    registry = read_json(registry_path)
    artifact = artifact_path(root, Path("project/artifacts") / registry["run_id"])
    manifest = read_json(artifact / "freeze_manifest.json")
    require(manifest["identity_sha256"] == registry["identity_sha256"] and manifest["identity"] == registry["identity"], "Frozen registration identity mismatch")
    # Incomplete/evaluating states must not cause results/test_predictions reads.
    if manifest.get("status") != "completed" or manifest.get("heldout_scored") is not True:
        return {"status": manifest.get("status", "incomplete"), "test_metrics_read": False, "methods": [], "contrasts": []}
    require(manifest["identity"]["source_artifact"] == main["evidence"]["artifact"], "Frozen evaluation belongs to a different primary run")
    for name, expected in manifest["identity"]["source_files_sha256"].items():
        path = main["path"] / name
        require(path.parent == main["path"] and sha256(path) == expected, "Frozen primary source changed")
    for name, expected in manifest["output_sha256"].items():
        path = artifact / name
        require(path.parent == artifact and sha256(path) == expected, "Frozen evaluated output changed")
    run = checked_run(root, artifact, heldout=True)
    require((artifact / "split_manifest.json").read_bytes() == (main["path"] / "split_manifest.json").read_bytes(), "Frozen query partition differs from primary")
    require([row for row in run["rows"] if row["split"] != "test"] == main["rows"], "Frozen evaluation changed original train/dev rows")
    protocol = read_json(artifact / "protocol.json")
    require(sha256(artifact / "protocol.json") == manifest["identity"]["protocol_sha256"], "Frozen protocol changed")
    comparisons = contrast_rows(run, "test", "head_onnx")
    primary = next(row for row in comparisons if row["baseline"] == "weighted_rrf"
                   and row["challenger"] == "logistic" and row["metric"] == "ndcg@10")
    original = run["result"]["primary_holdout_contrast"]
    for key in ("baseline", "challenger", "metric"):
        require(original[key] == primary[key], "Primary preregistered contrast differs")
    for key in ("difference", "ci_lower", "ci_upper"):
        close(original[key], primary[key], "Frozen primary contrast differs from saved test rows")
    return {"status": "completed", "test_metrics_read": True, "methods": method_rows(run, "test", "head_onnx"),
            "contrasts": comparisons, "primary_contrast": primary,
            "stopping_rule": protocol["stopping_rule"], "limitations": protocol["limitations"],
            "source": run["evidence"], "freeze_identity_sha256": manifest["identity_sha256"],
            "freeze_manifest_sha256": sha256(artifact / "freeze_manifest.json"),
            "protocol_sha256": sha256(artifact / "protocol.json")}


def csv_file(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_contrasts(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 7.2), constrained_layout=True)
    keys = list(dict.fromkeys((row["pipeline"], row["baseline"], row["challenger"]) for row in rows))
    for axis, metric in zip(axes, ("ndcg@10", "recall@20")):
        for y, key in enumerate(keys):
            row = next(value for value in rows if (value["pipeline"], value["baseline"], value["challenger"]) == key and value["metric"] == metric)
            color = "#226b98" if row["baseline"] == "bm25" else "#a14c2b"
            axis.hlines(y, row["ci_lower"], row["ci_upper"], color=color, linewidth=1.7)
            axis.plot(row["difference"], y, "o", color=color, markersize=4)
        axis.axvline(0, color="#555555", linestyle="--", linewidth=.8)
        axis.set_title(metric + " difference")
        axis.set_xlabel("Challenger minus baseline; 95% paired component interval")
        axis.set_yticks(range(len(keys)))
        axis.set_yticklabels([f"{LABELS[pipeline]} | {METHOD_LABELS[challenger]} - {METHOD_LABELS[baseline]}" for pipeline, baseline, challenger in keys] if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=.18)
    figure.suptitle("Development comparisons on identical query partitions (exploratory)")
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def report_markdown(payload):
    data, dev = payload["data_scope"], payload["development"]
    lines = ["# TechQA 第二轮实验结果", "",
             "主线为预先确定的 BGE FP32 文档头检索与轻量监督重排序。开发集用于固定范围的融合选择和敏感性分析；三条管线使用完全相同的问题分区。", "",
             f"固定语料含 {data['raw_documents']:,} 篇原始文档，去重后 {data['unique_documents']:,} 篇；{data['queries_before_quarantine']} 个候选问题中隔离 {data['quarantined_queries']} 个跨历史分区问题，保留 {data['queries']} 个问题。",
             f"Train、Dev、Test 分别为 {data['splits']['train']['queries']}、{data['splits']['dev']['queries']}、{data['splits']['test']['queries']} 个问题；开发集含 {data['splits']['dev']['groups']} 个独立组件。", "",
             "## 开发集方法比较", "",
             "Head LSA 和 BGE FP32 使用前 256 个空白分隔词；BGE 再按本地 Tokenizer 截断至 512 token。Full LSA 使用原始全文。nDCG@10 与 Recall@20 按问题宏平均；Recall 针对提供的正例，未标注候选作为代理负例。", "",
             "| 管线 | 方法 | nDCG@10 | Recall@20 | 候选召回 |", "| --- | --- | ---: | ---: | ---: |"]
    for row in dev["methods"]:
        lines.append(f"| {LABELS[row['pipeline']]} | {METHOD_LABELS.get(row['method'], row['method'])} | {row['ndcg@10']:.6f} | {row['recall@20']:.6f} | {row['candidate_recall']:.6f} |")
    lines += ["", "![开发集配对比较](iteration2_development_contrasts.png)", "",
              "区间按独立组件配对重采样，保留组件内全部问题；图中差值为学习模型减对应基线。开发集区间属于探索性比较，包含融合选择和多项敏感性分析的影响。", "",
              "精确数值：[方法表](iteration2_development_methods.csv)、[逐问题指标](iteration2_development_per_query.csv)、[配对区间](iteration2_development_contrasts.csv)、[完整 JSON](iteration2_summary.json)。", "",
              "## 固定模型参数", "", "| 管线 | Logistic C | RF 深度 | RF 最小叶节点 | 树数 | 训练权重 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for key, settings in payload["pipelines"].items():
        parameters = settings["selected_parameters"]
        total = settings["training"].get("query_weight_total")
        weighting = f"每问题总权重 {total:g}" if total is not None else "各问题等总权重，样本权重再归一至均值 1"
        lines.append(f"| {LABELS[key]} | {parameters['logistic']['C']:g} | {parameters['random_forest']['max_depth']} | {parameters['random_forest']['min_samples_leaf']} | {settings['training']['rf_trees']} | {weighting} |")
    lines += ["", "模型参数由各自训练组件交叉验证选择。加权 RRF 的常数和语义权重取自各管线预先给定的开发集网格；语义权重为 0 时，该配置不使用语义名次项。", "", "## 单因素敏感性分析", ""]
    ablations = payload["ablations"]
    if ablations["status"] == "complete":
        lines += ["六个变体固定复用主线训练 CV 参数和同一原始候选、特征、分区，主线仍按预先记录的方案保留。负采样上限比较固定每问题总权重，控制 Logistic 有效正则化的变化。", "",
                  "| 变体 | 模型 | nDCG@10 | Recall@20 | 相对主线 nDCG 差值及 95% CI |", "| --- | --- | ---: | ---: | --- |"]
        for row in ablations["methods"]:
            comparison = next(value for value in ablations["contrasts_vs_main"] if value["variant"] == row["pipeline"] and value["model"] == row["method"] and value["metric"] == "ndcg@10")
            lines.append(f"| {row['pipeline']} | {METHOD_LABELS[row['method']]} | {row['ndcg@10']:.6f} | {row['recall@20']:.6f} | {comparison['difference']:+.6f} [{comparison['ci_lower']:+.6f}, {comparison['ci_upper']:+.6f}] |")
        lines += ["", "[单因素精确数值](iteration2_ablation_methods.csv)与[相对主线区间](iteration2_ablation_contrasts.csv)。", ""]
    else:
        lines += [f"六个变体的汇总状态为 `{ablations['status']}`；完成且通过独立核验后才纳入比较表。", ""]
    lines += ["## 冻结保留集评价", ""]
    holdout = payload["holdout"]
    if holdout["status"] == "completed":
        counts = holdout["methods"][0]
        lines += [f"以下对应已完成的冻结评价，共 {counts['queries']} 个问题、{counts['groups']} 个独立组件；模型、融合参数、候选及特征均来自已固定的主线。", "",
                  "| 方法 | nDCG@10 | Recall@20 | 候选召回 |", "| --- | ---: | ---: | ---: |"]
        for row in holdout["methods"]:
            lines.append(f"| {METHOD_LABELS.get(row['method'], row['method'])} | {row['ndcg@10']:.6f} | {row['recall@20']:.6f} | {row['candidate_recall']:.6f} |")
        primary = holdout["primary_contrast"]
        lines += ["", f"本地预先记录的主要对比为 Logistic 相对加权 RRF 的 nDCG@10：差值 {primary['difference']:+.6f}，95% 组件配对区间 [{primary['ci_lower']:+.6f}, {primary['ci_upper']:+.6f}]。该协议未作外部注册。", "",
                  "该区间跨过 0，主要对比没有显示学习重排稳定优于强融合；其他方法的最高点分数不能替代这一预先指定的主要结论。" if primary['ci_lower'] <= 0 <= primary['ci_upper'] else "主要对比区间未跨 0；解释仍以预先限定的评价对象和方向为准。", "",
                  f"停止规则：{holdout['stopping_rule']}", "", "[保留集精确方法表](iteration2_holdout_methods.csv)与[配对区间](iteration2_holdout_contrasts.csv)。", ""]
    else:
        lines += [f"状态为 `{holdout['status']}`；本次汇总未读取保留集评价指标，也未运行评价。", ""]
    lines += ["## 结果边界和复现来源", "",
              "本实验评价固定技术支持档案中的已提供相关性标注；标注稀疏、来源文本冲突和近重复审查覆盖限制仍保留。语义逐对判读由 Codex 助手完成，真实成员或领域专家尚未独立核验。检索指标不代表生成答案正确率或穷尽相关性召回。结果 JSON 中的采样耗时为单次观察；独立重复延迟另存 latency_benchmark.json，缓存加载与首次编码分开记录。", ""]
    for key, settings in payload["pipelines"].items():
        evidence = settings["evidence"]
        rel = lambda value: str(Path(value).relative_to("project")).replace("\\", "/")
        result_link = evidence["run_id"] + "/results.json"
        config_link = "../" + rel(evidence["configuration"])
        proof_link = evidence["run_id"] + "/verification.json"
        lines.append(f"- {LABELS[key]}：[{evidence['run_id']}]({result_link})、[配置]({config_link})、[独立核验]({proof_link})。")
    if holdout["status"] == "completed":
        run_id = holdout["source"]["run_id"]
        lines.append(f"- 冻结评价：[{run_id}]({run_id}/results.json)、[冻结记录]({run_id}/freeze_manifest.json)、[协议]({run_id}/protocol.json)、[独立核验]({run_id}/verification.json)。")
    model = payload["pipelines"]["head_onnx"]["semantic_metadata"]
    revision = model.get("model_revision", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
    lines += [f"- BGE 模型：[固定版本 BAAI bge small en v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/{revision})；实际 ONNX、输入和配置哈希保存在 JSON 与各 run 的核验记录中。", ""]
    return "\n".join(lines)


def summarize(main_artifact, *, root=ROOT, head_lsa=HEAD_LSA, full_lsa=FULL_LSA, include_holdout=False, write=True):
    root = Path(root).resolve()
    runs = {"head_lsa": checked_run(root, artifact_path(root, Path("project/artifacts") / head_lsa)),
            "full_lsa": checked_run(root, artifact_path(root, Path("project/artifacts") / full_lsa)),
            "head_onnx": checked_run(root, artifact_path(root, main_artifact))}
    main = runs["head_onnx"]
    require(main["config"].get("analysis_name") == "head_onnx"
            and not main["config"].get("ablation_name")
            and main["config"].get("fixed_model_parameters") is None,
            "The declared head_onnx main pipeline cannot be replaced by a development-selected ablation")
    expected = {"head_lsa": ("lsa", "head"), "full_lsa": ("lsa", "full"), "head_onnx": ("onnx", "head")}
    for name, run in runs.items():
        require((run["config"]["semantic_backend"], run["config"].get("document_policy", "head")) == expected[name], "Source pipeline policy mismatch: " + name)
        require((run["path"] / "split_manifest.json").read_bytes() == (main["path"] / "split_manifest.json").read_bytes(),
                "Three split manifests must be byte-identical before direct comparison")
        require(run["result"]["source_hashes"] == main["result"]["source_hashes"], "Pipeline dataset snapshots differ")
    require(main["result"]["data_audit"].get("split_policy") == "historical_split_lock_with_component_quarantine",
            "Primary source is not the reviewed v2 historical-lock/quarantine cohort")
    methods, contrasts, raw_metrics, pipelines = [], [], [], {}
    for name, run in runs.items():
        methods += method_rows(run, "dev", name)
        contrasts += contrast_rows(run, "dev", name)
        raw_metrics += raw_metric_rows(run, "dev", name)
        pipelines[name] = {"evidence": run["evidence"], "selected_parameters": {
            kind: run["result"]["selected_models"][kind]["parameters"] for kind in ("logistic", "random_forest")},
            "training": {key: run["config"].get(key) for key in ("max_proxy_negatives", "training_seed", "query_weight_total", "rf_trees")},
            "weighted_rrf_selection": run["result"].get("weighted_rrf_selection"),
            "semantic_metadata": run["result"]["retrieval_metadata"]["semantic"]}
    audit = main["result"]["data_audit"]
    payload = {"scope": "Verified fixed-cohort development comparison; heldout only from completed frozen registration",
               "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "summarizer_sha256": sha256(Path(__file__)), "main_pipeline": "head_onnx",
               "main_selection_policy": "Predetermined main pipeline; development maxima do not replace it",
               "data_scope": {"raw_documents": audit["raw_documents"], "unique_documents": audit["unique_documents"],
                              "queries_before_quarantine": audit.get("canonical_queries_before_quarantine", audit["canonical_answerable_queries"]),
                              "quarantined_queries": len(audit.get("quarantined_queries", [])),
                              "queries": audit["canonical_answerable_queries"], "splits": audit["splits"]},
               "pipelines": pipelines, "development": {"methods": methods, "contrasts": contrasts, "per_query_metrics": raw_metrics},
               "ablations": collect_ablations(root, main), "holdout": collect_holdout(root, main, include_holdout)}
    if write:
        output = root / "project/results"
        method_fields = ["pipeline", "split", "method", "ndcg@10", "recall@20", "candidate_recall", "queries", "groups", "source_run"]
        contrast_fields = ["pipeline", "split", "baseline", "challenger", "metric", "difference", "ci_lower", "ci_upper", "n_queries", "n_groups", "iterations", "seed"]
        csv_file(output / "iteration2_development_methods.csv", methods, method_fields)
        csv_file(output / "iteration2_development_per_query.csv", raw_metrics,
                 ["pipeline", "split", "query_id", "group_id", "method", "ndcg@10", "recall@20", "candidate_recall"])
        csv_file(output / "iteration2_development_contrasts.csv", contrasts, contrast_fields)
        csv_file(output / "iteration2_ablation_methods.csv", payload["ablations"]["methods"], method_fields)
        csv_file(output / "iteration2_ablation_contrasts.csv", payload["ablations"]["contrasts_vs_main"], ["variant", "model", *contrast_fields[1:]])
        csv_file(output / "iteration2_holdout_methods.csv", payload["holdout"]["methods"], method_fields)
        csv_file(output / "iteration2_holdout_contrasts.csv", payload["holdout"]["contrasts"], contrast_fields)
        plot_contrasts(contrasts, output / "iteration2_development_contrasts.png")
        (output / "iteration2_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (output / "iteration2_summary.md").write_text(report_markdown(payload), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--include-holdout", action="store_true", help="Include only an already completed frozen evaluation; never score")
    args = parser.parse_args()
    try:
        payload = summarize(args.main, include_holdout=args.include_holdout)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Summary rejected: {error}\n")
    print(json.dumps({"main_pipeline": payload["main_pipeline"], "development_rows": len(payload["development"]["methods"]),
                      "ablations": payload["ablations"]["status"], "holdout": payload["holdout"]["status"],
                      "output": str(ROOT / "project/results/iteration2_summary.md")}, indent=2))


if __name__ == "__main__":
    main()
