"""Diagnose train/dev retrieval failures; no test scores and no model fitting."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import sqlite3
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/src"))
for directory in (".vendor", ".vendor-hf"):
    if (ROOT / "project" / directory).is_dir():
        sys.path.insert(0, str(ROOT / "project" / directory))

import numpy as np
from threadpoolctl import threadpool_limits
from ragstudy.data import sha256
from ragstudy.metrics import paired_group_bootstrap, ranking_metrics
from ragstudy.retrieval import BM25Index, top_k


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalized(text):
    return " ".join(text.split()).casefold()


def locate_answer(answer, document, head_words):
    """Exact normalized-string location; do not infer a location from overlap."""
    answer = normalized(answer)
    if not answer:
        return None
    full = normalized(document)
    position = full.find(answer)
    if position < 0:
        return None
    start = full[:position].count(" ")
    end = start + len(answer.split())
    return {"start_word_zero_based": start, "end_word_exclusive": end,
            "region": "head" if end <= head_words else "tail" if start >= head_words else "crosses_boundary"}


def first_gold_rank(ranking, gold):
    return next((rank for rank, doc in enumerate(ranking, 1) if doc in gold), None)


def summarize(rows):
    summary = {}
    for split in ("train", "dev"):
        cohort = [row for row in rows if row["split"] == split]
        assert cohort
        stats = {"queries": len(cohort), "groups": len({row["group_id"] for row in cohort}),
                 "original_candidate_misses": sum(row["original_candidate_recall"] == 0 for row in cohort),
                 "head_bm25_better_than_rrf_ndcg10": sum(row["head_bm25_ndcg10"] > row["rrf_ndcg10"] for row in cohort),
                 "rrf_better_than_head_bm25_ndcg10": sum(row["rrf_ndcg10"] > row["head_bm25_ndcg10"] for row in cohort),
                 "full_bm25_improves_ndcg10": sum(row["full_bm25_ndcg10"] > row["head_bm25_ndcg10"] for row in cohort),
                 "full_bm25_worsens_ndcg10": sum(row["full_bm25_ndcg10"] < row["head_bm25_ndcg10"] for row in cohort),
                 "full_bm25_rescues_original_candidate_misses": sum(
                     row["original_candidate_recall"] == 0 and row["full_bm25_rank100"] is not None for row in cohort),
                 "answer_location_counts": dict(Counter(row["answer_location"] for row in cohort)),
                 "retrieval": {}, "ranking_failures_despite_available_candidate": {}}
        for method in ("head_bm25", "full_bm25"):
            stats["retrieval"][method] = {
                "ndcg@10": float(np.mean([row[method + "_ndcg10"] for row in cohort])),
                "recall@20": float(np.mean([row[method + "_recall20"] for row in cohort])),
                "recall@100": float(np.mean([row[method + "_recall100"] for row in cohort]))}
        for method in ("rrf", "logistic", "random_forest"):
            stats["ranking_failures_despite_available_candidate"][method] = sum(
                row["original_candidate_recall"] > 0 and row[method + "_ndcg10"] == 0 for row in cohort)
        regions = {}
        for region in sorted({row["answer_location"] for row in cohort}):
            subgroup = [row for row in cohort if row["answer_location"] == region]
            regions[region] = {"queries": len(subgroup),
                               "original_candidate_misses": sum(row["original_candidate_recall"] == 0 for row in subgroup),
                               "head_bm25_recall100": float(np.mean([row["head_bm25_recall100"] for row in subgroup])),
                               "full_bm25_recall100": float(np.mean([row["full_bm25_recall100"] for row in subgroup]))}
        stats["answer_region_subgroups"] = regions
        paired = [{"query_id": row["query_id"], "group_id": row["group_id"],
                   "head": {"ndcg@10": row["head_bm25_ndcg10"]},
                   "full": {"ndcg@10": row["full_bm25_ndcg10"]}} for row in cohort]
        stats["full_vs_head_paired_ndcg10"] = paired_group_bootstrap(
            paired, "head", "full", "ndcg@10", iterations=2000, seed=20261114)
        summary[split] = stats
    return summary


def choose_cases(rows, detail):
    cases, selected = [], set()
    policies = [
        ("原候选漏检被全文BM25找回", lambda r: r["original_candidate_recall"] == 0 and r["full_bm25_rank100"] is not None,
         lambda r: (r["full_bm25_rank100"], r["query_id"])),
        ("尾部答案与全文BM25改善", lambda r: r["answer_location"] in ("tail", "crosses_boundary")
         and r["full_bm25_ndcg10"] > r["head_bm25_ndcg10"],
         lambda r: (r["head_bm25_ndcg10"] - r["full_bm25_ndcg10"], r["query_id"])),
        ("原BM25优于RRF", lambda r: r["head_bm25_ndcg10"] > r["rrf_ndcg10"],
         lambda r: (r["rrf_ndcg10"] - r["head_bm25_ndcg10"], r["query_id"])),
        ("全文BM25退化", lambda r: r["full_bm25_ndcg10"] < r["head_bm25_ndcg10"],
         lambda r: (r["full_bm25_ndcg10"] - r["head_bm25_ndcg10"], r["query_id"])),
    ]
    for category, predicate, order in policies:
        available = [row for row in rows if row["query_id"] not in selected and predicate(row)]
        # Development failures support the next decision; training is a fallback.
        available.sort(key=lambda row: (row["split"] != "dev", *order(row)))
        for row in available[:3]:
            selected.add(row["query_id"])
            cases.append({"category": category, **row, **detail[row["query_id"]]})
    assert len(cases) <= 12 and all(case["split"] in ("train", "dev") for case in cases)
    return cases


def markdown_report(summary, corpus, cases, head_words):
    dev = summary["dev"]
    head, full = dev["retrieval"]["head_bm25"], dev["retrieval"]["full_bm25"]
    lines = ["# TechQA 训练与开发集检索失败诊断", "",
             f"开发集 {dev['queries']} 个查询中，原候选并集漏检 {dev['original_candidate_misses']} 个；"
             f"全文 BM25 在 top100 找回其中 {dev['full_bm25_rescues_original_candidate_misses']} 个。"
             f"全文 BM25 的 nDCG@10 从 {head['ndcg@10']:.4f} 变为 {full['ndcg@10']:.4f}，"
             f"Recall@20 从 {head['recall@20']:.4f} 变为 {full['recall@20']:.4f}。"
             "单纯改为全文没有带来本开发集的总体收益；全文输入的改善与退化均保留，不能据个案判断所有长文都受益。", "",
             "## 比较范围", "",
             f"沿用原训练和开发划分、{corpus['documents']:,} 篇去重文档、BM25 参数和确定性的并列分数规则。"
             f"原输入是每篇前 {head_words} 个空白分隔词；对照只把 BM25 文档改为全文。"
             "索引全文来自全部未标注语料，不按 gold 筛文档，不训练模型，不把答案加进特征或候选。"
             "原 SQLite 只查询 train/dev，测试指标及案例不参与本次诊断。训练结果包含拟合数据，应以开发结果决定下一步。"
             "查询ID中的TRAIN或DEV前缀来自源文件，不表示本次自定义划分；本轮仍非官方保留集评测。", "",
             f"语料中 {corpus['longer_than_head']:,} 篇超过 {head_words} 词；完整正文词数中位数 "
             f"{corpus['document_words_median']:.0f}、第95百分位 {corpus['document_words_p95']:.0f}。", "",
             "## 候选与排名失败", "",
             "| 划分 | 查询 | 原候选漏检 | gold已在候选但RRF未进top10 | Logistic未进top10 | RF未进top10 | BM25的nDCG高于RRF |", 
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in ("train", "dev"):
        s = summary[split]
        failures = s["ranking_failures_despite_available_candidate"]
        lines.append(f"| {split} | {s['queries']} | {s['original_candidate_misses']} | {failures['rrf']} | "
                     f"{failures['logistic']} | {failures['random_forest']} | {s['head_bm25_better_than_rrf_ndcg10']} |")
    lines += ["", "候选漏检无法由候选内重排修复。BM25优于RRF的查询说明当前LSA与固定融合会稀释强词项证据；"
              "学习器超过RRF不等于超过BM25。", "", "## 文档输入对照", "",
              "| 划分 | 输入 | nDCG@10 | Recall@20 | Recall@100 |", "| --- | --- | ---: | ---: | ---: |"]
    for split in ("train", "dev"):
        for method, name in (("head_bm25", f"前{head_words}词"), ("full_bm25", "全文")):
            s = summary[split]["retrieval"][method]
            lines.append(f"| {split} | {name} | {s['ndcg@10']:.4f} | {s['recall@20']:.4f} | {s['recall@100']:.4f} |")
    lines += ["", f"开发集有 {dev['full_bm25_improves_ndcg10']} 个查询的 nDCG@10 提高、"
              f"{dev['full_bm25_worsens_ndcg10']} 个降低，其余相同。全文改变词频和长度归一化，"
              "因此平均变化不能全部归因于答案被截断。下一步应把全文BM25、统一分块后按文档聚合的检索与原前缀基线并列，"
              "固定同一划分和完整文档集合，再选择输入策略。", "",
              f"开发集全文减前缀的 nDCG@10 差值为 {dev['full_vs_head_paired_ndcg10']['difference']:+.4f}，"
              f"按 {dev['groups']} 个组件配对 bootstrap 的探索性95%区间为 "
              f"[{dev['full_vs_head_paired_ndcg10']['ci_lower']:+.4f}, {dev['full_vs_head_paired_ndcg10']['ci_upper']:+.4f}]。"
              "该开发集用于后续方案选择，区间不能当作新的最终测试证据。", "", "## 已知答案的位置", "",
              f"在原 gold 正文中查找源答案的大小写及空白归一化精确字符串，按空白词位置与第 {head_words} 词边界比较。"
              "此项使用答案只做人工错误诊断；找不到精确字符串的查询记为未知，不能据此认定没有证据。"
              "完整答案落在尾部能证明这一字符串不在前缀，但仍不能证明前缀完全没有可用线索。", "",
              "| 划分 | 精确答案位置 | 查询 | 原候选漏检 | 前缀BM25 Recall@100 | 全文BM25 Recall@100 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    labels = {"head": "前缀内", "tail": "边界后", "crosses_boundary": "跨边界", "not_exactly_located": "未精确定位"}
    for split in ("train", "dev"):
        for region, s in summary[split]["answer_region_subgroups"].items():
            lines.append(f"| {split} | {labels[region]} | {s['queries']} | {s['original_candidate_misses']} | "
                         f"{s['head_bm25_recall100']:.4f} | {s['full_bm25_recall100']:.4f} |")
    lines += ["", "## 下一轮比较重点", "",
              "优先检查固定融合为何降低强词项命中的名次。技术实体、CVE编号、版本号及错误码的一致性都可以从查询与文档直接计算，"
              "适合考虑作为可解释特征；这些建议需要训练组件CV支持，不能按开发案例逐条添加例外。BM25必须继续作为单独的强基线。", "",
              "长文分块应固定窗口、重叠和文档聚合规则，对所有方法使用一致的可见正文，并把块映射回原文档ID计算指标。"
              "若把前缀与全文或分块检索合并，必须重新报告候选规模和漏检率，所有重排方法排序同一完整并集。"
              "源答案及本次位置标注只用于诊断，不应转化为检索或训练时可见的字段。", "",
              "## 代表案例", "",
              "每类最多选三个，优先开发集；先按该类量化差异排序，再按查询ID确定并列顺序。案例仅用于解释，不代表随机样本。"]
    for number, case in enumerate(cases, 1):
        ranks = lambda key: str(case[key]) if case[key] is not None else "未进已记录范围"
        question = " ".join(case["question"].split())
        if len(question) > 350:
            question = question[:347] + "..."
        lines += ["", f"### 案例 {number} {case['category']}", "",
                  f"查询 `{case['query_id']}`，{case['split']}：{question}", "",
                  f"gold：`{', '.join(case['gold_document_ids'])}`。前缀BM25 gold名次 {ranks('head_bm25_rank100')}，"
                  f"全文BM25 {ranks('full_bm25_rank100')}，原RRF top20内名次 {ranks('rrf_rank20')}；"
                  f"原候选召回 {case['original_candidate_recall']:.0f}。"
                  f"源答案位置：{labels[case['answer_location']]}，起始空白词序号 {case['answer_start_word']}（从0计）。", "",
                  "源答案摘录：" + case["source_answer_excerpt"], "",
                  "gold正文相关位置摘录：" + case["gold_evidence_excerpt"]]
    lines += ["", "## 复核材料", "",
              "[汇总与来源哈希](summary.json)、[全部训练开发查询诊断](per_query_diagnostics.csv)、"
              "[最多十二个案例](representative_cases.json)。全文索引耗时是本机离线构建成本，不能作为在线查询延迟。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "project/artifacts/20261114T113500971378Z_lsa")
    parser.add_argument("--output", type=Path, default=ROOT / "project/results/iteration2_diagnostics")
    args = parser.parse_args()
    config = json.loads((args.run / "configuration.json").read_text(encoding="utf-8"))
    audit = json.loads((args.run / "data_audit.json").read_text(encoding="utf-8"))
    source = ROOT / config["source"]
    for name, digest in audit["source_hashes"].items():
        assert sha256(source / name) == digest, f"Original source changed: {name}"
    database_path = (args.run / "experiments.sqlite").resolve()
    with sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True) as database:
        queries = [{"query_id": qid, "question": question, "split": split, "group_id": group}
                   for qid, question, split, group in database.execute(
                       "SELECT query_id,question,split,group_id FROM queries WHERE split IN ('train','dev') ORDER BY query_id")]
        passages = list(database.execute("SELECT document_id,retrieval_text FROM passages ORDER BY document_id"))
        gold = defaultdict(set)
        for qid, doc in database.execute("SELECT g.query_id,g.document_id FROM qrels g JOIN queries q USING(query_id) WHERE q.split IN ('train','dev')"):
            gold[qid].add(doc)
        candidates = defaultdict(set)
        for qid, doc in database.execute("SELECT c.query_id,c.document_id FROM candidates c JOIN queries q USING(query_id) WHERE q.split IN ('train','dev')"):
            candidates[qid].add(doc)
        rankings = defaultdict(lambda: defaultdict(list))
        for qid, method, rank, doc in database.execute(
                "SELECT r.query_id,r.method,r.rank,r.document_id FROM rankings r JOIN queries q USING(query_id) "
                "WHERE q.split IN ('train','dev') ORDER BY r.query_id,r.method,r.rank"):
            rankings[qid][method].append(doc)
    allowed_ids = {query["query_id"] for query in queries}
    assert queries and all(query["split"] in ("train", "dev") for query in queries)
    manifest = {query["query_id"]: query for query in json.loads((args.run / "split_manifest.json").read_text(encoding="utf-8"))
                if query["query_id"] in allowed_ids}
    source_ids = {qid for query in manifest.values() for qid in query["source_record_ids"]}
    answer_rows = {row["id"]: row for row in json.loads((source / "train.json").read_text(encoding="utf-8"))
                   if row["id"] in source_ids}
    ids, heads = zip(*passages)
    original = {}
    canonical_ids = set(ids)
    with zipfile.ZipFile(source / "corpus.zip") as archive:
        for member in archive.infolist():
            document_id = Path(member.filename).name
            if not member.is_dir() and document_id in canonical_ids:
                original[document_id] = archive.read(member).decode("utf-8", errors="replace")
    assert len(original) == len(ids)
    assert all(" ".join(original[doc].split()[:config["head_words"]]) == head for doc, head in passages)
    full_texts = [original[doc] for doc in ids]
    word_lengths = [len(text.split()) for text in full_texts]
    print(f"Confirmed {len(queries)} train/dev queries and {len(ids)} unchanged corpus documents", flush=True)
    began = time.perf_counter()
    head_index = BM25Index(list(heads))
    head_seconds = time.perf_counter() - began
    print(f"Head BM25 ready in {head_seconds:.2f}s; building full-text BM25", flush=True)
    began = time.perf_counter()
    full_index = BM25Index(full_texts)
    full_seconds = time.perf_counter() - began
    print(f"Full BM25 ready in {full_seconds:.2f}s", flush=True)
    rows, detail = [], {}
    for number, query in enumerate(queries, 1):
        qid, question = query["query_id"], query["question"]
        head_order = [ids[index] for index in top_k(head_index.score(question), 100, ids, min_score=0)]
        full_order = [ids[index] for index in top_k(full_index.score(question), 100, ids, min_score=0)]
        assert head_order[:20] == rankings[qid]["bm25"], f"Historical BM25 ranking mismatch: {qid}"
        row = {"query_id": qid, "split": query["split"], "group_id": query["group_id"],
               "gold_document_ids": ";".join(sorted(gold[qid])),
               "original_candidate_recall": len(gold[qid] & candidates[qid]) / len(gold[qid]),
               "candidate_count": len(candidates[qid])}
        for method, order in (("head_bm25", head_order), ("full_bm25", full_order)):
            metrics = ranking_metrics(order, gold[qid])
            row.update({method + "_ndcg10": metrics["ndcg@10"], method + "_recall20": metrics["recall@20"],
                        method + "_recall100": len(gold[qid] & set(order)) / len(gold[qid]),
                        method + "_rank100": first_gold_rank(order, gold[qid])})
        for method in ("rrf", "logistic", "random_forest"):
            metrics = ranking_metrics(rankings[qid][method], gold[qid])
            row[method + "_ndcg10"] = metrics["ndcg@10"]
            row[method + "_recall20"] = metrics["recall@20"]
        row["rrf_rank20"] = first_gold_rank(rankings[qid]["rrf"], gold[qid])
        answers = list(dict.fromkeys(answer_rows[source_id]["answer"] for source_id in manifest[qid]["source_record_ids"]))
        locations = [{**location, "document_id": doc, "answer": answer}
                     for answer in answers for doc in sorted(gold[qid])
                     if (location := locate_answer(answer, original[doc], config["head_words"])) is not None]
        locations.sort(key=lambda location: (location["region"] != "head", location["start_word_zero_based"], location["document_id"]))
        best = locations[0] if locations else None
        row.update({"answer_location": best["region"] if best else "not_exactly_located",
                    "answer_start_word": best["start_word_zero_based"] if best else None,
                    "answer_end_word": best["end_word_exclusive"] if best else None,
                    "gold_max_words": max(len(original[doc].split()) for doc in gold[qid])})
        words = original[best["document_id"] if best else sorted(gold[qid])[0]].split()
        start = max(0, best["start_word_zero_based"] - 20) if best else 0
        detail[qid] = {"question": question, "gold_document_ids": sorted(gold[qid]),
                       "source_record_ids": manifest[qid]["source_record_ids"],
                       "source_answer_excerpt": " ".join((best["answer"] if best else answers[0]).split())[:500],
                       "gold_evidence_excerpt": " ".join(words[start:start + 85]),
                       "head_bm25_top3": head_order[:3], "full_bm25_top3": full_order[:3],
                       "rrf_top3": rankings[qid]["rrf"][:3]}
        rows.append(row)
        if number % 100 == 0:
            print(f"Diagnosed {number}/{len(queries)} train/dev queries", flush=True)
    summary = summarize(rows)
    corpus = {"documents": len(ids), "head_words": config["head_words"],
              "longer_than_head": sum(length > config["head_words"] for length in word_lengths),
              "document_words_median": float(np.median(word_lengths)), "document_words_p95": float(np.percentile(word_lengths, 95)),
              "head_index_seconds": head_seconds, "full_index_seconds": full_seconds,
              "head_bm25_metadata": head_index.metadata, "full_bm25_metadata": full_index.metadata}
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "train/dev error analysis only",
              "test_metrics_read": False, "test_queries_scored": 0, "answers_used_for_training_or_retrieval": False,
              "historical_head_top20_match": True, "corpus": corpus, "summary": summary,
              "source_hashes": audit["source_hashes"], "historical_database_sha256": sha256(database_path),
              "historical_configuration_sha256": sha256(args.run / "configuration.json"), "script_sha256": sha256(Path(__file__)),
              "answer_location_rule": "first exact match after whitespace/case normalization; zero-based whitespace word positions; unmatched is unknown",
              "full_text_comparison": "same unlabelled canonical corpus, BM25 k1/b unchanged; no model fitting, gold insertion or test scoring"}
    cases = choose_cases(rows, detail)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", result)
    write_json(args.output / "representative_cases.json", cases)
    with (args.output / "per_query_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "README.md").write_text(markdown_report(summary, corpus, cases, config["head_words"]), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
