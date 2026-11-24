"""Build five small, executable notebooks that review genuine local run artifacts."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
SETUP = '''from pathlib import Path
import json, sys, os
import pandas as pd
from IPython.display import display, Image
root = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'project/src/ragstudy').is_dir())
for folder in ('.vendor', '.vendor-hf', '.vendor-onnx-compat'):
    path = root / 'project' / folder
    if path.is_dir():
        sys.path.insert(0, str(path))
sys.path.insert(0, str(root / 'project/src'))
artifacts = sorted(path for path in (root / 'project/artifacts').iterdir()
                   if (path / 'results.json').is_file() and (path / 'search_bundle.joblib').is_file())
assert artifacts, 'First run python project/scripts/run_experiment.py --backend lsa'
artifact = Path(os.environ['RAG_ARTIFACT']).resolve() if os.environ.get('RAG_ARTIFACT') else artifacts[-1]
result = json.loads((artifact / 'results.json').read_text(encoding='utf-8'))
print('Actual local run:', artifact.name, '| backend:', result['retrieval_metadata']['semantic']['method'])'''

NOTEBOOKS = [
    ("01_data_audit", "TechQA 数据审计与分组划分",
     "复核固定输入、别名和组件划分。此队列是自定义有答案检索探索实验，不能称为官方 TechQA 测试集。",
     ["audit = result['data_audit']\ndisplay(pd.DataFrame(audit['splits']).T)\ndisplay({key: audit[key] for key in ['raw_documents','unique_documents','canonical_answerable_queries','independent_components','head_words','source_hashes']})",
      "queries = json.loads((artifact / 'split_manifest.json').read_text(encoding='utf-8'))\nframe = pd.DataFrame(queries)\nassert frame.groupby('group_id')['split'].nunique().max() == 1\nprint('Group-disjoint:', True)\nprint('Automated near-duplicate pairs:', len(audit['near_duplicate_pairs']))\nprint('Human review:', audit['semantic_duplicate_human_review'])"]),
    ("02_retrieval_baselines", "检索基线与候选召回",
     "BM25、语义后端和 RRF 使用结果清单中的实际参数。LSA 是语料拟合的潜在语义基线；只有 neural_encoder=true 才属于神经向量检索。",
     ["display(result['retrieval_metadata'])\ndisplay(pd.DataFrame(result['summary']['dev']['methods']).T.loc[['bm25','semantic','rrf']])\nprint('Union candidate recall:', result['summary']['dev']['candidate_recall'])",
      "display(Image(filename=str(artifact / 'development_metrics.png')))\nprint('Input heads:', result['data_audit']['head_words'], 'words; long-document evidence can be lost.')"]),
    ("03_features_and_rerankers", "重排特征与训练选择",
     "自主训练代码位于 src/ragstudy/experiment.py。每个训练折独立拟合标准化；全部候选随 query 所属组件进出验证折。未标注候选只是代理负例。",
     ["display(pd.DataFrame(result['cross_validation']))\ndisplay(pd.DataFrame(result['selected_models']).T[['parameters','training_pairs','cv_ndcg@10']])\nprint('Features:', result['feature_names'])",
      "from ragstudy.search import load_bundle\nbundle = load_bundle(artifact)\nmodel_bundle = bundle['models']['logistic']\nmodel = model_bundle['model']\nselected_features = [result['feature_names'][i] for i in model_bundle.get('feature_indices', range(len(result['feature_names'])))]\ndisplay(pd.DataFrame({'feature':selected_features, 'standardized_logistic_coefficient':model.coef_[0]}))\nprint('Scores support ranking; they are not calibrated relevance probabilities.')"]),
    ("04_development_evaluation", "开发集与冻结保留集评价",
     "指标以 query 宏平均，区间以共享 gold／近重复组件配对 bootstrap。开发区间是探索证据；保留集仅接受冻结模型、候选和本地预先记录协议的评价，不据其结果继续调参。语义审阅为 AI 辅助，实际成员尚未独立核验。",
     ["display(pd.DataFrame(result['summary']['dev']['methods']).T)\ndisplay(pd.DataFrame(result['summary']['dev']['paired_comparisons']).T[['difference','ci_lower','ci_upper','n_queries','n_groups']])\nprint('Final test evaluated:', result['final_test_evaluated'])\nif result['final_test_evaluated']:\n    assert result['frozen_evaluation']['no_new_training_or_indexing'] is True\n    display(pd.DataFrame(result['summary']['test']['methods']).T)\n    display(result['primary_holdout_contrast'])\nelse:\n    assert 'test' not in result['summary']",
      "verification_path = artifact / 'verification.json'\nif verification_path.is_file():\n    verification = json.loads(verification_path.read_text(encoding='utf-8'))\n    display(verification)\nelse:\n    print('Run scripts/verify_experiment.py for independent SQL/metric/candidate verification.')",
      "latency_path = artifact / 'latency_benchmark.json'\nif latency_path.is_file():\n    latency = json.loads(latency_path.read_text(encoding='utf-8'))\n    display(pd.DataFrame({name: values['total'] for name,values in latency['summary'].items()}).T)\n    print(latency['timing_boundary'])\nelse:\n    print('Repeated warm latency has not been measured for this run.')"]),
    ("05_citation_demo", "带原文引用的证据检索",
     "此演示返回固定语料中的文档摘录和可追溯 ID。排序分数不代表答案正确率；当前没有调用生成式模型。",
     ["from ragstudy.search import load_bundle, search\nfrom threadpoolctl import threadpool_limits\nwith threadpool_limits(limits=2):\n    evidence = search(load_bundle(artifact), 'How do I resolve an IBM WebSphere connection timeout?', method='logistic', limit=3)\ndisplay(pd.DataFrame(evidence['evidence']))\ndisplay(evidence['timing_ms'])"]),
]


def main():
    target = ROOT / "project/notebooks"
    target.mkdir(parents=True, exist_ok=True)
    for name, title, explanation, code_cells in NOTEBOOKS:
        book = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(f"# {title}\n\n{explanation}"),
                                        nbf.v4.new_code_cell(SETUP),
                                        *[nbf.v4.new_code_cell(code) for code in code_cells]])
        book.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nbf.write(book, target / (name + ".ipynb"))
        print(target / (name + ".ipynb"))


if __name__ == "__main__":
    main()
