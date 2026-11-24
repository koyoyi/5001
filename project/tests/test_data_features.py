"""Small, offline regression tests for data leakage and ranking-row integrity."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import zipfile

import numpy as np
import pytest

from ragstudy.data import assign_splits, load_techqa
from ragstudy.features import FEATURE_NAMES, build_features, sample_training_rows


REVISION = "techqa-fixture-pinned-revision"
QUESTIONS = [
    "Amethyst orchard irrigation pressure",
    "Bronze satellite navigation calibration",
    "Crimson filesystem journal recovery",
    "Delta compiler vectorization warning",
    "Emerald database transaction deadlock",
    "Fuchsia network certificate renewal",
    "Golden display brightness oscillation",
    "Hazel container image synchronization",
]


def answerable(record_id, question, document_id, text):
    return {"id": record_id, "question": question, "is_impossible": False,
            "contexts": [{"filename": document_id, "text": text}]}


@pytest.fixture
def techqa_bundle(tmp_path):
    """Write real ZIP/JSON/checksum files with eight independent base queries."""
    def write(*, extra_documents=None, extra_rows=(), manifest_transform=None):
        source = tmp_path / "nvidia__TechQA-RAG-Eval"
        source.mkdir(exist_ok=True)
        documents = {f"d{i:02}.txt": f"Evidence for {question}. Unique reference {i}."
                     for i, question in enumerate(QUESTIONS)}
        rows = [answerable(f"q{i:02}", question, f"d{i:02}.txt", documents[f"d{i:02}.txt"])
                for i, question in enumerate(QUESTIONS)]
        documents.update(extra_documents or {})
        rows.extend(deepcopy(extra_rows))
        (source / "train.json").write_text(json.dumps(rows), encoding="utf-8")
        (source / "README.md").write_text("Offline TechQA fixture", encoding="utf-8")
        with zipfile.ZipFile(source / "corpus.zip", "w") as archive:
            for name, text in documents.items():
                archive.writestr("corpus/" + name, text)
        files = [{"repo": "nvidia/TechQA-RAG-Eval", "revision": REVISION,
                  "local_file": source.name + "/" + name,
                  "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()}
                 for name in ("README.md", "train.json", "corpus.zip")]
        if manifest_transform is not None:
            files = manifest_transform(files)
        (tmp_path / "download_manifest.json").write_text(
            json.dumps({"files": files}), encoding="utf-8")
        return source
    return write


@pytest.mark.parametrize("filename", ["train.json", "corpus.zip"])
def test_pinned_input_changes_are_rejected_before_loading(techqa_bundle, filename):
    source = techqa_bundle()
    with (source / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_techqa(source)


def test_manifest_needs_only_the_two_selected_input_entries(techqa_bundle):
    source = techqa_bundle(manifest_transform=lambda entries: entries[1:])
    dataset = load_techqa(source)
    assert dataset["audit"]["revision"] == REVISION
    assert dataset["audit"]["canonical_answerable_queries"] == 8


def test_revision_comes_from_selected_inputs_not_manifest_position(techqa_bundle):
    def unrelated_third_entry(entries):
        unrelated = {"repo": "unrelated/dataset", "revision": "wrong-dataset-revision",
                     "local_file": "other/readme.md", "sha256": "unused"}
        return [entries[2], entries[1], unrelated, entries[0]]
    source = techqa_bundle(manifest_transform=unrelated_third_entry)
    assert load_techqa(source)["audit"]["revision"] == REVISION


def test_manifest_ignores_same_basename_from_another_source_directory(techqa_bundle):
    def prepend_unrelated_input(entries):
        unrelated = dict(entries[1], local_file="another-dataset/train.json", sha256="wrong-hash",
                         revision="unrelated-revision", repo="unrelated/dataset")
        return [unrelated, *entries]
    source = techqa_bundle(manifest_transform=prepend_unrelated_input)
    assert load_techqa(source)["audit"]["revision"] == REVISION


@pytest.mark.parametrize("field,value", [("revision", "different-revision"),
                                        ("repo", "unrelated/dataset")])
def test_selected_train_and_corpus_must_have_consistent_provenance(techqa_bundle, field, value):
    def conflict(entries):
        entries[2][field] = value
        return entries
    with pytest.raises(ValueError):
        load_techqa(techqa_bundle(manifest_transform=conflict))


@pytest.mark.parametrize("head_words", [0, -1, 1.5])
def test_head_word_limit_must_be_a_positive_integer(techqa_bundle, head_words):
    with pytest.raises(ValueError):
        load_techqa(techqa_bundle(), head_words=head_words)


@pytest.mark.parametrize("threshold", [0, -.1, 1.1, np.nan])
def test_near_duplicate_threshold_must_be_finite_and_in_unit_interval(techqa_bundle, threshold):
    with pytest.raises(ValueError):
        load_techqa(techqa_bundle(), near_duplicate_threshold=threshold)


@pytest.mark.parametrize("document_id,text,error", [
    ("d00.txt", "A conflicting passage from another document.", "Reference text mismatch"),
    ("missing.txt", "A missing passage.", "Missing reference document"),
])
def test_positive_context_must_match_the_original_corpus(
        techqa_bundle, document_id, text, error):
    row = answerable("bad-reference", "Indigo printer authentication issue", document_id, text)
    with pytest.raises(ValueError, match=error):
        load_techqa(techqa_bundle(extra_rows=[row]))


def test_mixed_answerability_is_excluded_as_a_whole_query(techqa_bundle):
    question = "Why does the violet gateway discard packets?"
    text = "Violet gateway routing evidence."
    positive = answerable("mixed-positive", question, "violet.txt", text)
    impossible = {"id": "mixed-impossible", "question": "  " + question.upper() + "  ",
                  "is_impossible": True, "contexts": []}
    unanswerable = {"id": "unanswerable-only", "question": "Silver keyboard wake behavior",
                    "is_impossible": True, "contexts": []}
    dataset = load_techqa(techqa_bundle(
        extra_documents={"violet.txt": text}, extra_rows=[positive, impossible, unanswerable]))
    assert len(dataset["queries"]) == 8
    assert dataset["audit"]["excluded_conflicts"] == [
        {"reason": "mixed_answerability", "ids": ["mixed-impossible", "mixed-positive"]}]
    assert all(not set(query["source_record_ids"]).intersection(
        {"mixed-impossible", "mixed-positive", "unanswerable-only"})
        for query in dataset["queries"])


def test_duplicate_documents_and_question_rows_share_canonical_gold(techqa_bundle):
    original = "Evidence for " + QUESTIONS[0] + ". Unique reference 0."
    duplicate = answerable("q00-copy", "  AMETHYST   ORCHARD IRRIGATION PRESSURE  ",
                           "zz-copy.txt", "\n" + original + "\n")
    dataset = load_techqa(techqa_bundle(extra_documents={"zz-copy.txt": "\n" + original + "\n"},
                                      extra_rows=[duplicate]))
    query = next(query for query in dataset["queries"] if query["query_id"] == "q00")
    assert query["source_record_ids"] == ["q00", "q00-copy"]
    assert query["positive_document_ids"] == ["d00.txt"]
    assert dataset["audit"]["raw_documents"] == 9
    assert dataset["audit"]["unique_documents"] == 8
    assert dataset["audit"]["duplicate_aliases"] == {"zz-copy.txt": "d00.txt"}
    assert "zz-copy.txt" not in dataset["document_ids"]


def test_shared_gold_and_near_duplicate_links_form_one_unsplit_component(techqa_bundle):
    original = "Evidence for " + QUESTIONS[0] + ". Unique reference 0."
    near_question = (
        "Why does hypervisor connectivity validation reject the certificate following host migration")
    rows = [answerable("shared-gold", "Saffron storage controller recovery", "d00.txt", original),
            answerable("near-a", near_question + "?", "d00.txt", original),
            answerable("near-b", near_question + "!", "near-b.txt", "A different reference.")]
    source = techqa_bundle(extra_documents={"near-b.txt": "A different reference."}, extra_rows=rows)
    dataset = load_techqa(source)
    linked = [query for query in dataset["queries"]
              if query["query_id"] in {"q00", "shared-gold", "near-a", "near-b"}]
    assert len(linked) == 4
    assert len({query["group_id"] for query in linked}) == 1
    assert len({query["split"] for query in linked}) == 1
    assert any({pair["left"], pair["right"]} == {"near-a", "near-b"}
               for pair in dataset["audit"]["near_duplicate_pairs"])
    component_splits, gold_splits = defaultdict(set), defaultdict(set)
    for query in dataset["queries"]:
        component_splits[query["group_id"]].add(query["split"])
        for document_id in query["positive_document_ids"]:
            gold_splits[document_id].add(query["split"])
    assert all(len(splits) == 1 for splits in component_splits.values())
    assert all(len(splits) == 1 for splits in gold_splits.values())
    assert set(query["split"] for query in dataset["queries"]) == {"train", "dev", "test"}
    assert load_techqa(source)["queries"] == dataset["queries"]


def test_head_only_retrieval_text_keeps_original_reference_intact(techqa_bundle):
    source = techqa_bundle()
    dataset = load_techqa(source, head_words=3)
    for document_id, retrieval_text in zip(dataset["document_ids"], dataset["documents"]):
        assert retrieval_text == " ".join(dataset["originals"][document_id].split()[:3])
        assert len(dataset["originals"][document_id].split()) > 3


def test_fewer_than_six_independent_components_is_rejected():
    queries = [{"query_id": str(i), "group_id": "one-shared-gold-component"} for i in range(10)]
    with pytest.raises(ValueError, match="six independent components"):
        assign_splits(queries)


def training_record(query_id, markers, labels, group_id="shared-component"):
    features = np.zeros((len(markers), len(FEATURE_NAMES)), dtype=float)
    features[:, 0] = markers
    return {"query": {"query_id": query_id, "group_id": group_id,
                      "positive_document_ids": ["unretrieved-gold"]},
            "candidate_ids": list(markers), "features": features,
            "labels": np.asarray(labels, dtype=int)}


def test_negative_sampling_preserves_positives_and_equal_total_query_weight():
    first = training_record("query-small", [101, 102], [1, 0])
    second = training_record("query-large", [201, 202, 203, 204, 205, 206], [1, 1, 0, 0, 0, 0])
    records = [first, second]
    original_features = [record["features"].copy() for record in records]
    features, labels, weights, groups, missed = sample_training_rows(records, max_negatives=3, seed=7)
    assert set(features[labels == 1, 0]) == {101, 201, 202}
    assert len(features[(features[:, 0] >= 200) & (labels == 0)]) == 3
    assert len(set(features[:, 0])) == len(features)
    assert set(features[:, 0]).issubset({101, 102, 201, 202, 203, 204, 205, 206})
    assert np.isclose(weights[features[:, 0] < 200].sum(),
                      weights[features[:, 0] >= 200].sum())
    assert np.isclose(weights.mean(), 1)
    assert set(groups) == {"shared-component"}
    assert missed == []
    for record, original in zip(records, original_features):
        np.testing.assert_array_equal(record["features"], original)
    repeated = sample_training_rows(records, max_negatives=3, seed=7)
    for expected, actual in zip((features, labels, weights, groups), repeated[:4]):
        np.testing.assert_array_equal(actual, expected)


def test_query_with_no_retrieved_positive_is_recorded_without_gold_injection():
    missed_record = training_record("missed-positive-query", [901, 902], [0, 0])
    usable = training_record("usable-query", [101, 102], [1, 0])
    records = [missed_record, usable]
    features, labels, _, _, missed = sample_training_rows(records, max_negatives=1)
    assert missed == ["missed-positive-query"]
    assert set(features[:, 0]) == {101, 102}
    assert labels.tolist() == [1, 0]
    assert records[0] is missed_record
    assert missed_record["candidate_ids"] == [901, 902]
    assert missed_record["labels"].tolist() == [0, 0]


def test_training_refuses_a_dataset_without_any_usable_positive_negative_pair():
    with pytest.raises(ValueError, match="both a retrieved positive and a proxy negative"):
        sample_training_rows([training_record("all-missed", [901, 902], [0, 0])])


def test_features_distinguish_missing_rank_from_a_retrieved_zero_score():
    documents = ["cache", "cache server logfile", "unrelated"]
    features = build_features("cache cache server", [2, 0, 1], documents,
                              np.asarray([0, 0, .4]), np.asarray([.2, .5, 0]),
                              [2, 1], [0, 1])
    columns = {name: index for index, name in enumerate(FEATURE_NAMES)}
    assert features.shape == (3, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    # Candidate order remains [2, 0, 1]; document 0 is absent from BM25,
    # whereas document 1 has a recorded second rank despite its zero score.
    assert features[1, columns["bm25_score"]] == features[2, columns["bm25_score"]] == 0
    assert features[1, columns["missing_bm25_rank"]] == 1
    assert features[2, columns["missing_bm25_rank"]] == 0
    assert features[1, columns["bm25_reciprocal_rank"]] == 0
    assert features[2, columns["bm25_reciprocal_rank"]] == .5
    assert features[0, columns["missing_semantic_rank"]] == 1
    np.testing.assert_allclose(features[:, columns["query_term_coverage"]], [0, .5, 1])


def test_empty_candidate_set_has_a_well_defined_feature_shape():
    features = build_features("cache", [], ["cache"], np.asarray([1]), np.asarray([1]), [0], [0])
    assert features.shape == (0, len(FEATURE_NAMES))
