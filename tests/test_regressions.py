"""Fast regression tests for retrieval priorities and benchmark scoring.

These tests use fake retrieval results and do not call an LLM or require API keys.
Run with: python -m pip install -e ".[dev]" && python -m pytest
"""
from scripts.benchmark import score_answer
from dpdp_rag import query


class FakeEmbedder:
    def encode(self, *_args, **_kwargs):
        return [[0.1, 0.2, 0.3]]


class FakeCollection:
    def count(self):
        return 20

    def get(self, where=None, include=None):
        if where == {"unit": "s. 2"}:
            return {
                "documents": [
                    "DPDP Act 2023, s. 2. Data Fiduciary and Data Processor definitions."
                ],
                "metadatas": [{
                    "source": "DPDP Act 2023",
                    "doc_type": "act",
                    "unit": "s. 2",
                    "part": 0,
                }],
            }
        return {"documents": [], "metadatas": []}


def hit(unit, text=None, source="DPDP Act 2023", doc_type="act"):
    return {
        "text": text or f"Statutory text for {unit}.",
        "meta": {
            "source": source,
            "doc_type": doc_type,
            "unit": unit,
            "part": 0,
        },
    }


def patch_retrieval(monkeypatch, ranked):
    monkeypatch.setattr(query, "_collection", lambda: FakeCollection())
    monkeypatch.setattr(query, "_get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(query, "_gpu_dense_search", lambda *_args: ranked)
    monkeypatch.setattr(query, "_bm25_search", lambda *_args, **_kwargs: ranked)
    monkeypatch.setattr(query, "_hybrid_rerank", lambda *_args: ranked)


def test_definition_question_prioritizes_section_2_and_keeps_section_8(monkeypatch):
    section_8 = hit("s. 8", "The Data Fiduciary remains responsible for compliance.")
    patch_retrieval(monkeypatch, [section_8])

    results = query.retrieve(
        "What is the difference between a Data Fiduciary and a Data Processor?",
        k=4,
    )

    assert results[0]["meta"]["unit"] == "s. 2"
    assert any(result["meta"]["unit"] == "s. 8" for result in results)


def test_section_8_duties_question_keeps_section_8_without_definition_override(monkeypatch):
    section_8 = hit("s. 8", "A Data Fiduciary must comply with the Act.")
    patch_retrieval(monkeypatch, [section_8])

    results = query.retrieve("What duties are imposed by Section 8?", k=4)

    assert results
    assert results[0]["meta"]["unit"] == "s. 8"


def test_expected_citation_is_counted_and_grounded():
    case = {
        "expect_units": ["s. 8"],
        "expect_facts": ["reasonable security safeguards"],
    }
    passage = hit("s. 8", "A Data Fiduciary shall take reasonable security safeguards.")
    scored = score_answer(
        case,
        "A Data Fiduciary shall take reasonable security safeguards (DPDP Act 2023, s. 8).",
        [passage],
    )

    assert scored["answer_ok"] is True
    assert scored["cites_expected"] is True
    assert scored["ungrounded_citations"] == []


def test_unanswerable_question_requires_a_decline():
    case = {"expect_abstain": True}
    answer = (
        "The provided corpus does not contain information about GDPR transfer rules. "
        "I cannot answer that from the available context."
    )

    scored = score_answer(case, answer, [])

    assert scored["abstained"] is True
    assert scored["answer_ok"] is True
