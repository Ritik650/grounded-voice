from __future__ import annotations

from app.rag import chunk_document


def test_chunking_respects_paragraphs():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunk_document(text, "doc.md", size=40, overlap=5)

    assert all(c.source == "doc.md" for c in chunks)
    assert all(len(c.text) <= 40 for c in chunks)
    assert "First paragraph" in chunks[0].text


def test_oversized_paragraph_is_split_with_overlap():
    chunks = chunk_document("x" * 250, "doc.md", size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_empty_document_yields_nothing():
    assert chunk_document("   \n\n  ", "doc.md") == []


def test_retrieval_finds_the_relevant_passage(kb):
    chunks = kb.retrieve("how much is the gigabit plan", k=2)
    assert chunks
    assert "eighty dollars" in " ".join(c.text for c in chunks)


def test_retrieval_handles_conversational_phrasing(kb):
    """Dense retrieval should bridge ASR-style phrasing to documentation wording."""
    chunks = kb.retrieve("when can I call you people", k=3)
    assert "7 a.m." in " ".join(c.text for c in chunks)


def test_empty_query_returns_nothing(kb):
    assert kb.retrieve("   ") == []


def test_context_block_is_numbered_and_attributed(kb):
    context, chunks = kb.context_for("late fee")
    assert chunks
    assert "[1]" in context
    assert chunks[0].source in context


def test_vocabulary_hint_surfaces_proper_nouns(kb):
    assert "Gigabit" in kb.vocabulary_hint()


def test_debug_retrieval_logs_transcript_and_scores(kb, caplog, monkeypatch):
    """The diagnostic must show what was heard and what came back, with scores."""
    from app.config import settings
    from app.pipeline import log_retrieval

    monkeypatch.setattr(settings, "debug_retrieval", True)
    chunks = kb.retrieve("how much is the gigabit plan", k=2)

    with caplog.at_level("INFO", logger="app.pipeline"):
        log_retrieval("how much is the gigabit plan", chunks)

    assert "how much is the gigabit plan" in caplog.text
    assert "rrf=" in caplog.text
    assert "test.md" in caplog.text


def test_debug_retrieval_is_silent_when_disabled(kb, caplog, monkeypatch):
    from app.config import settings
    from app.pipeline import log_retrieval

    monkeypatch.setattr(settings, "debug_retrieval", False)
    with caplog.at_level("INFO", logger="app.pipeline"):
        log_retrieval("anything", kb.retrieve("gigabit", k=1))

    assert "[retrieval]" not in caplog.text


def test_debug_retrieval_reports_an_empty_result(caplog, monkeypatch):
    """An empty retrieval is the most important case to see, not the least."""
    from app.config import settings
    from app.pipeline import log_retrieval

    monkeypatch.setattr(settings, "debug_retrieval", True)
    with caplog.at_level("INFO", logger="app.pipeline"):
        log_retrieval("something unindexed", [])

    assert "nothing retrieved" in caplog.text
