from __future__ import annotations

from app import llm

CONTEXT = (
    "[1] (plans.md) The Gigabit plan costs eighty dollars a month and includes "
    "unlimited data.\n\n"
    "[2] (support.md) Support is open from 7 a.m. to 11 p.m. local time."
)


def test_answers_from_context():
    reply = llm.answer("how much is the gigabit plan", CONTEXT, backend="extractive")
    assert "eighty dollars" in reply


def test_refuses_when_context_is_empty():
    assert llm.answer("how much is the gigabit plan", "", backend="extractive") == (
        llm.NO_CONTEXT_REPLY
    )


def test_refuses_when_context_is_irrelevant():
    reply = llm.answer("who won the world cup", CONTEXT, backend="extractive")
    assert reply == llm.NO_CONTEXT_REPLY


def test_answer_never_invents_words_outside_the_context():
    """The extractive backend's one hard guarantee."""
    reply = llm.answer("gigabit plan cost", CONTEXT, backend="extractive")
    context_words = set(CONTEXT.lower().replace("(", " ").replace(")", " ").split())
    assert all(w.lower() in context_words for w in reply.split())


def test_answer_stays_within_the_word_budget():
    long_context = " ".join(f"Fact number {i} concerns the gigabit plan." for i in range(50))
    reply = llm.answer("gigabit plan", long_context, backend="extractive")
    assert len(reply.split()) <= llm.settings.llm_max_words + 15


def test_speakable_strips_markdown_and_citations():
    assert llm.speakable("**Bold** text [1] with `code`") == "Bold text with code"
    assert llm.speakable("- a bullet") == "a bullet"


def test_hard_wrapped_sentences_are_not_truncated():
    """Markdown wraps prose at ~90 columns; a sentence spanning two lines is still one
    sentence. Getting this wrong silently dropped the price from a real answer."""
    context = (
        "The Gigabit plan provides 1000 megabits per second download and 500 megabits\n"
        "upload for eighty dollars a month. It includes a mesh router."
    )
    reply = llm.answer("how much is the gigabit plan", context, backend="extractive")
    assert "eighty dollars" in reply


def test_headings_do_not_leak_into_answers():
    context = "## Late payments\n\nAccounts more than 14 days overdue are charged five dollars."
    reply = llm.answer("what is the late fee", context, backend="extractive")
    assert not reply.startswith("Late payments")


def test_stream_sentences_groups_tokens_into_sentences(monkeypatch):
    class FakeBackend:
        name = "fake"

        def stream(self, question, context):
            yield from ["The refund ", "window is 30 days. ", "Equipment ", "is extra."]

        def complete(self, question, context):
            return "".join(self.stream(question, context))

    monkeypatch.setitem(llm.BACKENDS, "fake", FakeBackend)
    llm.get_llm.cache_clear()

    sentences = list(llm.stream_sentences("q", "ctx", backend="fake"))
    assert sentences == ["The refund window is 30 days.", "Equipment is extra."]
    llm.get_llm.cache_clear()


def test_unknown_backend_is_rejected():
    llm.get_llm.cache_clear()
    try:
        llm.get_llm("nonexistent")
    except ValueError as exc:
        assert "nonexistent" in str(exc)
    else:
        raise AssertionError("expected ValueError")
