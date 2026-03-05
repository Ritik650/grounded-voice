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


class _Boom:
    """A backend that fails the way a rate-limited hosted API fails."""

    name = "boom"

    def __init__(self, code=429, after_tokens=0):
        self.code = code
        self.after_tokens = after_tokens

    def stream(self, question, context):
        for i in range(self.after_tokens):
            yield f"token{i} "
        raise RuntimeError(f"{self.code} RESOURCE_EXHAUSTED: quota exceeded")

    def complete(self, question, context):
        return "".join(self.stream(question, context))


def test_rate_limited_backend_falls_back_to_extractive(monkeypatch, caplog):
    """A 429 must degrade the turn, not kill the session."""
    monkeypatch.setitem(llm.BACKENDS, "boom", _Boom)
    llm.get_llm.cache_clear()

    with caplog.at_level("WARNING"):
        sentences = list(llm.stream_sentences("gigabit plan cost", CONTEXT, backend="boom"))

    assert sentences, "no fallback answer produced"
    assert "eighty dollars" in " ".join(sentences)
    assert "rate limited (429)" in caplog.text
    llm.get_llm.cache_clear()


def test_network_error_falls_back_to_extractive(monkeypatch):
    class Offline(_Boom):
        def stream(self, question, context):
            raise ConnectionError("getaddrinfo failed")
            yield  # pragma: no cover - generator marker

    monkeypatch.setitem(llm.BACKENDS, "offline", Offline)
    llm.get_llm.cache_clear()

    assert "eighty dollars" in llm.answer("gigabit plan cost", CONTEXT, backend="offline")
    llm.get_llm.cache_clear()


def test_failure_after_speaking_starts_does_not_restart_the_answer(monkeypatch):
    """Once audio is playing, a substituted answer would contradict what was said."""
    monkeypatch.setitem(llm.BACKENDS, "midfail", lambda: _Boom(after_tokens=3))
    llm.get_llm.cache_clear()

    sentences = list(llm.stream_sentences("gigabit plan cost", CONTEXT, backend="midfail"))

    assert "eighty dollars" not in " ".join(sentences)
    assert "token0" in " ".join(sentences)
    llm.get_llm.cache_clear()


def test_extractive_failures_are_not_swallowed(monkeypatch):
    """The fallback must not mask a genuine bug in the fallback itself."""
    monkeypatch.setattr(
        llm.ExtractiveBackend, "complete", lambda *a, **k: (_ for _ in ()).throw(ValueError("bug"))
    )
    llm.get_llm.cache_clear()
    try:
        llm.answer("q", CONTEXT, backend="extractive")
    except ValueError:
        pass
    else:
        raise AssertionError("expected the extractive error to propagate")
    llm.get_llm.cache_clear()


def test_status_code_extraction():
    class Err(Exception):
        code = 429

    assert llm._status_code(Err()) == 429
    assert llm._is_bad_request(Exception("400 INVALID_ARGUMENT thinking_budget"))
    assert not llm._is_bad_request(Exception("429 RESOURCE_EXHAUSTED"))


def test_unknown_backend_is_rejected():
    llm.get_llm.cache_clear()
    try:
        llm.get_llm("nonexistent")
    except ValueError as exc:
        assert "nonexistent" in str(exc)
    else:
        raise AssertionError("expected ValueError")
