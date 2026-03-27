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

        def stream(self, question, context, history=None):
            yield from ["The refund ", "window is 30 days. ", "Equipment ", "is extra."]

        def complete(self, question, context, history=None):
            return "".join(self.stream(question, context))

    monkeypatch.setitem(llm.BACKENDS, "fake", FakeBackend)
    llm.get_llm.cache_clear()

    sentences = list(llm.stream_sentences("q", "ctx", backend="fake"))
    assert sentences == ["The refund window is 30 days.", "Equipment is extra."]
    llm.get_llm.cache_clear()


class _PromptSpy:
    """Captures the prompt a backend was handed, so we can assert on what the model
    would actually see rather than on the answer it happened to produce."""

    name = "spy"
    last_prompt = ""

    def stream(self, question, context, history=None):
        type(self).last_prompt = llm.build_prompt(question, context, history)
        yield "Eighty dollars a month."

    def complete(self, question, context, history=None):
        return "".join(self.stream(question, context, history))


def test_followup_prompt_carries_the_previous_turn(monkeypatch):
    """The core memory requirement: a pronoun follow-up must reach the model with the
    turn that gives it a referent."""
    monkeypatch.setitem(llm.BACKENDS, "spy", _PromptSpy)
    llm.get_llm.cache_clear()

    history = [llm.Turn(user="Tell me about the Gigabit plan", assistant="It is our fastest tier.")]
    list(
        llm.stream_sentences(
            "and how much does that cost?", CONTEXT, backend="spy", history=history
        )
    )

    prompt = _PromptSpy.last_prompt
    assert "Tell me about the Gigabit plan" in prompt
    assert "It is our fastest tier." in prompt
    # The follow-up itself and the retrieved context must still be present.
    assert "and how much does that cost?" in prompt
    assert "eighty dollars" in prompt
    # And history must be framed as reference-resolution only, not as fact source.
    assert "not a source" in prompt
    llm.get_llm.cache_clear()


def test_prompt_has_no_history_section_when_memory_is_empty(monkeypatch):
    monkeypatch.setitem(llm.BACKENDS, "spy", _PromptSpy)
    llm.get_llm.cache_clear()

    llm.answer("how much is the gigabit plan", CONTEXT, backend="spy", history=[])

    assert llm.HISTORY_PREAMBLE not in _PromptSpy.last_prompt
    llm.get_llm.cache_clear()


def test_history_is_length_capped():
    long_turn = llm.Turn(user="q " * 500, assistant="a " * 500)
    rendered = llm.format_history([long_turn], max_chars=100)
    assert len(rendered) < 300
    assert rendered.endswith("...")


def test_extractive_ignores_history_without_breaking():
    history = [llm.Turn(user="something unrelated", assistant="entirely unrelated")]
    reply = llm.answer(
        "how much is the gigabit plan", CONTEXT, backend="extractive", history=history
    )
    assert "eighty dollars" in reply


class _Boom:
    """A backend that fails the way a rate-limited hosted API fails."""

    name = "boom"

    def __init__(self, code=429, after_tokens=0):
        self.code = code
        self.after_tokens = after_tokens

    def stream(self, question, context, history=None):
        for i in range(self.after_tokens):
            yield f"token{i} "
        raise RuntimeError(f"{self.code} RESOURCE_EXHAUSTED: quota exceeded")

    def complete(self, question, context, history=None):
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
        def stream(self, question, context, history=None):
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
