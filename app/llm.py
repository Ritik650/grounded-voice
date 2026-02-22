"""Answer generation, grounded in retrieved context.

Three interchangeable backends behind one interface:

  extractive -- no API key, no network, deterministic. The default, so the pipeline,
                the tests and CI all run out of the box and the eval numbers are
                reproducible on any machine.
  gemini     -- hosted, best quality, adds network RTT to the critical path.
  ollama     -- local model, keeps everything offline.

All three stream. In a voice pipeline that matters more than raw quality: TTS can
start speaking the first sentence while the model is still writing the second, which
removes most of the generation time from time-to-first-audio.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from functools import cache

from .config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a voice assistant. You are speaking out loud, so:\n"
    "- Answer ONLY from the provided context. If the context does not contain the "
    "answer, say you don't have that information. Never guess.\n"
    f"- Be brief: at most {settings.llm_max_words} words, ideally one or two sentences.\n"
    "- Write plain spoken prose. No markdown, no bullet points, no numbered lists, "
    "no emoji, no citation brackets -- all of it gets read aloud literally.\n"
    "- Expand abbreviations and symbols into words a person would say."
)

NO_CONTEXT_REPLY = "I don't have that in my knowledge base."

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9']+")
_CONTEXT_BLOCK = re.compile(r"^\[(\d+)\]\s*\(([^)]*)\)\s*", re.MULTILINE)

# Deliberately broad. Spoken questions are full of conversational filler ("actually,
# what are your support hours") and every filler word that survives into the term set
# is a chance for an unrelated sentence to look relevant.
STOPWORDS = {
    "a",
    "about",
    "actually",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "get",
    "gets",
    "had",
    "has",
    "have",
    "here",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "many",
    "may",
    "me",
    "might",
    "more",
    "most",
    "much",
    "must",
    "my",
    "need",
    "not",
    "of",
    "on",
    "only",
    "or",
    "our",
    "out",
    "over",
    "per",
    "please",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "very",
    "want",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def build_prompt(question: str, context: str) -> str:
    context_block = context.strip() or "(no relevant documents were retrieved)"
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context_block}\n\nQuestion: {question}\nSpoken answer:"


class ExtractiveBackend:
    """Ranks context sentences against the question and returns the best ones.

    Not a language model and does not pretend to be one -- it cannot paraphrase or
    combine facts. What it can do is never hallucinate (every word it returns came
    from the KB) and never vary between runs, which is exactly what you want holding
    the default slot in a latency benchmark.
    """

    name = "extractive"

    def stream(self, question: str, context: str) -> Iterator[str]:
        yield self.complete(question, context)

    def complete(self, question: str, context: str) -> str:
        if not context.strip():
            return NO_CONTEXT_REPLY

        q_terms = _content_words(question)
        if not q_terms:
            return NO_CONTEXT_REPLY

        scored: list[tuple[float, str]] = []
        for rank, block in enumerate(_context_blocks(context)):
            # Trust the retriever's ordering. Hybrid search already judged which
            # passage answers the question using semantics this backend has no access
            # to; lexical overlap alone happily prefers a keyword-dense sentence from
            # the fourth-ranked chunk over the right one from the first.
            rank_weight = 1.0 / (1.0 + 0.35 * rank)
            for raw in _split_sentences(block):
                sentence = raw.strip()
                # Short fragments are almost always markdown headings, which read as a
                # non-sequitur when spoken ("Late payments. Accounts more than...").
                if len(sentence) < 20:
                    continue
                terms = _content_words(sentence)
                if not terms:
                    continue
                overlap = len(q_terms & terms)
                if overlap:
                    # Normalise by sentence length so a long paragraph doesn't win on
                    # raw term count alone.
                    scored.append((rank_weight * overlap / (len(terms) ** 0.5), sentence))

        if not scored:
            return NO_CONTEXT_REPLY

        scored.sort(key=lambda s: -s[0])
        reply, words = [], 0
        covered: set[str] = set()

        for _score, sentence in scored[:4]:
            terms = _content_words(sentence)
            # A second sentence earns its place only by answering part of the question
            # the first one didn't. Ranking by score alone picks near-duplicates -- for
            # "how much is the gigabit plan" the Starter plan's sentence scores almost
            # as high, and appending it doubled the spoken reply to 12 seconds while
            # adding nothing. Spoken replies pay for every redundant word.
            new_terms = (q_terms & terms) - covered
            if reply and not new_terms:
                continue
            n = len(sentence.split())
            if words + n > settings.llm_max_words and reply:
                break
            reply.append(sentence)
            covered |= q_terms & terms
            words += n
            if len(reply) == 2:
                break
        return " ".join(reply)


class GeminiBackend:
    name = "gemini"

    def __init__(self):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "google-genai is not installed. `pip install -r requirements-llm.txt` "
                "or set LLM_BACKEND=extractive."
            ) from exc
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (LLM_BACKEND=gemini).")
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def stream(self, question: str, context: str) -> Iterator[str]:
        stream = self._client.models.generate_content_stream(
            model=settings.gemini_model, contents=build_prompt(question, context)
        )
        for event in stream:
            if event.text:
                yield event.text

    def complete(self, question: str, context: str) -> str:
        return "".join(self.stream(question, context)).strip()


class OllamaBackend:
    name = "ollama"

    def __init__(self):
        import httpx

        self._client = httpx.Client(base_url=settings.ollama_host, timeout=120.0)

    def stream(self, question: str, context: str) -> Iterator[str]:
        import json

        payload = {
            "model": settings.ollama_model,
            "prompt": build_prompt(question, context),
            "stream": True,
            "options": {"temperature": 0.2, "num_predict": settings.llm_max_words * 3},
        }
        with self._client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("response"):
                    yield chunk["response"]

    def complete(self, question: str, context: str) -> str:
        return "".join(self.stream(question, context)).strip()


BACKENDS = {
    "extractive": ExtractiveBackend,
    "gemini": GeminiBackend,
    "ollama": OllamaBackend,
}


@cache
def get_llm(backend: str | None = None):
    name = (backend or settings.llm_backend).lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown LLM backend {name!r}. Options: {sorted(BACKENDS)}")
    try:
        return BACKENDS[name]()
    except Exception as exc:
        if name == "extractive":
            raise
        # A missing key shouldn't take the whole service down mid-conversation --
        # degrade to grounded extraction and say so loudly in the logs.
        log.warning("LLM backend %r unavailable (%s); falling back to extractive.", name, exc)
        return ExtractiveBackend()


def answer(question: str, context: str, backend: str | None = None) -> str:
    return speakable(get_llm(backend).complete(question, context))


def stream_sentences(question: str, context: str, backend: str | None = None) -> Iterator[str]:
    """Regroup the token stream into whole sentences.

    TTS quality collapses if you synthesize token fragments -- prosody is decided per
    utterance, so "The refund" / " window is" / " 30 days" spoken as three clips sounds
    like three unrelated statements. Sentences are the smallest unit that still sounds
    natural, and they arrive fast enough to keep the latency win.
    """
    buffer = ""
    for token in get_llm(backend).stream(question, context):
        buffer += token
        while True:
            match = _SENTENCE_END.search(buffer)
            if not match:
                break
            sentence, buffer = buffer[: match.start()], buffer[match.end() :]
            cleaned = speakable(sentence)
            if cleaned:
                yield cleaned
    tail = speakable(buffer)
    if tail:
        yield tail


def speakable(text: str) -> str:
    """Strip anything that would be read aloud as punctuation noise."""
    text = re.sub(r"[*_`#]+", "", text)  # markdown emphasis / headings
    text = re.sub(r"\[\d+\]", "", text)  # citation markers
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)  # bullets
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    """Split into sentences, paragraph by paragraph.

    Two failure modes to avoid, and fixing one naively causes the other:

    * Splitting only on `[.!?]` glues markdown headings onto the sentence below them,
      because headings have no terminal punctuation ("Late payments Accounts more
      than 14 days overdue...").
    * Splitting on every newline shreds hard-wrapped prose, which is how markdown is
      normally written. That silently truncated a real answer here at "...500 megabits
      upload for", dropping the price.

    Paragraphs are the right unit: blank lines separate headings from body text, while
    single newlines inside a paragraph are just wrapping and get unwrapped to spaces.
    """
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        unwrapped = " ".join(paragraph.split())
        if unwrapped:
            sentences.extend(_SENTENCE_END.split(unwrapped))
    return sentences


def _context_blocks(context: str) -> list[str]:
    """Split a rendered context back into per-chunk blocks, best-ranked first.

    `KnowledgeBase.context_for` formats each chunk as "[n] (source) text", so the
    retriever's ranking survives into the prompt string. Parsing it back is mildly
    inelegant, but it keeps one `(question, context)` interface across all three
    backends instead of giving the extractive one a privileged structured input.
    """
    matches = list(_CONTEXT_BLOCK.finditer(context))
    if not matches:
        return [context]
    bounds = [m.end() for m in matches] + [len(context)]
    starts = [m.start() for m in matches][1:] + [len(context)]
    return [context[bounds[i] : starts[i]].strip() for i in range(len(matches))]


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}
