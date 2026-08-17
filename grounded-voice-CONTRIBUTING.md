# Contributing to Real-Time Voice Assistant with Domain Grounding

Thanks for your interest. Issues and PRs are welcome — especially anything that improves latency, ASR accuracy, or extends the grounding/retrieval layer.

## Development setup

```bash
git clone https://github.com/Ritik650/grounded-voice.git
cd grounded-voice

python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py                 # Silero VAD + Piper voice, ~65 MB
```

For the Gemini backend during development: `pip install -r requirements-llm.txt` and set `GEMINI_API_KEY` in `.env`.

## Before opening a PR

```bash
pytest tests/ -v
ruff check app eval tests scripts
```

If your change touches a latency-sensitive path (VAD, ASR queuing, TTS pipelining, retrieval), run the relevant eval harness before and after, and include both numbers in your PR description:

```bash
python -m eval.latency_eval --runs 4
python -m eval.wer_eval --dataset librispeech --limit 300 --model base.en   # if ASR-related
```

This project's whole ethos is "nothing here is estimated" — every number in the README came from a harness in `eval/`. Keep that standard: don't hand-wave a latency claim, run the harness.

## Working on the streaming pipeline

- **Stay torch-free.** The project deliberately avoids PyTorch (faster-whisper on CTranslate2, Silero/Piper/embeddings on ONNX Runtime) to keep the image near 2 GB. A new dependency that pulls in torch needs a strong justification in the PR.
- **Barge-in is stateful, not just cancellation.** If you're touching `app/server.py`'s `barge_in` logic, read the [barge-in design note](README.md#three-things-that-were-not-obvious) first — cancelling the generator alone does not implement it, since TTS runs ~8× faster than real time and audio is already queued client-side.
- **VAD context window matters.** Silero's ONNX graph needs 576 samples (64 of context + 512 new), not a bare 512-sample window — see `app/vad.py` and the corresponding design note before changing frame sizing.

## Adding a new LLM backend

Follow the pattern of the existing `extractive`/`gemini`/`ollama` backends in `app/config.py`. New backends should:
- Fail gracefully to the extractive backend on error (rate limit, timeout, DNS), logging why, without killing the WebSocket session.
- Handle in-flight truncation correctly if tokens were already being spoken when the failure hit — don't restart with a different answer mid-sentence.
- Still refuse out-of-KB questions; grounding must survive the backend swap.

## Style

- Keep the per-stage timing instrumentation intact for any new pipeline stage — the `/metrics` endpoint and the eval harnesses depend on it, and it's what makes latency regressions visible instead of hidden inside an inflated neighboring stage.
- New tests should target the failure modes that are wrong in a naive implementation (see the existing test suite's philosophy in [Testing](README.md#testing)), not just the easy-to-assert happy path.

## Security issues

Please don't open a public issue for security concerns — see [SECURITY.md](SECURITY.md) instead.
