# Security Policy

## Scope

This service accepts live audio over an unauthenticated WebSocket, optionally calls an external LLM API (Gemini), and can persist vectors to Qdrant. Security reports are welcome, particularly for:

- **The WebSocket endpoint** — no authentication is implemented by default; this is a single-worker demo/reference implementation, not a multi-tenant production service. If you deploy this publicly, put it behind your own auth layer first.
- **Audio input handling** — anything in the VAD/ASR path that could be exploited via a malformed or adversarial audio stream.
- **Secrets handling** — `GEMINI_API_KEY` or other `.env` values appearing somewhere they shouldn't (logs, error messages, client-visible responses).
- **Dependency vulnerabilities** in `requirements.txt` / `requirements-llm.txt`.

## Known scope limitations

- There is no built-in authentication or rate limiting on the WebSocket or `/chat` endpoints. Conversation memory is session-scoped (per WebSocket connection) and not persisted, but nothing stops an unauthenticated client from opening a session.
- `GEMINI_API_KEY` is read from environment variables. Never commit a populated `.env` — `.env.example` should be the only version-controlled copy.
- The knowledge base in `data/kb/` is intentionally fictional demo content; if you point this at real documents, treat the retrieval/grounding layer as exposing whatever's in that folder to anyone who can reach the WebSocket.

## Supported Versions

Only the latest commit on `main` is supported.

| Version | Supported |
|---|---|
| `main` (latest) | ✅ |
| Older commits | ❌ |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately via one of:
- GitHub's [private vulnerability reporting](https://github.com/Ritik650/grounded-voice/security/advisories/new) (Security tab → Report a vulnerability)
- Email: ry9812262@gmail.com

Please include a description, reproduction steps, and potential impact. This is a solo-maintained project, so response times aren't guaranteed, but reports will be acknowledged and addressed as soon as possible.
