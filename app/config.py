"""Central settings. Everything tunable lives here so the eval harness can sweep
model sizes / thresholds without touching module code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Audio ---------------------------------------------------------------
    # 16 kHz mono s16le is the lowest common denominator: Whisper, Silero VAD and
    # Piper all expect it, so the wire format never needs converting mid-pipeline.
    sample_rate: int = 16_000
    frame_ms: int = 32  # 512 samples @ 16 kHz -- the window size Silero v5+ requires

    # --- ASR -----------------------------------------------------------------
    asr_model_size: str = "base.en"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_beam_size: int = 1  # greedy; beam=5 roughly doubles latency for ~1% WER
    asr_language: str = "en"

    # --- VAD -----------------------------------------------------------------
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250  # ignore lip smacks / door clicks
    vad_min_silence_ms: int = 600  # end-of-utterance hangover
    vad_speech_pad_ms: int = 200  # pre-roll kept before speech onset
    vad_max_utterance_ms: int = 20_000  # hard cut so a hot mic can't buffer forever

    # --- Retrieval -----------------------------------------------------------
    qdrant_url: str = ":memory:"  # ":memory:" for local/CI, http://qdrant:6333 in compose
    qdrant_collection: str = "kb"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    retrieval_top_k: int = 4
    chunk_chars: int = 700
    chunk_overlap: int = 120

    # --- LLM -----------------------------------------------------------------
    # "extractive" needs no API key and no network -- it is the default so the
    # pipeline, the tests and CI all run out of the box.
    llm_backend: str = "extractive"  # extractive | gemini | ollama
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    llm_max_words: int = 60  # long replies destroy perceived latency via TTS

    # --- TTS -----------------------------------------------------------------
    tts_engine: str = "auto"  # auto | piper | null
    piper_voice: str = "en_US-lessac-medium"
    tts_chunk_ms: int = 200  # size of audio frames streamed back to the browser
    tts_length_scale: float = 1.0  # <1 speaks faster, shortening time-to-last-audio

    # --- Paths ---------------------------------------------------------------
    models_dir: Path = MODELS_DIR
    kb_dir: Path = DATA_DIR / "kb"
    samples_dir: Path = DATA_DIR / "samples"

    @property
    def vad_model_path(self) -> Path:
        return self.models_dir / "silero_vad.onnx"

    @property
    def piper_model_path(self) -> Path:
        return self.models_dir / "piper" / f"{self.piper_voice}.onnx"

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


settings = Settings()
