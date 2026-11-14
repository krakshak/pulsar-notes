"""Streaming ASR Transcriber using parakeet-mlx (with mlx-whisper fallback).

Features:
- Enforces HF_HUB_OFFLINE=1 before model loading (zero outbound network calls).
- Consumes 16kHz mono float32 AudioChunks from AudioEngine in RAM.
- Tags segments with speaker labels ([Me] / [Others]) into an ephemeral list[str].
- Scrubs AudioChunk PCM arrays immediately after transcription.
- Exposes unload_asr_model() to delete model weights, gc.collect(), and mx.clear_cache()
  so Unified Memory is freed before Ollama summarization runs.
"""

import argparse
import gc
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Enforce offline flags BEFORE importing huggingface_hub / parakeet_mlx
from backend import config  # noqa: F401
from backend.audio_engine import AudioChunk, AudioEngine
from backend.config import DEFAULT_ASR_MODEL, FALLBACK_ASR_MODEL


def get_mlx_memory_mb() -> float:
    """Returns current MLX active + cache memory in megabytes (MB)."""
    if "mlx.core" not in sys.modules:
        return 0.0
    try:
        mx = sys.modules["mlx.core"]
        active = mx.get_active_memory()
        cache = mx.get_cache_memory()
        return round((active + cache) / (1024 * 1024), 2)
    except Exception:
        return 0.0


class StreamingTranscriber:
    """Manages local MLX ASR model lifecycle and ephemeral RAM transcript buffer."""

    def __init__(self, model_name: str = DEFAULT_ASR_MODEL):
        self.model_name = model_name
        self.model: Any = None
        self._backend: str = "whisper" if "whisper" in model_name.lower() else "parakeet"

        # Ephemeral RAM-only transcript storage (never written to disk)
        self.transcript_segments: List[str] = []
        self.latest_snippet: str = ""
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._on_segment: Optional[Callable[[str], None]] = None

    def load_model(self) -> None:
        """Loads the ASR model strictly from local HuggingFace cache."""
        config.enforce_offline_env()

        if self._backend == "whisper":
            # mlx_whisper loads lazily or via path_or_hf_repo; verify import works
            import mlx_whisper  # noqa: F401
            self.model = self.model_name
        else:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained
            self.model = from_pretrained(self.model_name)
            # Materialize lazy weight casts on the loading thread so worker threads
            # don't reference streams from another thread.
            mx.eval(self.model.parameters())

    def unload_asr_model(self) -> float:
        """Unloads ASR model from Unified Memory and clears MLX Metal cache."""
        self.stop_worker()

        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()

        if "mlx.core" in sys.modules:
            try:
                sys.modules["mlx.core"].clear_cache()
            except Exception:
                pass

        return get_mlx_memory_mb()

    def transcribe_chunk(self, chunk: AudioChunk) -> str:
        """Transcribes a single in-memory AudioChunk and appends to RAM buffer."""
        if chunk.data is None or chunk.data.size == 0:
            return ""

        text = ""
        try:
            if self._backend == "whisper":
                import mlx_whisper

                result = mlx_whisper.transcribe(
                    chunk.data,
                    path_or_hf_repo=self.model_name,
                )
                text = (result.get("text") or "").strip()
            else:
                if self.model is None:
                    raise RuntimeError("ASR model is not loaded. Call load_model() first.")

                import mlx.core as mx
                from parakeet_mlx.audio import get_logmel

                audio_mx = mx.array(chunk.data, dtype=mx.float32)
                min_samples = self.model.preprocessor_config.hop_length
                if len(audio_mx) < min_samples:
                    return ""

                mel = get_logmel(audio_mx, self.model.preprocessor_config)
                result = self.model.generate(mel)[0]
                text = (result.text or "").strip()
        finally:
            # Scrub the chunk's PCM buffer immediately after inference
            if isinstance(chunk.data, np.ndarray):
                chunk.data.fill(0.0)

        if text:
            formatted_line = f"{chunk.speaker_tag} {text}"
            with self._lock:
                self.transcript_segments.append(formatted_line)
                self.latest_snippet = formatted_line

            if self._on_segment is not None:
                try:
                    self._on_segment(formatted_line)
                except Exception:
                    pass

        return text

    def start_worker(
        self,
        audio_queue: queue.Queue,
        on_segment: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Starts background worker thread consuming AudioChunks from AudioEngine."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._on_segment = on_segment
        self._stop_event.clear()

        def _consume_loop():
            while not self._stop_event.is_set() or not audio_queue.empty():
                try:
                    chunk = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                try:
                    self.transcribe_chunk(chunk)
                except Exception as exc:
                    sys.stderr.write(f"Transcription error: {exc}\n")
                finally:
                    audio_queue.task_done()

        self._worker_thread = threading.Thread(
            target=_consume_loop,
            name="ASRWorkerThread",
            daemon=True,
        )
        self._worker_thread.start()

    def stop_worker(self, timeout: float = 5.0) -> None:
        """Signals the background worker to drain remaining chunks and exit."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None

    def get_transcript(self) -> List[str]:
        """Returns a thread-safe copy of the in-memory transcript segments."""
        with self._lock:
            return list(self.transcript_segments)

    def get_transcript_text(self) -> str:
        """Returns the full tagged transcript joined by newlines."""
        return "\n".join(self.get_transcript())

    def get_stats(self) -> Dict[str, Any]:
        """Returns word count, estimated tokens, and latest snippet from RAM."""
        with self._lock:
            full_text = " ".join(self.transcript_segments)
            word_count = len(full_text.split()) if full_text.strip() else 0
            estimated_tokens = int(round(word_count * 1.33))
            return {
                "word_count_in_ram": word_count,
                "estimated_tokens": estimated_tokens,
                "latest_snippet": self.latest_snippet,
                "segment_count": len(self.transcript_segments),
            }

    def purge_transcript(self) -> None:
        """Clears the ephemeral transcript list from RAM and triggers GC."""
        with self._lock:
            self.transcript_segments.clear()
            self.latest_snippet = ""
        gc.collect()


# ==============================================================================
# CLI Live Demo Mode (--live-demo)
# ==============================================================================

def run_live_demo(seconds: float = 10.0, model_name: str = DEFAULT_ASR_MODEL) -> None:
    """Runs an end-to-end offline live transcription smoke test."""
    print(f"=== Offline MLX Transcriber Live Demo ({seconds}s) ===")
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
    print(f"Loading ASR model: {model_name} ...")

    transcriber = StreamingTranscriber(model_name=model_name)
    transcriber.load_model()
    mem_loaded = get_mlx_memory_mb()
    print(f"Model loaded. MLX Memory: {mem_loaded} MB")

    engine = AudioEngine()
    transcriber.start_worker(
        engine.audio_queue,
        on_segment=lambda line: print(f"  {line}", flush=True),
    )
    engine.start()

    print("\nListening (speak into your mic or play system audio)...\n")
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        mem_after_unload = transcriber.unload_asr_model()
        transcriber.purge_transcript()
        remaining = len(transcriber.get_transcript())

        print("\n=== Live Demo Complete ===")
        print(f"MLX Memory after unload: {mem_after_unload} MB")
        print(f"Transcript buffer length after purge: {remaining}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline MLX Streaming Transcriber")
    parser.add_argument("--live-demo", action="store_true", help="Run live offline transcription demo")
    parser.add_argument("--seconds", type=float, default=10.0, help="Recording duration in seconds")
    parser.add_argument("--model", type=str, default=DEFAULT_ASR_MODEL, help="ASR model ID")
    args = parser.parse_args()

    if args.live_demo:
        run_live_demo(seconds=args.seconds, model_name=args.model)
    else:
        parser.print_help()

