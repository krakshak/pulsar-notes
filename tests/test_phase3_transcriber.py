"""Automated Test Suite for Phase 3: parakeet-mlx Streaming Transcription & Unified Memory Unload.

Tests:
1. test_offline_model_load_blocks_network: Monkeypatches socket.socket.connect to block all network calls
   and verifies StreamingTranscriber loads from local cache.
2. test_speaker_tagging_and_ram_buffer: Feeds AudioChunks tagged [Me] and [Others], verifies speaker
   tagging in transcript_segments, PCM scrubbing on chunks, and purge_transcript() clearing RAM.
3. test_mlx_memory_unload: Loads the ASR model, verifies MLX memory > 0, calls unload_asr_model(),
   and verifies MLX memory drops back to near zero.
"""

import socket
from types import SimpleNamespace
import numpy as np
import pytest

from backend.audio_engine import AudioChunk
from backend.transcriber import StreamingTranscriber, get_mlx_memory_mb


def test_offline_model_load_blocks_network(monkeypatch):
    """Monkeypatches socket.socket.connect to raise an error on any outbound connection
    and verifies transcriber.py loads from local cache without touching the network.
    """
    def blocked_connect(*args, **kwargs):
        raise RuntimeError("NETWORK BLOCKED: Outbound connection attempted during offline model load!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    transcriber = StreamingTranscriber()
    try:
        transcriber.load_model()
        assert transcriber.model is not None
    finally:
        transcriber.unload_asr_model()


def test_speaker_tagging_and_ram_buffer(monkeypatch):
    """Feeds audio arrays tagged [Me] and [Others], verifies transcript_segments and purge_transcript()."""
    transcriber = StreamingTranscriber()

    # Use a lightweight fake model so speaker-tagging & RAM buffer behavior is 100% deterministic
    class FakePreprocessorConfig:
        hop_length = 160

    class FakeModel:
        preprocessor_config = FakePreprocessorConfig()

        def generate(self, mel):
            return [SimpleNamespace(text="Q4 roadmap review is ready.")]

    transcriber.model = FakeModel()
    monkeypatch.setattr("parakeet_mlx.audio.get_logmel", lambda audio, cfg: audio)

    # 1. Feed [Me] chunk
    me_audio = np.ones(16000 * 2, dtype=np.float32) * 0.2
    me_chunk = AudioChunk(speaker_tag="[Me]", data=me_audio, rms=0.2)
    transcriber.transcribe_chunk(me_chunk)

    # 2. Feed [Others] chunk
    others_audio = np.ones(16000 * 2, dtype=np.float32) * 0.3
    others_chunk = AudioChunk(speaker_tag="[Others]", data=others_audio, rms=0.3)
    transcriber.transcribe_chunk(others_chunk)

    # Verify speaker-tagged lines in RAM buffer
    segments = transcriber.get_transcript()
    assert len(segments) == 2
    assert segments[0].startswith("[Me] ")
    assert segments[1].startswith("[Others] ")
    assert "Q4 roadmap review is ready." in segments[0]

    # Verify chunk PCM buffers were scrubbed to 0.0 immediately after transcription
    assert np.all(me_audio == 0.0)
    assert np.all(others_audio == 0.0)

    # Verify stats
    stats = transcriber.get_stats()
    assert stats["segment_count"] == 2
    assert stats["word_count_in_ram"] > 0
    assert stats["estimated_tokens"] > 0

    # Verify purge_transcript() empties the RAM buffer to length 0
    transcriber.purge_transcript()
    assert len(transcriber.get_transcript()) == 0
    assert transcriber.get_transcript_text() == ""
    assert transcriber.get_stats()["word_count_in_ram"] == 0


def test_mlx_memory_unload():
    """Loads the ASR model, checks MLX memory, calls unload_asr_model(), and verifies memory drops near zero."""
    import mlx.core as mx

    transcriber = StreamingTranscriber()
    transcriber.load_model()

    # Evaluate parameters to ensure Metal buffers are materialized in Unified Memory
    mx.eval(transcriber.model.parameters())
    mem_loaded_bytes = mx.get_active_memory() + mx.get_cache_memory()
    mem_loaded_mb = get_mlx_memory_mb()
    assert mem_loaded_bytes > 0, "Expected non-zero MLX memory after loading Parakeet TDT 0.6B v3"
    assert mem_loaded_mb > 500.0, f"Expected > 500 MB loaded, got {mem_loaded_mb} MB"

    # Unload ASR model and clear Metal cache
    mem_after_mb = transcriber.unload_asr_model()
    mem_after_bytes = mx.get_active_memory() + mx.get_cache_memory()

    assert transcriber.model is None
    assert mem_after_bytes < mem_loaded_bytes * 0.05, (
        f"MLX memory did not drop near zero: before={mem_loaded_mb} MB, after={mem_after_mb} MB"
    )

