"""Automated Unit Tests for Phase 2: Dual-Stream Audio Capture & Ephemeral PCM Scrubbing.

Tests:
1. test_resampling_48k_stereo_to_16k_mono: Verifies 48kHz stereo downmix and polyphase resampling to 16kHz mono float32.
2. test_silence_gate_drops_dead_air: Confirms near-zero noise/silence is dropped by the RMS/VAD gate.
3. test_pcm_buffer_zeroed_after_consume: Asserts consumed PCM slices are overwritten with 0.0s and zero disk files exist.
4. test_audio_engine_end_to_end_dsp: Tests full AudioEngine processing with speech-level synthetic signals.
"""

from pathlib import Path
import numpy as np
import pytest

from backend.audio_engine import (
    AudioEngine,
    AudioStreamBuffer,
    compute_rms,
    downmix_to_mono,
    is_silent,
    resample_audio,
)


def test_resampling_48k_stereo_to_16k_mono():
    """Feeds synthetic 48kHz stereo sine wave and verifies output is (16000 * seconds,) mono float32."""
    orig_sr = 48000
    target_sr = 16000
    seconds = 3.0
    t = np.linspace(0, seconds, int(orig_sr * seconds), endpoint=False)

    # 440 Hz sine wave on left channel, 880 Hz on right channel
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    stereo_48k = np.column_stack((left, right)).astype(np.float32)

    assert stereo_48k.shape == (int(orig_sr * seconds), 2)
    assert stereo_48k.dtype == np.float32

    # Downmix to mono
    mono = downmix_to_mono(stereo_48k)
    assert mono.shape == (int(orig_sr * seconds),)
    assert mono.dtype == np.float32

    # Resample to 16kHz
    resampled_16k = resample_audio(mono, orig_sr=orig_sr, target_sr=target_sr)
    expected_length = int(target_sr * seconds)
    assert resampled_16k.shape == (expected_length,)
    assert resampled_16k.dtype == np.float32

    # Verify signal energy was preserved (not all zeros)
    rms_val = compute_rms(resampled_16k)
    assert rms_val > 0.1
    assert not is_silent(resampled_16k, threshold=0.01)


def test_silence_gate_drops_dead_air():
    """Feeds near-zero amplitude noise and confirms the VAD/RMS gate does not enqueue chunks."""
    buffer = AudioStreamBuffer(
        speaker_tag="[Me]",
        target_sr=16000,
        window_seconds=5.0,
        overlap_seconds=0.5,
        silence_threshold=0.005,
    )

    # 6 seconds of dead air (amplitude 0.0001)
    silence = np.random.uniform(-0.0001, 0.0001, size=(48000 * 6, 1)).astype(np.float32)
    chunks = buffer.add_pcm_data(silence, orig_sr=48000)

    # Since audio was silent, no chunk should be emitted for transcription
    assert len(chunks) == 0
    assert buffer.total_frames_dropped_silence > 0
    assert buffer.total_frames_enqueued == 0


def test_pcm_buffer_zeroed_after_consume():
    """Verifies consumed PCM buffer slice is filled with 0.0s and zero disk artifacts exist."""
    buffer = AudioStreamBuffer(
        speaker_tag="[Me]",
        target_sr=16000,
        window_seconds=5.0,
        overlap_seconds=0.5,
        silence_threshold=0.005,
    )

    # 6 seconds of audible 440Hz tone (enough to trigger a 5s window with 0.5s overlap)
    orig_sr = 48000
    t = np.linspace(0, 6.0, orig_sr * 6, endpoint=False)
    tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    chunks = buffer.add_pcm_data(tone, orig_sr=orig_sr)
    assert len(chunks) >= 1
    assert chunks[0].speaker_tag == "[Me]"
    assert chunks[0].data.shape == (16000 * 5,)

    # Verify that the consumed hop slice in the buffer was explicitly overwritten with 0.0
    assert buffer.last_scrubbed_slice is not None
    assert np.all(buffer.last_scrubbed_slice == 0.0)

    # Clear buffer and verify all internal references are zeroed
    buffer.clear()
    assert buffer.buffer.size == 0
    assert buffer.last_scrubbed_slice is None

    # Verify ZERO raw audio files exist on disk anywhere in the project (excluding .venv)
    project_root = Path(__file__).resolve().parent.parent
    audio_files = [
        p for p in project_root.rglob("*")
        if p.suffix in {".wav", ".raw", ".pcm"} and ".venv" not in p.parts
    ]
    assert len(audio_files) == 0, f"Found unintended audio files on disk: {audio_files}"


def test_audio_engine_end_to_end_dsp():
    """Tests AudioEngine processing with synthetic dual-channel speech-level audio."""
    engine = AudioEngine(
        target_sr=16000,
        window_seconds=5.0,
        overlap_seconds=0.5,
        silence_threshold=0.005,
    )

    # Feed 5.5 seconds of [Me] audio into mic_buffer
    t = np.linspace(0, 5.5, 48000 * 55 // 10, endpoint=False)
    me_signal = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    engine.mic_buffer.add_pcm_data(me_signal, orig_sr=48000, out_queue=engine.audio_queue)

    # Feed 5.5 seconds of [Others] audio into system_buffer
    others_signal = (0.3 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
    engine.system_buffer.add_pcm_data(others_signal, orig_sr=48000, out_queue=engine.audio_queue)

    # Check that both chunks were enqueued
    queued_chunks = []
    while not engine.audio_queue.empty():
        queued_chunks.append(engine.audio_queue.get_nowait())

    tags = [c.speaker_tag for c in queued_chunks]
    assert "[Me]" in tags
    assert "[Others]" in tags

    for c in queued_chunks:
        assert c.data.shape == (16000 * 5,)
        assert c.data.dtype == np.float32
        assert c.rms > 0.01

    # Check meters
    mic_lvl, sys_lvl = engine.get_levels()
    assert mic_lvl > 0.0
    assert sys_lvl > 0.0

    # Stop and clear
    engine.stop()
    assert engine.audio_queue.empty()
    assert engine.mic_buffer.buffer.size == 0
    assert engine.system_buffer.buffer.size == 0
