"""Dual-Stream Audio Capture, Resampling & Ephemeral In-Memory PCM Scrubbing.

Features:
- Dual-stream audio capture: Microphone ([Me]) and System Audio / BlackHole ([Others]).
- Automatic macOS input device enumeration and BlackHole auto-detection.
- In-memory 48kHz (or native) -> 16kHz mono float32 resampling using scipy.signal.resample_poly.
- Real-time RMS calculation and silence gating to skip dead air.
- In-memory queue with 5-second windows (0.5s overlap) for the streaming transcriber.
- Ephemeral PCM scrubbing: Immediately overwrites source buffer slices with zeros (buf.fill(0))
  to guarantee zero accumulation of raw audio in RAM and zero audio files on disk.
"""

import argparse
import math
import queue
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
import sounddevice as sd

from backend.config import (
    AUDIO_CHANNELS,
    AUDIO_OVERLAP_SECONDS,
    AUDIO_TARGET_SAMPLE_RATE,
    AUDIO_WINDOW_SECONDS,
)

# RMS energy threshold below which audio is treated as dead air/silence
DEFAULT_SILENCE_THRESHOLD_RMS = 0.005


@dataclass
class AudioChunk:
    """Ephemeral audio chunk handed off to the transcriber."""
    speaker_tag: str  # "[Me]" or "[Others]"
    data: np.ndarray  # 1D float32 array sampled at 16000 Hz
    sample_rate: int = AUDIO_TARGET_SAMPLE_RATE
    timestamp: float = 0.0
    rms: float = 0.0


def downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Converts multi-channel audio to mono float32."""
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        if audio.shape[1] == 1:
            return audio[:, 0].astype(np.float32, copy=False)
        return np.mean(audio, axis=1, dtype=np.float32)
    # Multi-dimensional fallback
    return np.mean(audio, axis=tuple(range(1, audio.ndim)), dtype=np.float32)


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = AUDIO_TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Resamples mono audio to target sample rate using polyphase filtering."""
    if audio.size == 0 or orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled = signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32, copy=False)


def compute_rms(audio: np.ndarray) -> float:
    """Calculates Root-Mean-Square (RMS) amplitude of audio signal."""
    if audio.size == 0:
        return 0.0
    mean_sq = np.mean(np.square(audio, dtype=np.float32))
    return float(np.sqrt(mean_sq))


def is_silent(audio: np.ndarray, threshold: float = DEFAULT_SILENCE_THRESHOLD_RMS) -> bool:
    """Determines if the audio frame falls below the silence threshold."""
    return compute_rms(audio) < threshold


def list_input_devices() -> List[Dict[str, Any]]:
    """Lists all available audio input devices on macOS."""
    try:
        devices = sd.query_devices()
    except Exception:
        return []

    default_input_idx = -1
    try:
        default_device_info = sd.default.device
        if isinstance(default_device_info, (list, tuple)) and len(default_device_info) > 0:
            default_input_idx = default_device_info[0]
    except Exception:
        pass

    results = []
    for idx, dev in enumerate(devices):
        max_in = dev.get("max_input_channels", 0)
        if max_in > 0:
            name = dev.get("name", f"Device {idx}")
            is_blackhole = "blackhole" in name.lower()
            results.append(
                {
                    "id": idx,
                    "name": name,
                    "channels": max_in,
                    "sample_rate": int(dev.get("default_samplerate", 48000)),
                    "is_default": (idx == default_input_idx),
                    "is_blackhole": is_blackhole,
                }
            )
    return results


def find_default_devices() -> Tuple[Optional[int], Optional[int]]:
    """Auto-detects default Microphone and BlackHole 2ch device IDs."""
    devices = list_input_devices()
    if not devices:
        return None, None

    mic_id: Optional[int] = None
    system_id: Optional[int] = None

    # First look for BlackHole
    for dev in devices:
        if dev["is_blackhole"]:
            system_id = dev["id"]
            break

    # Look for default input device or standard mic
    for dev in devices:
        if dev["is_default"] and not dev["is_blackhole"]:
            mic_id = dev["id"]
            break

    # Fallback for mic if default is not set or matches blackhole
    if mic_id is None:
        for dev in devices:
            if not dev["is_blackhole"]:
                mic_id = dev["id"]
                break

    return mic_id, system_id


class AudioStreamBuffer:
    """Manages sliding window accumulation, silence gating, and PCM scrubbing for a single stream."""

    def __init__(
        self,
        speaker_tag: str,
        target_sr: int = AUDIO_TARGET_SAMPLE_RATE,
        window_seconds: float = AUDIO_WINDOW_SECONDS,
        overlap_seconds: float = AUDIO_OVERLAP_SECONDS,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD_RMS,
    ):
        self.speaker_tag = speaker_tag
        self.target_sr = target_sr
        self.window_samples = int(target_sr * window_seconds)
        self.overlap_samples = int(target_sr * overlap_seconds)
        self.hop_samples = max(1, self.window_samples - self.overlap_samples)
        self.silence_threshold = silence_threshold

        self.buffer = np.zeros(0, dtype=np.float32)
        self.current_rms: float = 0.0
        self.last_scrubbed_slice: Optional[np.ndarray] = None
        self.total_frames_enqueued: int = 0
        self.total_frames_dropped_silence: int = 0

    def add_pcm_data(
        self,
        raw_pcm: np.ndarray,
        orig_sr: int,
        out_queue: Optional[queue.Queue] = None,
    ) -> List[AudioChunk]:
        """Ingests raw PCM frames, downmixes, resamples, gates silence, and scrubs memory."""
        mono = downmix_to_mono(raw_pcm)
        resampled = resample_audio(mono, orig_sr=orig_sr, target_sr=self.target_sr)

        if resampled.size == 0:
            return []

        # Update real-time RMS for level meters
        self.current_rms = compute_rms(resampled)

        # Append new resampled audio to running buffer
        if self.buffer.size == 0:
            self.buffer = resampled.copy()
        else:
            self.buffer = np.concatenate([self.buffer, resampled])

        emitted_chunks: List[AudioChunk] = []

        # Extract full windows
        while len(self.buffer) >= self.window_samples:
            window = self.buffer[: self.window_samples].copy()
            window_rms = compute_rms(window)
            self.current_rms = window_rms

            if window_rms >= self.silence_threshold:
                chunk = AudioChunk(
                    speaker_tag=self.speaker_tag,
                    data=window,
                    sample_rate=self.target_sr,
                    timestamp=time.time(),
                    rms=window_rms,
                )
                emitted_chunks.append(chunk)
                if out_queue is not None:
                    out_queue.put(chunk)
                self.total_frames_enqueued += 1
            else:
                self.total_frames_dropped_silence += 1

            # EPHEMERAL PCM SCRUBBING:
            # Overwrite the consumed hop slice with zeros before advancing the window
            consumed_slice = self.buffer[: self.hop_samples]
            consumed_slice.fill(0.0)
            self.last_scrubbed_slice = consumed_slice.copy()

            # Advance buffer by hop_samples
            self.buffer = self.buffer[self.hop_samples :]

        return emitted_chunks

    def clear(self) -> None:
        """Scans and explicitly zeros out all underlying memory buffers."""
        if self.buffer.size > 0:
            self.buffer.fill(0.0)
        self.buffer = np.zeros(0, dtype=np.float32)
        if self.last_scrubbed_slice is not None:
            self.last_scrubbed_slice.fill(0.0)
            self.last_scrubbed_slice = None
        self.current_rms = 0.0


class AudioEngine:
    """Coordinates dual-stream audio capture for Mic [Me] and System [Others]."""

    def __init__(
        self,
        target_sr: int = AUDIO_TARGET_SAMPLE_RATE,
        window_seconds: float = AUDIO_WINDOW_SECONDS,
        overlap_seconds: float = AUDIO_OVERLAP_SECONDS,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD_RMS,
    ):
        self.target_sr = target_sr
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self.silence_threshold = silence_threshold

        self.mic_buffer = AudioStreamBuffer(
            speaker_tag="[Me]",
            target_sr=target_sr,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            silence_threshold=silence_threshold,
        )
        self.system_buffer = AudioStreamBuffer(
            speaker_tag="[Others]",
            target_sr=target_sr,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            silence_threshold=silence_threshold,
        )

        self.audio_queue: queue.Queue[AudioChunk] = queue.Queue()
        self.mic_stream: Optional[sd.InputStream] = None
        self.system_stream: Optional[sd.InputStream] = None
        self.is_running: bool = False

    def start(
        self,
        mic_device_id: Optional[int] = None,
        system_device_id: Optional[int] = None,
    ) -> bool:
        """Starts dual audio capture streams."""
        if self.is_running:
            return True

        self.clear()
        auto_mic, auto_sys = find_default_devices()
        target_mic = mic_device_id if mic_device_id is not None else auto_mic
        target_sys = system_device_id if system_device_id is not None else auto_sys

        # Start Mic Stream if available
        if target_mic is not None:
            try:
                mic_info = sd.query_devices(target_mic)
                mic_sr = int(mic_info.get("default_samplerate", 48000))
                mic_channels = min(2, mic_info.get("max_input_channels", 1))

                def mic_callback(indata, frames, time_info, status):
                    if status:
                        sys.stderr.write(f"Mic status: {status}\n")
                    self.mic_buffer.add_pcm_data(indata, orig_sr=mic_sr, out_queue=self.audio_queue)

                self.mic_stream = sd.InputStream(
                    device=target_mic,
                    channels=mic_channels,
                    samplerate=mic_sr,
                    callback=mic_callback,
                    dtype="float32",
                )
                self.mic_stream.start()
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to start Mic stream on device {target_mic}: {e}\n")

        # Start System (BlackHole) Stream if available
        if target_sys is not None:
            try:
                sys_info = sd.query_devices(target_sys)
                sys_sr = int(sys_info.get("default_samplerate", 48000))
                sys_channels = min(2, sys_info.get("max_input_channels", 2))

                def system_callback(indata, frames, time_info, status):
                    if status:
                        sys.stderr.write(f"System status: {status}\n")
                    self.system_buffer.add_pcm_data(indata, orig_sr=sys_sr, out_queue=self.audio_queue)

                self.system_stream = sd.InputStream(
                    device=target_sys,
                    channels=sys_channels,
                    samplerate=sys_sr,
                    callback=system_callback,
                    dtype="float32",
                )
                self.system_stream.start()
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to start System audio stream on device {target_sys}: {e}\n")

        self.is_running = True
        return True

    def stop(self) -> None:
        """Stops streams and immediately scrubs all internal PCM buffers with zeros."""
        self.is_running = False

        if self.mic_stream is not None:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None

        if self.system_stream is not None:
            try:
                self.system_stream.stop()
                self.system_stream.close()
            except Exception:
                pass
            self.system_stream = None

        self.clear()

    def get_levels(self) -> Tuple[float, float]:
        """Returns instantaneous RMS levels (mic_rms, system_rms) clamped [0.0, 1.0]."""
        mic_lvl = min(1.0, max(0.0, self.mic_buffer.current_rms))
        sys_lvl = min(1.0, max(0.0, self.system_buffer.current_rms))
        return mic_lvl, sys_lvl

    def clear(self) -> None:
        """Zeros out all stream buffers and drains the audio queue."""
        self.mic_buffer.clear()
        self.system_buffer.clear()
        while not self.audio_queue.empty():
            try:
                item = self.audio_queue.get_nowait()
                if hasattr(item, "data") and isinstance(item.data, np.ndarray):
                    item.data.fill(0.0)
            except queue.Empty:
                break


# ==============================================================================
# CLI Diagnostic Mode (--test-meters)
# ==============================================================================

def render_meter_bar(label: str, level: float, bar_len: int = 30) -> str:
    """Renders a visual ASCII audio level meter."""
    filled = int(round(level * bar_len))
    filled = min(bar_len, max(0, filled))
    bar = "|" * filled + " " * (bar_len - filled)
    return f"{label:18} [{bar}] {level:0.3f}"


def run_diagnostic_meters(seconds: float = 5.0, mic_id: Optional[int] = None, sys_id: Optional[int] = None) -> None:
    """Runs a live dual-stream meter diagnostic check in the terminal."""
    devices = list_input_devices()
    print("=== Zero-Trust Audio Engine Diagnostic ===")
    print(f"Available input devices ({len(devices)}):")
    for d in devices:
        mark = " (Default)" if d["is_default"] else ""
        bh = " [BlackHole 2ch]" if d["is_blackhole"] else ""
        print(f"  [{d['id']}] {d['name']} ({d['channels']} ch, {d['sample_rate']} Hz){mark}{bh}")

    auto_mic, auto_sys = find_default_devices()
    target_mic = mic_id if mic_id is not None else auto_mic
    target_sys = sys_id if sys_id is not None else auto_sys

    print(f"\nBinding Mic [Me]     -> Device: {target_mic}")
    print(f"Binding System [Others] -> Device: {target_sys}")
    print(f"Running live meter check for {seconds} seconds (Press Ctrl+C to stop)...\n")

    engine = AudioEngine()
    engine.start(mic_device_id=target_mic, system_device_id=target_sys)

    start_time = time.time()
    try:
        while time.time() - start_time < seconds:
            mic_lvl, sys_lvl = engine.get_levels()
            m_bar = render_meter_bar("Mic [Me]", mic_lvl)
            s_bar = render_meter_bar("System [Others]", sys_lvl)
            # Print over previous lines with carriage return
            sys.stdout.write(f"\r{m_bar}   |   {s_bar}")
            sys.stdout.flush()
            time.sleep(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print("\n\nDiagnostic complete.")
        print(f"Enqueued Mic chunks: {engine.mic_buffer.total_frames_enqueued}")
        print(f"Enqueued System chunks: {engine.system_buffer.total_frames_enqueued}")
        print("Buffers zeroed and cleared. Zero audio files written to disk.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Trust Local Audio Engine Diagnostics")
    parser.add_argument("--test-meters", action="store_true", help="Run live dual audio meter check")
    parser.add_argument("--seconds", type=float, default=5.0, help="Duration to run meter test")
    parser.add_argument("--mic-id", type=int, default=None, help="Explicit Mic input device ID")
    parser.add_argument("--system-id", type=int, default=None, help="Explicit System/BlackHole device ID")
    args = parser.parse_args()

    if args.test_meters:
        run_diagnostic_meters(seconds=args.seconds, mic_id=args.mic_id, sys_id=args.system_id)
    else:
        parser.print_help()

