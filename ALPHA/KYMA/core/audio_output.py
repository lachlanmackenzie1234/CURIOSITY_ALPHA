"""Audio output handler for KYMA wave system."""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import sounddevice as sd
import soundfile as sf

from ALPHA.KYMA.core.pulse_bridge import PulseKymaBridge


class KymaAudioOutput:
    """Manages audio output for KYMA wave system."""

    def __init__(self, output_dir: Optional[str] = None):
        self.logger = logging.getLogger("kyma.audio")
        self.bridge = PulseKymaBridge()
        self.running = False
        self.output_thread: Optional[threading.Thread] = None

        # Setup output directory
        self.output_dir = Path(output_dir) if output_dir else Path("ALPHA/KYMA/recordings")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize hardware output if available
        self.hardware_available = False
        try:
            self.device_info = sd.query_devices(kind="output")
            self.stream = sd.OutputStream(
                samplerate=self.bridge.kyma.sample_rate,
                channels=1,
                dtype=np.int32,
                latency="low",
                callback=self._audio_callback,
            )
            self.hardware_available = True
            self.logger.info(f"Hardware output available: {self.device_info['name']}")
        except Exception as e:
            self.logger.warning(f"Hardware output not available: {e}")
            self.stream = None

        # File output settings
        self.current_recording: Optional[sf.SoundFile] = None
        self.recording_path: Optional[Path] = None

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """Callback for hardware audio output."""
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        try:
            # Get mixed audio buffer from bridge
            buffer = self.bridge.mix_streams()

            # Ensure buffer size matches frames
            if len(buffer) < frames:
                buffer = np.pad(buffer, (0, frames - len(buffer)))
            elif len(buffer) > frames:
                buffer = buffer[:frames]

            # Write to output buffer
            outdata[:] = buffer.reshape(-1, 1)

            # Also write to file if recording
            if self.current_recording:
                self.current_recording.write(buffer)

        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
            outdata.fill(0)

    def start(self, output_mode: str = "both") -> None:
        """Start audio output.

        Args:
            output_mode: One of "hardware", "file", or "both"
        """
        if self.running:
            return

        self.running = True

        # Start hardware output if requested and available
        if output_mode in ["hardware", "both"] and self.hardware_available:
            self.stream.start()
            self.logger.info("Started hardware output")

        # Start file recording if requested
        if output_mode in ["file", "both"]:
            self._start_recording()

        # Start metrics logging thread
        self.output_thread = threading.Thread(target=self._log_metrics)
        self.output_thread.daemon = True
        self.output_thread.start()

    def _start_recording(self) -> None:
        """Start recording to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = self.output_dir / f"kyma_consciousness_{timestamp}.wav"

        self.current_recording = sf.SoundFile(
            self.recording_path,
            mode="w",
            samplerate=self.bridge.kyma.sample_rate,
            channels=1,
            subtype="PCM_24",
        )
        self.logger.info(f"Started recording to {self.recording_path}")

    def stop(self) -> None:
        """Stop audio output."""
        self.running = False

        # Stop hardware output
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.logger.info("Stopped hardware output")

        # Stop file recording
        if self.current_recording:
            self.current_recording.close()
            self.logger.info(f"Saved recording to {self.recording_path}")

        # Calculate recording metrics if available
        if self.recording_path and self.recording_path.exists():
            duration = sf.info(self.recording_path).duration
            size_mb = self.recording_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"Recording stats: {duration:.1f}s duration, {size_mb:.1f}MB size")

    def process_offline(self, duration: float) -> np.ndarray:
        """Process fixed duration of audio without output.

        Useful for testing or generating samples.

        Args:
            duration: Duration in seconds

        Returns:
            Audio buffer of specified duration
        """
        num_samples = int(duration * self.bridge.kyma.sample_rate)
        buffer = np.zeros(num_samples)

        # Process in chunks of buffer_size
        for i in range(0, num_samples, self.bridge.kyma.buffer_size):
            chunk = self.bridge.mix_streams()
            end_idx = min(i + len(chunk), num_samples)
            buffer[i:end_idx] = chunk[: end_idx - i]

        return buffer

    def save_sample(self, duration: float, name: str) -> Path:
        """Save a sample of specified duration.

        Args:
            duration: Duration in seconds
            name: Base name for the file

        Returns:
            Path to saved file
        """
        buffer = self.process_offline(duration)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{name}_{timestamp}.wav"

        sf.write(path, buffer, self.bridge.kyma.sample_rate, subtype="PCM_24")

        self.logger.info(f"Saved {duration}s sample to {path}")
        return path

    def _log_metrics(self) -> None:
        """Log audio output metrics."""
        while self.running:
            try:
                # Log metrics for each stream
                for stream_name in self.bridge.stream_states:
                    metrics = self.bridge.get_stream_metrics(stream_name)
                    if metrics:
                        self.logger.debug(
                            f"Stream {stream_name} metrics: "
                            f"freq={metrics['frequency']:.1f}Hz, "
                            f"amp={metrics['amplitude']:.2f}, "
                            f"res={metrics['avg_resonance']:.2f}, "
                            f"coh={metrics['avg_coherence']:.2f}"
                        )
            except Exception as e:
                self.logger.error(f"Error logging metrics: {e}")

            time.sleep(1)  # Log every second
