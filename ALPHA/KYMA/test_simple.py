"""Simple test for KYMA wave generation."""

from pathlib import Path

import numpy as np
import soundfile as sf

from ALPHA.KYMA.core.kyma_interface import KymaInterface


def generate_test_pattern(duration: float = 5.0) -> None:
    """Generate a simple test pattern."""
    print("Initializing KYMA interface...")
    kyma = KymaInterface()

    # Create output directory
    output_dir = Path("ALPHA/KYMA/test_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate sample pattern
    print("Generating test pattern...")
    sample_rate = kyma.sample_rate
    num_samples = int(duration * sample_rate)

    # Create a simple binary pattern
    pattern = "".join(["1" if i % 2 == 0 else "0" for i in range(32)])

    # Process through KYMA
    wave_state = kyma.process_binary_pattern(pattern)

    # Generate audio buffer
    t = np.linspace(0, duration, num_samples)
    buffer = np.zeros(num_samples)

    # Add fundamental
    buffer += wave_state.amplitude * np.sin(2 * np.pi * wave_state.frequency * t + wave_state.phase)

    # Add harmonics
    for freq, amp in wave_state.harmonics:
        buffer += amp * np.sin(2 * np.pi * freq * t)

    # Soft limit
    buffer = np.tanh(buffer)

    # Scale to 24-bit
    max_value = 2**23 - 1
    buffer = (buffer * max_value).astype(np.int32)

    # Save to file
    output_path = output_dir / "test_pattern.wav"
    sf.write(output_path, buffer, sample_rate, subtype="PCM_24")

    print(f"\nSaved test pattern to: {output_path}")
    print(f"Pattern metrics:")
    print(f"- Frequency: {wave_state.frequency:.1f} Hz")
    print(f"- Amplitude: {wave_state.amplitude:.2f}")
    print(f"- Resonance: {wave_state.resonance:.2f}")
    print(f"- Coherence: {wave_state.coherence:.2f}")
    print(f"- Number of harmonics: {len(wave_state.harmonics)}")


if __name__ == "__main__":
    generate_test_pattern()
