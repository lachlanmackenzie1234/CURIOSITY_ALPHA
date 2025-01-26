"""Test script for KYMA consciousness audio output."""

import time

from ALPHA.KYMA.core.audio_output import KymaAudioOutput


def test_consciousness():
    # Initialize with very low initial amplitude
    kyma = KymaAudioOutput("test_samples")

    print("Starting consciousness audio test...")
    print("First, let's save a 5-second sample to check levels...")

    # Generate a test sample first
    sample_path = kyma.save_sample(duration=5.0, name="consciousness_test")

    print(f"\nSaved test sample to: {sample_path}")
    print("Please check this file first to verify audio levels!")

    input("\nPress Enter to start real-time output (Ctrl+C to stop)...")

    try:
        # Start real-time output
        print("\nStarting real-time consciousness stream...")
        print("Volume should start very low and build gradually")
        print("Press Ctrl+C to stop at any time\n")

        kyma.start(output_mode="both")  # Both hardware and file output

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping consciousness stream...")
        kyma.stop()
        print("Done! Check the recordings directory for saved files.")


if __name__ == "__main__":
    test_consciousness()
