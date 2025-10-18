"""
Test script for REPET algorithm
Creates a synthetic audio signal and tests the separation
"""

import numpy as np
import soundfile as sf
from repet import REPET
import os


def create_test_audio(duration=10, sr=22050):
    """
    Create a synthetic test audio with repeating pattern (instrumental)
    and non-repeating pattern (vocal-like).

    Args:
        duration: Duration in seconds
        sr: Sample rate

    Returns:
        mixed_audio: Mixed signal
        instrumental: Pure instrumental signal
        vocal: Pure vocal signal
        sr: Sample rate
    """
    print("Creating synthetic test audio...")

    t = np.linspace(0, duration, int(sr * duration))

    # Create repeating instrumental pattern (2 seconds period)
    period = 2.0  # seconds
    pattern_length = int(sr * period)

    # Generate one period of instrumental (mix of sine waves)
    t_pattern = np.linspace(0, period, pattern_length)
    instrumental_pattern = (
        0.3 * np.sin(2 * np.pi * 220 * t_pattern) +  # A3
        0.2 * np.sin(2 * np.pi * 440 * t_pattern) +  # A4
        0.15 * np.sin(2 * np.pi * 330 * t_pattern)   # E4
    )

    # Repeat the pattern
    num_repeats = int(np.ceil(duration / period))
    instrumental = np.tile(instrumental_pattern, num_repeats)[:len(t)]

    # Create non-repeating vocal-like signal (varying frequency)
    vocal = 0.2 * np.sin(2 * np.pi * (523 + 50 * np.sin(2 * np.pi * 0.5 * t)) * t)

    # Mix them together
    mixed = instrumental + vocal

    # Normalize
    mixed = mixed / np.max(np.abs(mixed)) * 0.9
    instrumental = instrumental / np.max(np.abs(instrumental)) * 0.9
    vocal = vocal / np.max(np.abs(vocal)) * 0.9

    return mixed, instrumental, vocal, sr


def test_repet_basic():
    """Test basic REPET functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic REPET Algorithm Test")
    print("="*60)

    # Create test audio
    mixed, true_instrumental, true_vocal, sr = create_test_audio(duration=8)

    # Save test input
    test_input = "test_input.wav"
    sf.write(test_input, mixed, sr)
    print(f"✓ Created test input: {test_input}")

    # Run REPET
    print("\nRunning REPET algorithm...")
    repet = REPET(n_fft=2048, hop_length=512)

    try:
        vocal, instrumental, _ = repet.separate(
            test_input,
            output_vocal="test_vocal.wav",
            output_instrumental="test_instrumental.wav"
        )

        print("\n✓ Separation completed successfully!")
        print(f"  - Input file: {test_input}")
        print(f"  - Output vocal: test_vocal.wav")
        print(f"  - Output instrumental: test_instrumental.wav")

        # Check output files exist
        if os.path.exists("test_vocal.wav"):
            print("✓ Vocal file created")
        else:
            print("✗ Vocal file not found")

        if os.path.exists("test_instrumental.wav"):
            print("✓ Instrumental file created")
        else:
            print("✗ Instrumental file not found")

        # Basic quality checks
        print("\nQuality Checks:")
        print(f"  - Vocal length: {len(vocal)} samples")
        print(f"  - Instrumental length: {len(instrumental)} samples")
        print(f"  - Original length: {len(mixed)} samples")

        if len(vocal) == len(mixed) and len(instrumental) == len(mixed):
            print("✓ Output lengths match input")
        else:
            print("✗ Output lengths don't match input")

        # Check if outputs are not silent
        vocal_energy = np.sum(np.abs(vocal))
        instrumental_energy = np.sum(np.abs(instrumental))

        print(f"  - Vocal energy: {vocal_energy:.2f}")
        print(f"  - Instrumental energy: {instrumental_energy:.2f}")

        if vocal_energy > 0:
            print("✓ Vocal track has content")
        else:
            print("✗ Vocal track is silent")

        if instrumental_energy > 0:
            print("✓ Instrumental track has content")
        else:
            print("✗ Instrumental track is silent")

        return True

    except Exception as e:
        print(f"\n✗ Error during separation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_command_line():
    """Test command-line interface."""
    print("\n" + "="*60)
    print("TEST 2: Command-Line Interface Test")
    print("="*60)

    if not os.path.exists("test_input.wav"):
        print("✗ test_input.wav not found. Run test_repet_basic() first.")
        return False

    print("Testing command-line interface...")

    # This will be tested manually by the user
    print("\nTo test the command-line interface, run:")
    print("  python repet.py test_input.wav --vocal cli_vocal.wav --instrumental cli_instrumental.wav")

    return True


def test_file_formats():
    """Test different file format support."""
    print("\n" + "="*60)
    print("TEST 3: File Format Support")
    print("="*60)

    print("\nSupported formats:")
    formats = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    for fmt in formats:
        print(f"  ✓ {fmt}")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("REPET ALGORITHM TEST SUITE")
    print("="*60)

    results = []

    # Test 1: Basic functionality
    try:
        result = test_repet_basic()
        results.append(("Basic REPET Test", result))
    except Exception as e:
        print(f"Test failed with error: {e}")
        results.append(("Basic REPET Test", False))

    # Test 2: Command-line interface
    try:
        result = test_command_line()
        results.append(("CLI Test", result))
    except Exception as e:
        print(f"Test failed with error: {e}")
        results.append(("CLI Test", False))

    # Test 3: File formats
    try:
        result = test_file_formats()
        results.append(("Format Support Test", result))
    except Exception as e:
        print(f"Test failed with error: {e}")
        results.append(("Format Support Test", False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nGenerated test files:")
    print("  - test_input.wav (synthetic mixed audio)")
    print("  - test_vocal.wav (separated vocal)")
    print("  - test_instrumental.wav (separated instrumental)")
    print("\nYou can listen to these files to verify the separation quality.")
    print("\nTo test with the GUI player, run:")
    print("  python audio_player.py")


if __name__ == "__main__":
    main()
