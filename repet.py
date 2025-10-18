"""
REPET (REpeating Pattern Extraction Technique) Algorithm
Separates vocals and instrumental from audio files
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter


class REPET:
    """
    REPET algorithm for audio source separation.
    Separates repeating background (instrumental) from non-repeating foreground (vocals).
    """

    def __init__(self, n_fft=2048, hop_length=512):
        """
        Initialize REPET processor.

        Args:
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
        """
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_audio(self, filepath):
        """
        Load audio file.

        Args:
            filepath: Path to audio file

        Returns:
            audio: Audio signal
            sr: Sample rate
        """
        audio, sr = librosa.load(filepath, sr=None, mono=False)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        return audio, sr

    def compute_stft(self, audio):
        """
        Compute Short-Time Fourier Transform.

        Args:
            audio: Audio signal

        Returns:
            Complex STFT matrix
        """
        return librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)

    def find_repeating_period(self, spectrogram, sr):
        """
        Find the repeating period in the audio using autocorrelation.

        Args:
            spectrogram: Magnitude spectrogram
            sr: Sample rate

        Returns:
            period: Repeating period in frames
        """
        # Compute beat spectrum (autocorrelation along time axis)
        beat_spectrum = np.mean([
            np.correlate(spectrogram[i, :], spectrogram[i, :], mode='full')
            for i in range(min(spectrogram.shape[0], 100))
        ], axis=0)

        # Take second half (positive lags)
        beat_spectrum = beat_spectrum[len(beat_spectrum)//2:]

        # Find peaks to determine period
        # Look for period between 1-10 seconds
        min_period = int(1.0 * sr / self.hop_length)
        max_period = int(10.0 * sr / self.hop_length)

        if len(beat_spectrum) > max_period:
            search_range = beat_spectrum[min_period:max_period]
            period = np.argmax(search_range) + min_period
        else:
            period = len(beat_spectrum) // 4  # Default fallback

        return max(period, 1)

    def compute_repeating_mask(self, spectrogram, period):
        """
        Compute the repeating pattern mask.

        Args:
            spectrogram: Magnitude spectrogram
            period: Repeating period in frames

        Returns:
            mask: Binary mask for repeating elements
        """
        num_frames = spectrogram.shape[1]
        num_repetitions = num_frames // period

        if num_repetitions < 2:
            # Not enough repetitions, return simple median filter
            repeating_spec = median_filter(spectrogram, size=(1, period))
        else:
            # Compute repeating segment by taking median over segments
            segments = []
            for i in range(num_repetitions):
                start = i * period
                end = min(start + period, num_frames)
                if end - start == period:
                    segments.append(spectrogram[:, start:end])

            if segments:
                # Stack and take median
                segments_array = np.stack(segments, axis=0)
                repeating_segment = np.median(segments_array, axis=0)

                # Tile the repeating segment
                repeating_spec = np.tile(repeating_segment, (1, num_repetitions))

                # Trim or pad to match original length
                if repeating_spec.shape[1] > num_frames:
                    repeating_spec = repeating_spec[:, :num_frames]
                elif repeating_spec.shape[1] < num_frames:
                    pad_width = num_frames - repeating_spec.shape[1]
                    repeating_spec = np.pad(repeating_spec,
                                           ((0, 0), (0, pad_width)),
                                           mode='edge')
            else:
                repeating_spec = spectrogram

        # Create soft mask using Wiener-like filtering
        eps = 1e-10
        mask = (repeating_spec + eps) / (spectrogram + eps)
        mask = np.minimum(mask, 1.0)

        return mask

    def separate(self, filepath, output_instrumental=None, output_vocal=None):
        """
        Separate audio into instrumental and vocal components.

        Args:
            filepath: Input audio file path
            output_instrumental: Output path for instrumental track
            output_vocal: Output path for vocal track

        Returns:
            vocal: Vocal audio signal
            instrumental: Instrumental audio signal
            sr: Sample rate
        """
        print(f"Loading audio file: {filepath}")
        audio, sr = self.load_audio(filepath)

        print("Computing STFT...")
        stft = self.compute_stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        print("Finding repeating period...")
        period = self.find_repeating_period(magnitude, sr)
        print(f"Detected period: {period} frames ({period * self.hop_length / sr:.2f} seconds)")

        print("Computing repeating pattern mask...")
        instrumental_mask = self.compute_repeating_mask(magnitude, period)
        vocal_mask = 1.0 - instrumental_mask

        # Apply masks
        instrumental_stft = instrumental_mask * magnitude * np.exp(1j * phase)
        vocal_stft = vocal_mask * magnitude * np.exp(1j * phase)

        print("Reconstructing audio...")
        instrumental = librosa.istft(instrumental_stft,
                                     hop_length=self.hop_length,
                                     length=len(audio))
        vocal = librosa.istft(vocal_stft,
                             hop_length=self.hop_length,
                             length=len(audio))

        # Save outputs if paths provided
        if output_instrumental:
            print(f"Saving instrumental to: {output_instrumental}")
            sf.write(output_instrumental, instrumental, sr)

        if output_vocal:
            print(f"Saving vocal to: {output_vocal}")
            sf.write(output_vocal, vocal, sr)

        print("Separation complete!")
        return vocal, instrumental, sr


def main():
    """
    Command-line interface for REPET algorithm.
    """
    import argparse

    parser = argparse.ArgumentParser(description='REPET Audio Source Separation')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('--vocal', default='vocal.wav', help='Output vocal file')
    parser.add_argument('--instrumental', default='instrumental.wav',
                       help='Output instrumental file')
    parser.add_argument('--n-fft', type=int, default=2048, help='FFT size')
    parser.add_argument('--hop-length', type=int, default=512, help='Hop length')

    args = parser.parse_args()

    # Create REPET instance and separate
    repet = REPET(n_fft=args.n_fft, hop_length=args.hop_length)
    repet.separate(args.input,
                  output_instrumental=args.instrumental,
                  output_vocal=args.vocal)


if __name__ == '__main__':
    main()
