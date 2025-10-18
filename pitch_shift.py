"""
Pitch shifting implementation with three methods:
1. TD-PSOLA (Time-Domain Pitch Synchronous Overlap-Add)
2. Phase Vocoder (librosa)
3. WSOLA (Waveform Similarity Overlap-Add) - like SoundTouch
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import get_window, correlate
from scipy.interpolate import interp1d


class PitchShifter:
    """
    Multi-method pitch shifter supporting TD-PSOLA, Phase Vocoder, and WSOLA.
    """

    def __init__(self, sr=22050):
        """
        Initialize pitch shifter.

        Args:
            sr: Sample rate
        """
        self.sr = sr

    def detect_pitch_enhanced(self, audio, frame_length=2048, hop_length=512):
        """
        Enhanced pitch detection using librosa's pyin with better interpolation.

        Args:
            audio: Audio signal
            frame_length: Length of each frame for analysis
            hop_length: Hop length between frames

        Returns:
            f0: Array of fundamental frequencies (Hz) for each frame
            voiced_flag: Boolean array indicating voiced frames
        """
        # Use librosa's pyin for pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr,
            frame_length=frame_length,
            hop_length=hop_length,
            fill_na=None
        )

        # Better interpolation for unvoiced segments
        if voiced_flag is not None and not voiced_flag.all():
            voiced_indices = np.where(voiced_flag)[0]
            if len(voiced_indices) > 1:
                # Interpolate only where we have voiced segments
                interp_func = interp1d(
                    voiced_indices,
                    f0[voiced_indices],
                    kind='cubic',
                    bounds_error=False,
                    fill_value=(f0[voiced_indices[0]], f0[voiced_indices[-1]])
                )
                all_indices = np.arange(len(f0))
                f0 = interp_func(all_indices)
            else:
                # If not enough voiced segments, use median
                median_pitch = np.median(f0[voiced_flag]) if any(voiced_flag) else 200.0
                f0 = np.full_like(f0, median_pitch, dtype=float)

        # Handle any remaining NaN values
        mask = np.isnan(f0)
        if mask.any():
            if not mask.all():
                f0[mask] = np.interp(np.flatnonzero(mask),
                                    np.flatnonzero(~mask),
                                    f0[~mask])
            else:
                f0 = np.full_like(f0, 200.0)

        return f0, voiced_flag

    def detect_pitch_from_audio(self, audio):
        """
        Detect average pitch from an audio signal.

        Args:
            audio: Audio signal

        Returns:
            Average pitch in Hz
        """
        f0, voiced_flag = self.detect_pitch_enhanced(audio)

        # Take median of voiced segments only
        if voiced_flag is not None and any(voiced_flag):
            median_pitch = np.median(f0[voiced_flag])
        else:
            median_pitch = np.median(f0[~np.isnan(f0)]) if not np.isnan(f0).all() else 200.0

        return median_pitch

    def find_pitch_marks_yin(self, audio, fmin=80, fmax=400):
        """
        Find pitch marks using YIN algorithm for better accuracy.

        Args:
            audio: Audio signal
            fmin: Minimum frequency
            fmax: Maximum frequency

        Returns:
            Array of pitch mark positions (in samples)
        """
        marks = []

        # Estimate period range
        max_period = int(self.sr / fmin)
        min_period = int(self.sr / fmax)

        pos = 0

        while pos < len(audio) - max_period:
            # Extract segment
            segment_len = min(max_period * 2, len(audio) - pos)
            segment = audio[pos:pos + segment_len]

            if len(segment) < max_period:
                break

            # Compute autocorrelation
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find peak in valid range
            valid_autocorr = autocorr[min_period:max_period]
            if len(valid_autocorr) > 0:
                local_peak = np.argmax(valid_autocorr)
                period = min_period + local_peak
            else:
                period = (min_period + max_period) // 2

            marks.append(pos)
            pos += period

        return np.array(marks, dtype=int)

    def td_psola(self, audio, pitch_shift_factor):
        """
        Improved TD-PSOLA with three-buffer overlap system.

        Uses 'past', 'present', 'future' buffers to ensure smooth transitions
        and eliminate discontinuities at grain boundaries.

        The key is:
        1. Extract overlapping grains (past, present, future)
        2. Resample each grain to change pitch
        3. Cross-fade between grains for smooth transitions
        4. Overlap-add at original spacing (preserves duration)

        Args:
            audio: Input audio signal
            pitch_shift_factor: Pitch shift factor (e.g., 2.0 = up one octave)

        Returns:
            Pitch-shifted audio with same duration
        """
        if abs(pitch_shift_factor - 1.0) < 0.001:
            return audio

        # Find pitch marks
        marks = self.find_pitch_marks_yin(audio, fmin=80, fmax=500)

        if len(marks) < 4:
            print("Warning: Not enough pitch marks, falling back to phase vocoder")
            return self.phase_vocoder(audio, pitch_shift_factor)

        # Calculate periods between marks
        periods = np.diff(marks)
        avg_period = int(np.median(periods))

        # Window size: 2 periods for main grain, larger for overlap context
        grain_len = avg_period * 2
        overlap_len = avg_period  # Extra overlap on each side

        # Output buffer (same length as input!)
        output = np.zeros(len(audio))

        # Overlap weight buffer to track contribution from each grain
        weight_sum = np.zeros(len(audio))

        # Process each pitch mark with three-buffer system
        for i in range(1, len(marks) - 2):
            # Get three consecutive marks: past, present, future
            mark_past = marks[i - 1]
            mark_present = marks[i]
            mark_future = marks[i + 1]

            # Calculate local period
            period = mark_present - mark_past
            next_period = mark_future - mark_present
            local_period = (period + next_period) // 2

            # Extract extended grain including context from past and future
            # This helps maintain continuity
            grain_start = max(0, mark_present - grain_len // 2 - overlap_len)
            grain_end = min(len(audio), mark_present + grain_len // 2 + overlap_len)
            extended_grain = audio[grain_start:grain_end]

            # Calculate actual grain length
            actual_grain_len = len(extended_grain)

            if actual_grain_len < grain_len // 2:
                continue

            # Create smooth window with extended tails for better overlap
            window = get_window('hann', actual_grain_len)

            # Apply window
            windowed_grain = extended_grain * window

            # RESAMPLE grain to change pitch
            new_len = int(actual_grain_len / pitch_shift_factor)
            if new_len > 8:  # Minimum size check
                # High-quality resampling
                old_indices = np.linspace(0, actual_grain_len - 1, actual_grain_len)
                new_indices = np.linspace(0, actual_grain_len - 1, new_len)
                resampled_grain = np.interp(new_indices, old_indices, windowed_grain)

                # Create blend window for this grain
                blend_window = get_window('hann', len(resampled_grain))

                # Calculate output position (centered at original mark)
                out_center = mark_present
                out_start = out_center - len(resampled_grain) // 2
                out_end = out_start + len(resampled_grain)

                # Bounds checking with clipping
                grain_start_clip = 0
                grain_end_clip = len(resampled_grain)

                if out_start < 0:
                    grain_start_clip = -out_start
                    out_start = 0

                if out_end > len(output):
                    grain_end_clip = len(resampled_grain) - (out_end - len(output))
                    out_end = len(output)

                if out_end > out_start and grain_end_clip > grain_start_clip:
                    # Get the portion to add
                    grain_portion = resampled_grain[grain_start_clip:grain_end_clip]
                    window_portion = blend_window[grain_start_clip:grain_end_clip]

                    # Weighted overlap-add
                    output[out_start:out_end] += grain_portion * window_portion
                    weight_sum[out_start:out_end] += window_portion

        # Normalize by overlap weights to prevent amplitude modulation
        # This is crucial for smooth output
        epsilon = 1e-8
        weight_sum = np.maximum(weight_sum, epsilon)
        output = output / weight_sum

        # Final normalization to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.9

        return output

    def phase_vocoder(self, audio, pitch_shift_factor):
        """
        Pitch shift using librosa's phase vocoder.

        Args:
            audio: Input audio signal
            pitch_shift_factor: Pitch shift factor

        Returns:
            Pitch-shifted audio
        """
        # Convert factor to semitones
        semitones = 12 * np.log2(pitch_shift_factor)

        # Use librosa's pitch_shift
        shifted = librosa.effects.pitch_shift(
            y=audio,
            sr=self.sr,
            n_steps=semitones,
            n_fft=2048,
            hop_length=512
        )

        return shifted

    def wsola(self, audio, pitch_shift_factor):
        """
        WSOLA (Waveform Similarity Overlap-Add) pitch shifting.
        Similar to SoundTouch library algorithm.

        Strategy:
        1. First resample to change pitch (changes both pitch and duration)
        2. Then use WSOLA time-stretching to restore original duration

        This is simpler and more effective than the inverse approach.

        Args:
            audio: Input audio signal
            pitch_shift_factor: Pitch shift factor (>1 = higher, <1 = lower)

        Returns:
            Pitch-shifted audio with same duration as input
        """
        if abs(pitch_shift_factor - 1.0) < 0.001:
            return audio

        # Step 1: Resample to change pitch (this also changes duration)
        # pitch_shift_factor > 1: make shorter (higher pitch)
        # pitch_shift_factor < 1: make longer (lower pitch)
        resampled_len = int(len(audio) / pitch_shift_factor)

        if resampled_len > 0:
            audio_indices = np.linspace(0, len(audio) - 1, len(audio))
            resample_indices = np.linspace(0, len(audio) - 1, resampled_len)
            resampled = np.interp(resample_indices, audio_indices, audio)
        else:
            return audio

        # Step 2: Time-stretch back to original duration using WSOLA
        # This preserves the new pitch while fixing the duration
        target_len = len(audio)
        time_stretch_factor = len(resampled) / target_len

        # WSOLA time-stretching parameters
        frame_size = 2048
        hop_synthesis = 512  # Fixed output hop
        hop_analysis = int(hop_synthesis * time_stretch_factor)  # Variable input hop

        # Ensure minimum hop
        hop_analysis = max(hop_analysis, 32)

        # Template size for correlation
        template_size = frame_size // 4
        search_range = 50  # Search range for best match

        window = get_window('hann', frame_size)

        # Calculate output
        num_frames = int((target_len - frame_size) / hop_synthesis)
        output = np.zeros(target_len)

        input_pos = 0
        output_pos = 0

        for i in range(num_frames):
            # Natural input position
            natural_pos = int(i * hop_analysis)

            # Search for best matching position (waveform similarity)
            if i > 0 and natural_pos < len(resampled) - frame_size:
                # Get template from previous output
                template_start = max(0, output_pos - template_size)
                template = output[template_start:output_pos]

                # Search around natural position
                search_start = max(0, natural_pos - search_range)
                search_end = min(len(resampled) - frame_size, natural_pos + search_range)

                best_corr = -np.inf
                best_pos = natural_pos

                for pos in range(search_start, search_end):
                    # Get candidate
                    candidate = resampled[pos:pos + len(template)]
                    if len(candidate) == len(template):
                        # Cross-correlation
                        corr = np.sum(template * candidate)
                        if corr > best_corr:
                            best_corr = corr
                            best_pos = pos

                input_pos = best_pos
            else:
                input_pos = natural_pos

            # Extract frame
            if input_pos + frame_size <= len(resampled):
                frame = resampled[input_pos:input_pos + frame_size] * window
            else:
                # Pad if at end
                remaining = len(resampled) - input_pos
                if remaining > 0:
                    frame = np.pad(resampled[input_pos:], (0, frame_size - remaining), mode='constant')
                    frame = frame * window
                else:
                    break

            # Overlap-add
            end_pos = min(len(output), output_pos + frame_size)
            frame_len = end_pos - output_pos
            if frame_len > 0:
                output[output_pos:end_pos] += frame[:frame_len]

            output_pos += hop_synthesis

        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.9

        return output

    def shift_pitch_semitones(self, audio, semitones, method='phase_vocoder'):
        """
        Shift pitch by semitones using specified method.

        Args:
            audio: Input audio signal
            semitones: Number of semitones to shift
            method: 'td_psola', 'phase_vocoder', or 'wsola'

        Returns:
            Pitch-shifted audio
        """
        if abs(semitones) < 0.01:
            return audio

        # Convert semitones to pitch shift factor
        pitch_shift_factor = 2 ** (semitones / 12.0)

        print(f"Shifting by {semitones:+.1f} semitones using {method}")

        if method == 'td_psola':
            return self.td_psola(audio, pitch_shift_factor)
        elif method == 'phase_vocoder':
            return self.phase_vocoder(audio, pitch_shift_factor)
        elif method == 'wsola':
            return self.wsola(audio, pitch_shift_factor)
        else:
            raise ValueError(f"Unknown method: {method}")

    def shift_to_target_pitch(self, audio, target_pitch_hz, method='phase_vocoder'):
        """
        Shift audio to match a target pitch.

        Args:
            audio: Input audio signal
            target_pitch_hz: Target pitch in Hz
            method: Pitch shifting method to use

        Returns:
            Pitch-shifted audio
        """
        # Detect current pitch
        current_pitch = self.detect_pitch_from_audio(audio)

        # Calculate semitone shift
        semitones = 12 * np.log2(target_pitch_hz / current_pitch)

        print(f"Current pitch: {current_pitch:.2f} Hz")
        print(f"Target pitch: {target_pitch_hz:.2f} Hz")
        print(f"Shift: {semitones:+.2f} semitones")

        return self.shift_pitch_semitones(audio, semitones, method=method)

    def process_file(self, input_path, output_path, semitones=0,
                    target_pitch_hz=None, pitch_shift_factor=None, method='phase_vocoder'):
        """
        Process an audio file with pitch shifting.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            semitones: Semitones to shift (if specified)
            target_pitch_hz: Target pitch in Hz (if specified)
            pitch_shift_factor: Direct pitch shift factor (if specified)
            method: 'td_psola', 'phase_vocoder', or 'wsola'
        """
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sr, mono=True)

        print(f"Loaded audio: {len(audio)} samples ({len(audio)/sr:.2f}s), {sr} Hz")

        # Apply pitch shift
        if target_pitch_hz is not None:
            shifted = self.shift_to_target_pitch(audio, target_pitch_hz, method=method)
        elif pitch_shift_factor is not None:
            semitones_calc = 12 * np.log2(pitch_shift_factor)
            shifted = self.shift_pitch_semitones(audio, semitones_calc, method=method)
        else:
            shifted = self.shift_pitch_semitones(audio, semitones, method=method)

        print(f"Output audio: {len(shifted)} samples ({len(shifted)/sr:.2f}s)")

        # Save
        sf.write(output_path, shifted, sr)
        print(f"Saved pitch-shifted audio to: {output_path}")


def main():
    """Command-line interface for pitch shifting."""
    import argparse

    parser = argparse.ArgumentParser(description='Pitch shift audio using multiple methods')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('output', help='Output audio file')
    parser.add_argument('--semitones', type=float, default=0,
                       help='Semitones to shift (e.g., 2 for up 2 semitones, -3 for down 3)')
    parser.add_argument('--factor', type=float,
                       help='Direct pitch shift factor (e.g., 1.5 for 50%% up)')
    parser.add_argument('--target-pitch', type=float,
                       help='Target pitch in Hz')
    parser.add_argument('--method', type=str, default='phase_vocoder',
                       choices=['td_psola', 'phase_vocoder', 'wsola'],
                       help='Pitch shifting method')
    parser.add_argument('--sr', type=int, default=22050,
                       help='Sample rate')

    args = parser.parse_args()

    # Create processor
    shifter = PitchShifter(sr=args.sr)

    # Process
    shifter.process_file(args.input, args.output,
                        semitones=args.semitones,
                        target_pitch_hz=args.target_pitch,
                        pitch_shift_factor=args.factor,
                        method=args.method)


if __name__ == '__main__':
    main()
