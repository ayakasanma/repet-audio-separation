# REPET Audio Source Separation

This project implements the REPET (REpeating Pattern Extraction Technique) algorithm for separating vocals and instrumental tracks from audio files.

## Features

- **REPET Algorithm**: Separates repeating background (instrumental) from non-repeating foreground (vocals)
- **High-Quality Pitch Shifting**: Uses librosa's phase vocoder for professional-grade pitch shifting
- **GUI Audio Player**: Built-in player for convenient comparison of original, vocal, and instrumental tracks
- **Voice Recording**: Record your voice to detect target pitch for automatic pitch matching
- **Command-line Interface**: Can be used as a standalone script or through the GUI

## Algorithm Overview

REPET works by:
1. Computing the Short-Time Fourier Transform (STFT) of the audio
2. Detecting repeating patterns in the spectrogram (typically instrumental parts)
3. Extracting the repeating pattern using median filtering across time segments
4. Creating masks to separate vocal and instrumental components
5. Reconstructing the separated audio signals

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### GUI Application (Recommended)

Launch the audio player with GUI:
```bash
python audio_player.py
```

**Steps:**
1. Click "Choose File" to select an input audio file
2. Click "Separate Audio" to process the file (this may take a minute)
3. (Optional) Apply pitch shifting:
   - Adjust the pitch slider (-12 to +12 semitones)
   - OR click "Record" and say "Ahhhh" to detect target pitch
   - Click "Apply Pitch Shift" to process
4. Use the playback buttons to compare:
   - **Original**: Listen to the original audio
   - **Vocal**: Listen to extracted vocals
   - **Instrumental**: Listen to extracted instrumental
   - **Vocal (Pitch Shifted)**: Listen to pitch-shifted vocals

### Command-Line Interface

**REPET Separation:**
```bash
python repet.py input_audio.wav --vocal vocal_output.wav --instrumental instrumental_output.wav
```

**Options:**
- `--vocal`: Output filename for vocal track (default: vocal.wav)
- `--instrumental`: Output filename for instrumental track (default: instrumental.wav)
- `--n-fft`: FFT window size (default: 2048)
- `--hop-length`: Hop length between frames (default: 512)

**Pitch Shifting:**
```bash
python pitch_shift.py input.wav output.wav --semitones 3
python pitch_shift.py input.wav output.wav --target-pitch 440
```

**Options:**
- `--semitones`: Shift by semitones (e.g., 2, -3)
- `--target-pitch`: Target pitch in Hz (e.g., 440)
- `--sr`: Sample rate (default: 22050)

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- OGG (.ogg)
- M4A (.m4a)

## Requirements

- Python 3.7+
- numpy
- librosa
- soundfile
- scipy
- pygame

## How It Works

### REPET Algorithm Steps:

1. **STFT Computation**: Convert audio to frequency domain
2. **Period Detection**: Find repeating patterns using autocorrelation
3. **Pattern Extraction**: Extract repeating background using median filtering
4. **Mask Generation**: Create soft masks for instrumental and vocal separation
5. **Signal Reconstruction**: Apply masks and convert back to time domain

### Audio Player Features:

- Load any supported audio format
- Process separation in background thread (non-blocking)
- Play and compare all three versions (original, vocal, instrumental)
- Volume control
- Play/Pause/Stop controls
- Real-time status updates

## Limitations

- Works best with music that has clear repeating patterns (most pop/rock songs)
- Separation quality depends on the complexity of the mix
- Processing time increases with file length
- Best results with songs that have distinct vocal and instrumental sections

## Tips for Best Results

1. Use high-quality audio files (lossless formats like WAV or FLAC)
2. The algorithm works best with songs that have:
   - Clear repeating instrumental patterns
   - Distinct vocal melodies
   - Good separation between vocal and instrumental frequencies
3. Adjust `n_fft` and `hop_length` parameters for different time-frequency resolutions

## Project Structure

```
final/
├── repet.py              # REPET algorithm implementation
├── pitch_shift.py        # Pitch shifting using phase vocoder
├── audio_player.py       # GUI audio player application
├── test_repet.py         # Testing suite
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Example Workflow

1. Run the GUI: `python audio_player.py`
2. Load your audio file
3. Click "Separate Audio" and wait for processing
4. Compare the results using the playback buttons
5. Separated files are saved in the same directory as the input file

## Technical Details

- **FFT Size**: 2048 samples (default)
- **Hop Length**: 512 samples (default)
- **Window**: Hann window (librosa default)
- **Mask Type**: Soft mask using Wiener-like filtering
- **Period Detection**: Autocorrelation-based beat spectrum analysis

## Future Enhancements

- Real-time visualization of spectrograms
- Parameter tuning interface in GUI
- Batch processing support
- Additional separation algorithms (e.g., REPET-SIM)
- Export to different audio formats
