# REPET Audio Separation - Android Prototype

This is a simple Android Studio prototype for the REPET audio separation project.

## Features

- Simple and clean UI
- Audio file selection from device storage
- Processing simulation for audio separation
- Minimal dependencies

## Project Structure

```
android-prototype/
├── app/
│   ├── build.gradle                      # App-level Gradle configuration
│   ├── proguard-rules.pro                # ProGuard rules
│   └── src/main/
│       ├── AndroidManifest.xml           # App manifest
│       ├── java/com/repet/audioseparation/
│       │   └── MainActivity.java         # Main activity
│       └── res/
│           ├── layout/
│           │   └── activity_main.xml     # Main UI layout
│           └── values/
│               ├── strings.xml           # String resources
│               └── colors.xml            # Color resources
├── build.gradle                          # Project-level Gradle
├── settings.gradle                       # Gradle settings
└── gradle.properties                     # Gradle properties
```

## How to Build

1. Open Android Studio
2. Select "Open an Existing Project"
3. Navigate to the `android-prototype` folder
4. Wait for Gradle sync to complete
5. Click "Run" or press Shift+F10

## Requirements

- Android Studio Arctic Fox or later
- Android SDK 24+ (Android 7.0+)
- Java 8+

## Current Implementation

This prototype currently includes:

- **File Selection**: Choose audio files from device storage
- **UI Framework**: Basic Material Design interface
- **Processing Simulation**: Demonstrates the workflow (actual REPET algorithm integration pending)

## Next Steps

To integrate the actual REPET algorithm:

1. Port the Python algorithm to Java/Kotlin
2. Use audio processing libraries like Tarsos DSP or Superpowered SDK
3. Implement FFT and STFT operations
4. Add real-time progress tracking
5. Implement playback controls for separated audio

## Permissions

The app requests the following permissions:
- `READ_EXTERNAL_STORAGE`: To read audio files
- `WRITE_EXTERNAL_STORAGE`: To save separated audio files

## License

Same as parent project
