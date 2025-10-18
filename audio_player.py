"""
Audio Player GUI for comparing original, vocal, and instrumental tracks
Apple-inspired design with seekable progress bar and pitch shifting
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas
import pygame
import pyaudio
import wave
import os
import sys
import threading
import time
import numpy as np
import librosa
import soundfile as sf
from mutagen import File as MutagenFile
from repet import REPET
from pitch_shift import PitchShifter


class RoundedButton(Canvas):
    """Custom rounded button widget inspired by Apple design."""

    def __init__(self, parent, text, command, bg_color="#007AFF", fg_color="white",
                 width=120, height=40, corner_radius=10, **kwargs):
        Canvas.__init__(self, parent, width=width, height=height,
                       bg=parent['bg'], highlightthickness=0, **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.corner_radius = corner_radius
        self.text = text
        self.width = width
        self.height = height
        self.is_disabled = False

        self.draw()
        self.bind("<Button-1>", self.on_click)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def draw(self, hover=False):
        self.delete("all")
        color = self.lighten_color(self.bg_color) if hover and not self.is_disabled else self.bg_color
        if self.is_disabled:
            color = "#D1D1D6"
            fg = "#999999"
        else:
            fg = self.fg_color

        # Draw rounded rectangle
        self.create_rounded_rect(2, 2, self.width-2, self.height-2,
                                self.corner_radius, fill=color, outline="")

        # Draw text
        self.create_text(self.width/2, self.height/2, text=self.text,
                        fill=fg, font=("SF Pro Text", 13, "bold"))

    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [x1+radius, y1,
                 x1+radius, y1,
                 x2-radius, y1,
                 x2-radius, y1,
                 x2, y1,
                 x2, y1+radius,
                 x2, y1+radius,
                 x2, y2-radius,
                 x2, y2-radius,
                 x2, y2,
                 x2-radius, y2,
                 x2-radius, y2,
                 x1+radius, y2,
                 x1+radius, y2,
                 x1, y2,
                 x1, y2-radius,
                 x1, y2-radius,
                 x1, y1+radius,
                 x1, y1+radius,
                 x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def lighten_color(self, color):
        # Simple color lightening for hover effect
        color_map = {
            "#007AFF": "#3395FF",
            "#34C759": "#5DD779",
            "#FF3B30": "#FF5B50",
            "#FF9500": "#FFB040",
            "#AF52DE": "#BF72EE",
            "#FF2D55": "#FF5775",
        }
        return color_map.get(color, color)

    def on_click(self, event):
        if not self.is_disabled and self.command:
            self.command()

    def on_enter(self, event):
        if not self.is_disabled:
            self.draw(hover=True)
            self.config(cursor="hand2")

    def on_leave(self, event):
        self.draw(hover=False)
        self.config(cursor="")

    def config_state(self, state):
        self.is_disabled = (state == "disabled")
        self.draw()


class AudioPlayer:
    """
    GUI-based audio player for comparing separated audio tracks.
    Apple-inspired design philosophy with pitch shifting.
    """

    def __init__(self, root):
        """
        Initialize the audio player.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("REPET Audio Separation")
        self.root.geometry("700x1000")
        self.root.resizable(True, True)

        # Apple-inspired color scheme
        self.bg_color = "#F2F2F7"
        self.card_bg = "#FFFFFF"
        self.text_primary = "#000000"
        self.text_secondary = "#8E8E93"
        self.accent_blue = "#007AFF"
        self.accent_green = "#34C759"
        self.accent_red = "#FF3B30"
        self.accent_orange = "#FF9500"
        self.accent_purple = "#AF52DE"
        self.accent_pink = "#FF2D55"

        self.root.configure(bg=self.bg_color)

        # Initialize pygame mixer
        pygame.mixer.init()

        # Audio file paths
        self.original_file = None
        self.vocal_file = None
        self.instrumental_file = None
        self.vocal_shifted_tdpsola = None
        self.vocal_shifted_phasevocoder = None
        self.vocal_shifted_wsola = None

        # Current playing track
        self.current_track = None
        self.current_filepath = None
        self.is_playing = False
        self.is_paused = False

        # Progress tracking
        self.audio_length = 0
        self.update_progress_job = None
        self.seeking = False
        self.start_time = 0

        # Pitch shifting
        self.pitch_shifter = PitchShifter(sr=22050)
        self.is_recording = False
        self.recorded_pitch = None

        # Create UI
        self.create_widgets()

    def create_card(self, parent, pady=10):
        """Create an Apple-style card container."""
        frame = tk.Frame(parent, bg=self.card_bg, relief=tk.FLAT)
        frame.pack(fill=tk.X, pady=pady, padx=20)
        frame.config(highlightbackground="#E5E5EA", highlightthickness=1)
        return frame

    def create_widgets(self):
        """Create all GUI widgets with Apple design."""

        # Header
        header = tk.Frame(self.root, bg=self.bg_color, height=100)
        header.pack(fill=tk.X, pady=(20, 10))
        header.pack_propagate(False)

        title = tk.Label(header,
                        text="REPET",
                        font=("SF Pro Display", 32, "bold"),
                        bg=self.bg_color,
                        fg=self.text_primary)
        title.pack(pady=(10, 0))

        subtitle = tk.Label(header,
                          text="Audio Source Separation & Pitch Shift",
                          font=("SF Pro Text", 14),
                          bg=self.bg_color,
                          fg=self.text_secondary)
        subtitle.pack()

        # Main container with scrollbar
        canvas = tk.Canvas(self.root, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.bg_color)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind mouse wheel scrolling (platform-specific)
        def _on_mousewheel(event):
            # macOS and Windows have different delta values
            if sys.platform == 'darwin':
                # macOS uses smaller delta values
                canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                # Windows
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            # Linux uses Button-4 and Button-5 for scroll
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        def _on_arrow_key(event):
            # Arrow key scrolling
            if event.keysym == 'Up':
                canvas.yview_scroll(-1, "units")
            elif event.keysym == 'Down':
                canvas.yview_scroll(1, "units")
            elif event.keysym == 'Prior':  # Page Up
                canvas.yview_scroll(-10, "units")
            elif event.keysym == 'Next':  # Page Down
                canvas.yview_scroll(10, "units")

        def _bind_to_mousewheel(event):
            if sys.platform == 'linux':
                canvas.bind_all("<Button-4>", _on_mousewheel_linux)
                canvas.bind_all("<Button-5>", _on_mousewheel_linux)
            else:
                canvas.bind_all("<MouseWheel>", _on_mousewheel)
            # Bind arrow keys
            canvas.bind_all("<Up>", _on_arrow_key)
            canvas.bind_all("<Down>", _on_arrow_key)
            canvas.bind_all("<Prior>", _on_arrow_key)
            canvas.bind_all("<Next>", _on_arrow_key)

        def _unbind_from_mousewheel(event):
            if sys.platform == 'linux':
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            else:
                canvas.unbind_all("<MouseWheel>")
            # Unbind arrow keys
            canvas.unbind_all("<Up>")
            canvas.unbind_all("<Down>")
            canvas.unbind_all("<Prior>")
            canvas.unbind_all("<Next>")

        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Store canvas for later use
        self.canvas = canvas

        # File selection card
        self.create_file_card(scrollable_frame)

        # Separation card
        self.create_separation_card(scrollable_frame)

        # Pitch shifting card
        self.create_pitch_card(scrollable_frame)

        # Track selection card
        self.create_track_card(scrollable_frame)

        # Playback card
        self.create_playback_card(scrollable_frame)

        # Status bar
        self.status_label = tk.Label(self.root,
                                     text="Ready",
                                     font=("SF Pro Text", 11),
                                     bg=self.bg_color,
                                     fg=self.text_secondary,
                                     anchor=tk.CENTER,
                                     pady=10)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def create_file_card(self, parent):
        """Create file selection card."""
        card = self.create_card(parent)

        inner = tk.Frame(card, bg=self.card_bg)
        inner.pack(fill=tk.X, padx=20, pady=20)

        label = tk.Label(inner,
                        text="Select Audio File",
                        font=("SF Pro Text", 16, "bold"),
                        bg=self.card_bg,
                        fg=self.text_primary)
        label.pack(anchor=tk.W, pady=(0, 10))

        btn_container = tk.Frame(inner, bg=self.card_bg)
        btn_container.pack(fill=tk.X, pady=5)

        self.load_btn = RoundedButton(btn_container,
                                      text="Choose File",
                                      command=self.load_audio,
                                      bg_color=self.accent_blue,
                                      width=150,
                                      height=36)
        self.load_btn.pack(anchor=tk.CENTER)

        self.file_label = tk.Label(inner,
                                   text="No file selected",
                                   font=("SF Pro Text", 13),
                                   bg=self.card_bg,
                                   fg=self.text_secondary,
                                   anchor=tk.W)
        self.file_label.pack(fill=tk.X, pady=(10, 0))

    def create_separation_card(self, parent):
        """Create separation card."""
        card = self.create_card(parent)

        inner = tk.Frame(card, bg=self.card_bg)
        inner.pack(fill=tk.X, padx=20, pady=20)

        label = tk.Label(inner,
                        text="Audio Separation",
                        font=("SF Pro Text", 16, "bold"),
                        bg=self.card_bg,
                        fg=self.text_primary)
        label.pack(anchor=tk.W, pady=(0, 10))

        self.separate_btn = RoundedButton(inner,
                                         text="Separate Audio",
                                         command=self.separate_audio,
                                         bg_color=self.accent_orange,
                                         width=200,
                                         height=40)
        self.separate_btn.config_state("disabled")
        self.separate_btn.pack(pady=5)

        self.progress_label = tk.Label(inner,
                                      text="",
                                      font=("SF Pro Text", 12),
                                      bg=self.card_bg,
                                      fg=self.text_secondary)
        self.progress_label.pack(pady=(10, 0))

    def create_pitch_card(self, parent):
        """Create pitch shifting card."""
        card = self.create_card(parent)

        inner = tk.Frame(card, bg=self.card_bg)
        inner.pack(fill=tk.X, padx=20, pady=20)

        label = tk.Label(inner,
                        text="Pitch Shifting (Vocal Only)",
                        font=("SF Pro Text", 16, "bold"),
                        bg=self.card_bg,
                        fg=self.text_primary)
        label.pack(anchor=tk.W, pady=(0, 15))

        # Pitch slider
        slider_frame = tk.Frame(inner, bg=self.card_bg)
        slider_frame.pack(fill=tk.X, pady=10)

        slider_label = tk.Label(slider_frame,
                               text="Pitch Shift (Semitones)",
                               font=("SF Pro Text", 13),
                               bg=self.card_bg,
                               fg=self.text_secondary)
        slider_label.pack(side=tk.LEFT, padx=(0, 10))

        self.pitch_value_label = tk.Label(slider_frame,
                                         text="0",
                                         font=("SF Pro Text", 13, "bold"),
                                         bg=self.card_bg,
                                         fg=self.text_primary,
                                         width=4)
        self.pitch_value_label.pack(side=tk.RIGHT)

        self.pitch_slider = ttk.Scale(slider_frame,
                                      from_=-12,
                                      to=12,
                                      orient=tk.HORIZONTAL,
                                      command=self.update_pitch_label)
        self.pitch_slider.set(0)
        self.pitch_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Recording section
        record_frame = tk.Frame(inner, bg=self.card_bg)
        record_frame.pack(fill=tk.X, pady=(10, 5))

        record_label = tk.Label(record_frame,
                               text='Or record "Ahhhh" for target pitch:',
                               font=("SF Pro Text", 13),
                               bg=self.card_bg,
                               fg=self.text_secondary)
        record_label.pack(pady=(0, 10))

        self.record_btn = RoundedButton(record_frame,
                                       text="🎤 Record",
                                       command=self.toggle_recording,
                                       bg_color=self.accent_pink,
                                       width=140,
                                       height=40)
        self.record_btn.pack(pady=5)

        # Three separate apply buttons
        btn_grid = tk.Frame(inner, bg=self.card_bg)
        btn_grid.pack(fill=tk.X, pady=(15, 5))

        methods_label = tk.Label(btn_grid,
                                text="Choose Pitch Shifting Method:",
                                font=("SF Pro Text", 13, "bold"),
                                bg=self.card_bg,
                                fg=self.text_primary)
        methods_label.pack(pady=(0, 10))

        self.apply_tdpsola_btn = RoundedButton(btn_grid,
                                              text="Apply TD-PSOLA",
                                              command=lambda: self.apply_pitch_shift('td_psola'),
                                              bg_color=self.accent_orange,
                                              width=180,
                                              height=40)
        self.apply_tdpsola_btn.config_state("disabled")
        self.apply_tdpsola_btn.pack(pady=3)

        self.apply_phasevocoder_btn = RoundedButton(btn_grid,
                                                   text="Apply Phase Vocoder",
                                                   command=lambda: self.apply_pitch_shift('phase_vocoder'),
                                                   bg_color=self.accent_blue,
                                                   width=180,
                                                   height=40)
        self.apply_phasevocoder_btn.config_state("disabled")
        self.apply_phasevocoder_btn.pack(pady=3)

        self.apply_wsola_btn = RoundedButton(btn_grid,
                                            text="Apply WSOLA",
                                            command=lambda: self.apply_pitch_shift('wsola'),
                                            bg_color=self.accent_green,
                                            width=180,
                                            height=40)
        self.apply_wsola_btn.config_state("disabled")
        self.apply_wsola_btn.pack(pady=3)

        self.pitch_status_label = tk.Label(inner,
                                          text="",
                                          font=("SF Pro Text", 12),
                                          bg=self.card_bg,
                                          fg=self.text_secondary)
        self.pitch_status_label.pack(pady=(10, 0))

    def create_track_card(self, parent):
        """Create track selection card."""
        card = self.create_card(parent)

        inner = tk.Frame(card, bg=self.card_bg)
        inner.pack(fill=tk.X, padx=20, pady=20)

        label = tk.Label(inner,
                        text="Select Track",
                        font=("SF Pro Text", 16, "bold"),
                        bg=self.card_bg,
                        fg=self.text_primary)
        label.pack(anchor=tk.W, pady=(0, 15))

        btn_frame = tk.Frame(inner, bg=self.card_bg)
        btn_frame.pack()

        self.original_btn = RoundedButton(btn_frame,
                                         text="Original",
                                         command=lambda: self.play_track('original'),
                                         bg_color=self.accent_purple,
                                         width=180,
                                         height=44)
        self.original_btn.config_state("disabled")
        self.original_btn.pack(pady=4)

        self.vocal_btn = RoundedButton(btn_frame,
                                       text="Vocal",
                                       command=lambda: self.play_track('vocal'),
                                       bg_color=self.accent_blue,
                                       width=180,
                                       height=44)
        self.vocal_btn.config_state("disabled")
        self.vocal_btn.pack(pady=4)

        self.instrumental_btn = RoundedButton(btn_frame,
                                             text="Instrumental",
                                             command=lambda: self.play_track('instrumental'),
                                             bg_color=self.accent_red,
                                             width=180,
                                             height=44)
        self.instrumental_btn.config_state("disabled")
        self.instrumental_btn.pack(pady=4)

        self.vocal_tdpsola_btn = RoundedButton(btn_frame,
                                              text="Vocal (TD-PSOLA)",
                                              command=lambda: self.play_track('vocal_tdpsola'),
                                              bg_color=self.accent_orange,
                                              width=180,
                                              height=44)
        self.vocal_tdpsola_btn.config_state("disabled")
        self.vocal_tdpsola_btn.pack(pady=4)

        self.vocal_phasevocoder_btn = RoundedButton(btn_frame,
                                                   text="Vocal (Phase Vocoder)",
                                                   command=lambda: self.play_track('vocal_phasevocoder'),
                                                   bg_color=self.accent_blue,
                                                   width=180,
                                                   height=44)
        self.vocal_phasevocoder_btn.config_state("disabled")
        self.vocal_phasevocoder_btn.pack(pady=4)

        self.vocal_wsola_btn = RoundedButton(btn_frame,
                                            text="Vocal (WSOLA)",
                                            command=lambda: self.play_track('vocal_wsola'),
                                            bg_color=self.accent_green,
                                            width=180,
                                            height=44)
        self.vocal_wsola_btn.config_state("disabled")
        self.vocal_wsola_btn.pack(pady=4)

        self.current_track_label = tk.Label(inner,
                                           text="",
                                           font=("SF Pro Text", 13),
                                           bg=self.card_bg,
                                           fg=self.text_secondary)
        self.current_track_label.pack(pady=(15, 0))

    def create_playback_card(self, parent):
        """Create playback controls card."""
        card = self.create_card(parent)

        inner = tk.Frame(card, bg=self.card_bg)
        inner.pack(fill=tk.X, padx=20, pady=20)

        # Progress section
        progress_label = tk.Label(inner,
                                 text="Playback",
                                 font=("SF Pro Text", 16, "bold"),
                                 bg=self.card_bg,
                                 fg=self.text_primary)
        progress_label.pack(anchor=tk.W, pady=(0, 15))

        # Time labels
        time_frame = tk.Frame(inner, bg=self.card_bg)
        time_frame.pack(fill=tk.X, pady=(0, 5))

        self.current_time_label = tk.Label(time_frame,
                                          text="0:00",
                                          font=("SF Pro Text", 12),
                                          bg=self.card_bg,
                                          fg=self.text_secondary)
        self.current_time_label.pack(side=tk.LEFT)

        self.total_time_label = tk.Label(time_frame,
                                        text="0:00",
                                        font=("SF Pro Text", 12),
                                        bg=self.card_bg,
                                        fg=self.text_secondary)
        self.total_time_label.pack(side=tk.RIGHT)

        # Progress bar
        progress_frame = tk.Frame(inner, bg=self.card_bg, height=30)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_canvas = Canvas(progress_frame,
                                      height=6,
                                      bg=self.card_bg,
                                      highlightthickness=0)
        self.progress_canvas.pack(fill=tk.X)

        self.progress_var = tk.DoubleVar(value=0)
        self.draw_progress_bar()

        self.progress_canvas.bind("<Button-1>", self.on_progress_click)
        self.progress_canvas.bind("<B1-Motion>", self.on_progress_drag)
        self.progress_canvas.bind("<ButtonRelease-1>", self.on_progress_release)

        # Control buttons
        control_frame = tk.Frame(inner, bg=self.card_bg)
        control_frame.pack(pady=(20, 10))

        self.play_pause_btn = RoundedButton(control_frame,
                                           text="▶ Play",
                                           command=self.toggle_play_pause,
                                           bg_color=self.accent_green,
                                           width=140,
                                           height=50,
                                           corner_radius=25)
        self.play_pause_btn.config_state("disabled")
        self.play_pause_btn.grid(row=0, column=0, padx=8)

        self.stop_btn = RoundedButton(control_frame,
                                      text="■ Stop",
                                      command=self.stop_playback,
                                      bg_color=self.accent_red,
                                      width=140,
                                      height=50,
                                      corner_radius=25)
        self.stop_btn.config_state("disabled")
        self.stop_btn.grid(row=0, column=1, padx=8)

        # Volume control
        vol_frame = tk.Frame(inner, bg=self.card_bg)
        vol_frame.pack(fill=tk.X, pady=(15, 0))

        vol_label = tk.Label(vol_frame,
                            text="Volume",
                            font=("SF Pro Text", 13),
                            bg=self.card_bg,
                            fg=self.text_secondary)
        vol_label.pack(side=tk.LEFT, padx=(0, 10))

        self.volume_label = tk.Label(vol_frame,
                                     text="70%",
                                     font=("SF Pro Text", 13, "bold"),
                                     bg=self.card_bg,
                                     fg=self.text_primary,
                                     width=4)
        self.volume_label.pack(side=tk.RIGHT)

        self.volume_scale = ttk.Scale(vol_frame,
                                      from_=0,
                                      to=100,
                                      orient=tk.HORIZONTAL,
                                      command=self.change_volume)
        self.volume_scale.set(70)
        self.volume_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

    def update_pitch_label(self, value):
        """Update pitch shift label."""
        semitones = int(float(value))
        self.pitch_value_label.config(text=f"{semitones:+d}")
        # Reset recorded pitch when slider is moved
        if abs(semitones) > 0.1:
            self.recorded_pitch = None
            self.pitch_status_label.config(text="Using slider value", fg=self.text_secondary)

    def toggle_recording(self):
        """Toggle audio recording."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.record_btn.text = "⏹ Stop Recording"
        self.record_btn.bg_color = self.accent_red
        self.record_btn.draw()
        self.pitch_status_label.config(text="🎤 Recording... Say 'Ahhhh'", fg=self.accent_red)

        # Run recording in thread
        thread = threading.Thread(target=self._record_thread)
        thread.daemon = True
        thread.start()

    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        self.record_btn.text = "🎤 Record"
        self.record_btn.bg_color = self.accent_pink
        self.record_btn.draw()

    def _record_thread(self):
        """Background thread for audio recording."""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 22050
        RECORD_SECONDS = 3

        p = pyaudio.PyAudio()

        try:
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)

            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                if not self.is_recording:
                    break
                data = stream.read(CHUNK)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            # Save recording
            temp_file = "temp_recording.wav"
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Detect pitch
            audio, sr = librosa.load(temp_file, sr=22050)
            self.recorded_pitch = self.pitch_shifter.detect_pitch_from_audio(audio)

            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

            # Update UI
            self.root.after(0, lambda: self.pitch_status_label.config(
                text=f"Detected pitch: {self.recorded_pitch:.1f} Hz",
                fg=self.accent_green))

        except Exception as e:
            self.root.after(0, lambda: self.pitch_status_label.config(
                text=f"Recording error: {str(e)}",
                fg=self.accent_red))
        finally:
            p.terminate()
            self.is_recording = False
            self.root.after(0, lambda: self.record_btn.config_state("normal"))

    def apply_pitch_shift(self, method):
        """Apply pitch shift to vocal track with specified method."""
        if not self.vocal_file or not os.path.exists(self.vocal_file):
            messagebox.showerror("Error", "Please separate audio first to get vocal track")
            return

        # Disable all buttons
        self.apply_tdpsola_btn.config_state("disabled")
        self.apply_phasevocoder_btn.config_state("disabled")
        self.apply_wsola_btn.config_state("disabled")

        method_names = {
            'td_psola': 'TD-PSOLA',
            'phase_vocoder': 'Phase Vocoder',
            'wsola': 'WSOLA'
        }
        self.pitch_status_label.config(text=f"Processing with {method_names[method]}...", fg=self.accent_orange)

        # Run in thread
        thread = threading.Thread(target=self._pitch_shift_thread, args=(method,))
        thread.daemon = True
        thread.start()

    def _pitch_shift_thread(self, method):
        """Background thread for pitch shifting."""
        try:
            base_name = os.path.splitext(self.vocal_file)[0]

            # Set output file based on method
            if method == 'td_psola':
                output_file = f"{base_name}_tdpsola.wav"
                self.vocal_shifted_tdpsola = output_file
            elif method == 'phase_vocoder':
                output_file = f"{base_name}_phasevocoder.wav"
                self.vocal_shifted_phasevocoder = output_file
            elif method == 'wsola':
                output_file = f"{base_name}_wsola.wav"
                self.vocal_shifted_wsola = output_file

            # Get pitch shift parameters
            if self.recorded_pitch is not None:
                # Use recorded pitch as target
                self.pitch_shifter.process_file(
                    self.vocal_file,
                    output_file,
                    target_pitch_hz=self.recorded_pitch,
                    method=method
                )
                message = f"Pitched to {self.recorded_pitch:.1f} Hz using {method}"
            else:
                # Use slider value
                semitones = int(self.pitch_slider.get())
                self.pitch_shifter.process_file(
                    self.vocal_file,
                    output_file,
                    semitones=semitones,
                    method=method
                )
                message = f"Shifted by {semitones:+d} semitones using {method}"

            # Update UI
            self.root.after(0, lambda: self._pitch_shift_complete(message, method))

        except Exception as e:
            error_msg = f"Pitch shift error: {str(e)}"
            self.root.after(0, lambda: self._pitch_shift_error(error_msg))

    def _pitch_shift_complete(self, message, method):
        """Called when pitch shift is complete."""
        self.pitch_status_label.config(text=f"✓ {message}", fg=self.accent_green)

        # Re-enable all buttons
        self.apply_tdpsola_btn.config_state("normal")
        self.apply_phasevocoder_btn.config_state("normal")
        self.apply_wsola_btn.config_state("normal")

        # Enable corresponding play button
        if method == 'td_psola':
            self.vocal_tdpsola_btn.config_state("normal")
        elif method == 'phase_vocoder':
            self.vocal_phasevocoder_btn.config_state("normal")
        elif method == 'wsola':
            self.vocal_wsola_btn.config_state("normal")

        messagebox.showinfo("Success", f"Pitch shifting complete!\n\n{message}")

    def _pitch_shift_error(self, error_msg):
        """Called when pitch shift encounters an error."""
        self.pitch_status_label.config(text="Pitch shift failed", fg=self.accent_red)
        self.apply_tdpsola_btn.config_state("normal")
        self.apply_phasevocoder_btn.config_state("normal")
        self.apply_wsola_btn.config_state("normal")
        messagebox.showerror("Error", error_msg)

    def draw_progress_bar(self):
        """Draw custom rounded progress bar."""
        self.progress_canvas.delete("all")
        width = self.progress_canvas.winfo_width()
        if width <= 1:
            width = 600

        # Background track
        self.progress_canvas.create_rectangle(0, 0, width, 6,
                                             fill="#E5E5EA",
                                             outline="")

        # Progress fill
        progress = self.progress_var.get()
        fill_width = (progress / 100) * width
        if fill_width > 0:
            self.progress_canvas.create_rectangle(0, 0, fill_width, 6,
                                                 fill=self.accent_blue,
                                                 outline="")

    def on_progress_click(self, event):
        """Handle progress bar click."""
        self.seeking = True
        self.update_progress_from_mouse(event.x)

    def on_progress_drag(self, event):
        """Handle progress bar drag."""
        if self.seeking:
            self.update_progress_from_mouse(event.x)

    def on_progress_release(self, event):
        """Handle progress bar release."""
        if self.seeking:
            self.update_progress_from_mouse(event.x)

            if self.is_playing and self.audio_length > 0:
                progress_percent = self.progress_var.get()
                seek_time = (progress_percent / 100) * self.audio_length

                was_paused = self.is_paused

                pygame.mixer.music.stop()
                pygame.mixer.music.load(self.current_filepath)
                pygame.mixer.music.play(start=seek_time)

                if was_paused:
                    pygame.mixer.music.pause()
                else:
                    self.start_time = time.time() - seek_time

            self.seeking = False

    def update_progress_from_mouse(self, x):
        """Update progress based on mouse position."""
        width = self.progress_canvas.winfo_width()
        if width > 0:
            progress = (x / width) * 100
            progress = max(0, min(100, progress))
            self.progress_var.set(progress)
            self.draw_progress_bar()

            if self.audio_length > 0:
                seek_time = (progress / 100) * self.audio_length
                self.current_time_label.config(text=self.format_time(seek_time))

    def load_audio(self):
        """Load an audio file."""
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All Files", "*.*")
            ]
        )

        if filepath:
            self.original_file = filepath
            filename = os.path.basename(filepath)
            self.file_label.config(text=filename, fg=self.text_primary)
            self.separate_btn.config_state("normal")
            self.original_btn.config_state("normal")
            self.status_label.config(text=f"Loaded: {filename}")

            # Reset separated files
            self.vocal_file = None
            self.instrumental_file = None
            self.vocal_shifted_tdpsola = None
            self.vocal_shifted_phasevocoder = None
            self.vocal_shifted_wsola = None
            self.vocal_btn.config_state("disabled")
            self.instrumental_btn.config_state("disabled")
            self.vocal_tdpsola_btn.config_state("disabled")
            self.vocal_phasevocoder_btn.config_state("disabled")
            self.vocal_wsola_btn.config_state("disabled")
            self.apply_tdpsola_btn.config_state("disabled")
            self.apply_phasevocoder_btn.config_state("disabled")
            self.apply_wsola_btn.config_state("disabled")

    def separate_audio(self):
        """Separate audio into vocal and instrumental tracks."""
        if not self.original_file:
            messagebox.showerror("Error", "Please load an audio file first")
            return

        self.separate_btn.config_state("disabled")
        self.progress_label.config(text="Processing... Please wait", fg=self.accent_orange)

        thread = threading.Thread(target=self._separate_audio_thread)
        thread.daemon = True
        thread.start()

    def _separate_audio_thread(self):
        """Background thread for audio separation."""
        try:
            base_name = os.path.splitext(self.original_file)[0]
            self.vocal_file = f"{base_name}_vocal.wav"
            self.instrumental_file = f"{base_name}_instrumental.wav"

            repet = REPET(n_fft=2048, hop_length=512)
            repet.separate(self.original_file,
                          output_vocal=self.vocal_file,
                          output_instrumental=self.instrumental_file)

            self.root.after(0, self._separation_complete)

        except Exception as e:
            error_msg = f"Error during separation: {str(e)}"
            self.root.after(0, lambda: self._separation_error(error_msg))

    def _separation_complete(self):
        """Called when separation is complete."""
        self.progress_label.config(text="Separation complete!", fg=self.accent_green)
        self.separate_btn.config_state("normal")
        self.vocal_btn.config_state("normal")
        self.instrumental_btn.config_state("normal")
        self.apply_tdpsola_btn.config_state("normal")
        self.apply_phasevocoder_btn.config_state("normal")
        self.apply_wsola_btn.config_state("normal")
        self.status_label.config(text="Ready to compare tracks or apply pitch shift")
        messagebox.showinfo("Success",
                          f"Audio separated successfully!\n\n"
                          f"Vocal: {os.path.basename(self.vocal_file)}\n"
                          f"Instrumental: {os.path.basename(self.instrumental_file)}")

    def _separation_error(self, error_msg):
        """Called when separation encounters an error."""
        self.progress_label.config(text="Separation failed", fg=self.accent_red)
        self.separate_btn.config_state("normal")
        self.status_label.config(text="Error during separation")
        messagebox.showerror("Error", error_msg)

    def get_audio_length(self, filepath):
        """Get the length of an audio file in seconds."""
        try:
            audio = MutagenFile(filepath)
            if audio is not None and hasattr(audio.info, 'length'):
                return audio.info.length
        except:
            pass

        try:
            sound = pygame.mixer.Sound(filepath)
            return sound.get_length()
        except:
            return 0

    def play_track(self, track_type):
        """Play a specific track."""
        if self.is_playing:
            pygame.mixer.music.stop()

        if track_type == 'original':
            filepath = self.original_file
            track_name = "Original"
        elif track_type == 'vocal':
            filepath = self.vocal_file
            track_name = "Vocal"
        elif track_type == 'instrumental':
            filepath = self.instrumental_file
            track_name = "Instrumental"
        elif track_type == 'vocal_tdpsola':
            filepath = self.vocal_shifted_tdpsola
            track_name = "Vocal (TD-PSOLA)"
        elif track_type == 'vocal_phasevocoder':
            filepath = self.vocal_shifted_phasevocoder
            track_name = "Vocal (Phase Vocoder)"
        elif track_type == 'vocal_wsola':
            filepath = self.vocal_shifted_wsola
            track_name = "Vocal (WSOLA)"
        else:
            return

        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", f"{track_name} track not available")
            return

        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()

            self.current_track = track_type
            self.current_filepath = filepath
            self.is_playing = True
            self.is_paused = False
            self.start_time = time.time()

            self.audio_length = self.get_audio_length(filepath)
            self.total_time_label.config(text=self.format_time(self.audio_length))

            self.progress_var.set(0)
            self.current_time_label.config(text="0:00")
            self.draw_progress_bar()

            self.update_progress()

            self.play_pause_btn.config_state("normal")
            self.play_pause_btn.text = "⏸ Pause"
            self.play_pause_btn.draw()
            self.stop_btn.config_state("normal")
            self.current_track_label.config(text=f"Now Playing: {track_name}",
                                           fg=self.accent_blue)
            self.status_label.config(text=f"Playing: {track_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {str(e)}")

    def update_progress(self):
        """Update progress bar and time labels."""
        if self.is_playing and not self.is_paused and not self.seeking:
            elapsed = time.time() - self.start_time

            if self.audio_length > 0 and elapsed <= self.audio_length:
                progress_percent = (elapsed / self.audio_length) * 100
                self.progress_var.set(progress_percent)
                self.draw_progress_bar()
                self.current_time_label.config(text=self.format_time(elapsed))

            if pygame.mixer.music.get_busy():
                self.update_progress_job = self.root.after(100, self.update_progress)
            else:
                self.stop_playback()

    def format_time(self, seconds):
        """Format seconds to MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def toggle_play_pause(self):
        """Toggle between play and pause."""
        if not self.is_playing and not self.is_paused:
            return

        if self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            elapsed_before_pause = self.progress_var.get() / 100 * self.audio_length
            self.start_time = time.time() - elapsed_before_pause
            self.play_pause_btn.text = "⏸ Pause"
            self.play_pause_btn.draw()
            self.status_label.config(text="Playing...")
            self.update_progress()
        else:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.play_pause_btn.text = "▶ Resume"
            self.play_pause_btn.draw()
            self.status_label.config(text="Paused")

    def stop_playback(self):
        """Stop current playback."""
        pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False

        if self.update_progress_job:
            self.root.after_cancel(self.update_progress_job)
            self.update_progress_job = None

        self.progress_var.set(0)
        self.draw_progress_bar()
        self.current_time_label.config(text="0:00")

        self.play_pause_btn.text = "▶ Play"
        self.play_pause_btn.config_state("disabled")
        self.stop_btn.config_state("disabled")
        self.current_track_label.config(text="", fg=self.text_secondary)
        self.status_label.config(text="Stopped")

    def change_volume(self, value):
        """Change playback volume."""
        volume = float(value) / 100
        pygame.mixer.music.set_volume(volume)
        self.volume_label.config(text=f"{int(float(value))}%")


def main():
    """Launch the audio player application."""
    root = tk.Tk()
    app = AudioPlayer(root)
    root.mainloop()


if __name__ == '__main__':
    main()
