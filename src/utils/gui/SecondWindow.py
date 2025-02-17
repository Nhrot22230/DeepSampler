import os
import sys

import librosa
import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np
import soundfile as sf
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QVBoxLayout, QWidget)
from widgets.toolbar import Toolbar


class SecondWindow(QMainWindow):
    def __init__(self, file_path, main_window, selected_model):
        super().__init__()
        self.main_window = main_window
        self.file_path = file_path
        self.separated_files = {}
        self.process_audio()
        self.selected_model = selected_model
        self.initUI()
        self.setWindowTitle("DinoSampler")
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        logo_path = os.path.join(scriptDir, "assets", "dinosampler_logo.png")
        self.setWindowIcon(QIcon(logo_path))

    def process_audio(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        converted_dir = os.path.join(script_dir, "converted_tracks")

        os.makedirs(converted_dir, exist_ok=True)

        if self.file_path.endswith(".mp4"):
            base_name = os.path.basename(self.file_path).replace(".mp4", ".wav")
            wav_path = os.path.join(converted_dir, base_name)

            # Convertir MP4 a WAV
            video = mp.AudioFileClip(self.file_path)
            video.write_audiofile(wav_path, codec="pcm_s16le")

            # Usar el nuevo archivo convertido
            self.file_path = wav_path

    def initUI(self):
        self.toolbar = Toolbar(self, self.selected_model)
        self.toolbar.create_toolbar()
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        layout = QVBoxLayout(centralWidget)
        infoLayout = QHBoxLayout()

        self.label = QLabel(f"Audio File: {self.file_path}")
        self.duration_label = QLabel()
        infoLayout.addWidget(self.label)
        infoLayout.addStretch()
        infoLayout.addWidget(self.duration_label)
        layout.addLayout(infoLayout)

        self.canvas = FigureCanvas(plt.figure(figsize=(6, 1)))
        layout.addWidget(self.canvas)
        self.plot_waveform()

        separateBtn = QPushButton("Separate")
        separateBtn.clicked.connect(self.separate)
        layout.addWidget(separateBtn)

        self.track_labels = ["Vocals", "Drums", "Bass", "Other"]
        self.waveform_canvases = []
        self.download_buttons = []

        for label in self.track_labels:
            track_layout = QVBoxLayout()
            track_label = QLabel(label)
            track_layout.addWidget(track_label)

            track_sound_layout = QHBoxLayout()
            canvas = FigureCanvas(plt.figure(figsize=(6, 1)))
            canvas.figure.set_facecolor("black")
            download_btn = QPushButton()
            scriptDir = os.path.dirname(os.path.realpath(__file__))
            icon_path = os.path.join(scriptDir, "assets", "download_icon.png")
            download_btn.setIcon(QIcon(icon_path))
            download_btn.setFixedSize(32, 32)
            download_btn.setEnabled(False)
            download_btn.clicked.connect(lambda _, lbl=label: self.download_file(lbl))

            track_sound_layout.addWidget(canvas)
            track_sound_layout.addWidget(download_btn)

            track_layout.addLayout(track_sound_layout)

            layout.addLayout(track_layout)
            self.waveform_canvases.append(canvas)
            self.download_buttons.append(download_btn)

    def plot_waveform(self):
        y, sr = librosa.load(
            self.file_path, sr=44100
        )  # Carga el audio sin modificar SR
        time = np.linspace(0, len(y) / sr, len(y))  # Eje de tiempo
        duration = len(y) / sr  # Duración en segundos

        minutes = int(duration // 60)
        seconds = int(duration % 60)
        formatted_duration = f"{minutes}:{seconds:02d}"

        self.duration_label.setText(f"Duration: {formatted_duration}")

        self.canvas.figure.set_facecolor("black")
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()
        ax.set_facecolor("black")
        ax.fill_between(time, y, color="deepskyblue", alpha=0.5)
        ax.plot(time, y, color="cyan", lw=0.25, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.margins(0)
        self.canvas.draw()

    def separate(self):
        y, sr = librosa.load(self.file_path, sr=None)
        time = np.linspace(0, len(y) / sr, len(y))

        output_dir = os.path.join(os.path.dirname(__file__), "temp_tracks")
        os.makedirs(output_dir, exist_ok=True)

        for i, (canvas, label, btn) in enumerate(
            zip(self.waveform_canvases, self.track_labels, self.download_buttons)
        ):
            ax = canvas.figure.add_subplot(111)
            ax.clear()
            ax.set_facecolor("black")
            ax.fill_between(time, y, color="deepskyblue", alpha=0.5)
            ax.plot(time, y, color="cyan", lw=0.25, alpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.margins(0)
            canvas.draw()

            output_filename = os.path.join(output_dir, f"{label.lower()}_separated.wav")
            sf.write(output_filename, y, sr)  # Guardar el audio en un archivo
            self.separated_files[label] = output_filename  # Guardar en el diccionario

            btn.setEnabled(True)

    def download_file(self, label):
        if label in self.separated_files:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save File", self.separated_files[label], "Audio Files (*.wav)"
            )
            if save_path:
                sf.write(
                    save_path,
                    librosa.load(self.separated_files[label], sr=None)[0],
                    librosa.load(self.separated_files[label], sr=None)[1],
                )

    def new_instance(self):
        if self.main_window:
            self.main_window.new_instance()

    def open_file(self):
        if self.main_window:
            self.main_window.open_file()
