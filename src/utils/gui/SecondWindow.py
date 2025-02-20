import os

import librosa
import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np
import soundfile as sf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from widgets.toolbar import Toolbar


class SecondWindow(QMainWindow):
    def __init__(self, file_path, main_window, selected_model):
        super().__init__()
        self.main_window = main_window
        self.file_path = file_path
        self.separated_files = {}
        self.selected_model = selected_model
        self.players = {}
        self.audio_outputs = {}
        self.currently_playing = None
        self.track_name = os.path.basename(file_path)
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.thumbnail_path = os.path.join(scriptDir, "assets", "track.png")
        self.track_duration = ""
        self.play_icon = QIcon(os.path.join(scriptDir, "assets", "play_icon.png"))
        self.pause_icon = QIcon(os.path.join(scriptDir, "assets", "pause_icon.png"))
        self.reset_icon = QIcon(os.path.join(scriptDir, "assets", "reset_icon.png"))
        self.extract_metadata()
        self.process_audio()
        self.media_player_main = QMediaPlayer()
        main_track_src = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), self.file_path
        )
        self.media_player_main.setSource(QUrl.fromLocalFile(main_track_src))
        self.audio_output_main = QAudioOutput()
        self.audio_output_main.setVolume(0.5)
        self.media_player_main.setAudioOutput(self.audio_output_main)
        self.initUI()
        self.setWindowTitle("DinoSampler")
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

        self.thumbnail_label = QLabel()
        if os.path.exists(self.thumbnail_path):
            pixmap = QPixmap(self.thumbnail_path)
        else:
            print(f"Error: No se encontró el thumbnail en {self.thumbnail_path}")
            pixmap = QPixmap()  # Crea un pixmap vacío en caso de error

        self.thumbnail_label.setPixmap(pixmap.scaled(100, 100))
        infoLayout.addWidget(self.thumbnail_label)

        textLayout = QVBoxLayout()
        self.track_label = QLabel(self.track_name)
        self.duration_label = QLabel(f"Duration: {self.track_duration}")
        textLayout.addWidget(self.track_label)
        textLayout.addWidget(self.duration_label)
        textLayout.addStretch()

        main_controls_layout = QHBoxLayout()
        self.play_main_button = QPushButton()
        self.play_main_button.setIcon(self.play_icon)
        self.play_main_button.setFixedSize(32, 32)
        self.play_main_button.clicked.connect(lambda: self.toggle_play_pause_main())
        self.play_main_button.setEnabled(True)
        main_controls_layout.addWidget(self.play_main_button)

        self.reset_main_button = QPushButton()
        self.reset_main_button.setIcon(self.reset_icon)
        self.reset_main_button.setFixedSize(32, 32)
        self.reset_main_button.clicked.connect(lambda: self.reset_track_main())
        self.reset_main_button.setEnabled(True)
        main_controls_layout.addWidget(self.reset_main_button)

        controls_layout_v = QVBoxLayout()
        controls_layout_v.addStretch()
        controls_layout_v.addLayout(main_controls_layout)
        infoLayout.addLayout(textLayout)
        infoLayout.addStretch()
        infoLayout.addLayout(controls_layout_v)
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
        self.reset_buttons = []
        self.play_buttons = {}

        for label in self.track_labels:
            track_layout = QVBoxLayout()
            track_label = QLabel(label)
            track_layout.addWidget(track_label)

            track_sound_layout = QHBoxLayout()
            canvas = FigureCanvas(plt.figure(figsize=(6, 1)))
            canvas.figure.set_facecolor("black")
            download_btn = QPushButton()
            assets_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "assets"
            )
            download_btn.setIcon(QIcon(os.path.join(assets_path, "download_icon.png")))
            download_btn.setFixedSize(32, 32)
            download_btn.setEnabled(False)
            download_btn.clicked.connect(lambda _, lbl=label: self.download_file(lbl))

            play_btn = QPushButton()
            play_btn.setIcon(self.play_icon)
            play_btn.setFixedSize(32, 32)
            play_btn.setEnabled(False)
            play_btn.clicked.connect(lambda _, lbl=label: self.toggle_play_pause(lbl))

            reset_btn = QPushButton()
            reset_btn.setIcon(self.reset_icon)
            reset_btn.setFixedSize(32, 32)
            reset_btn.setEnabled(False)
            reset_btn.clicked.connect(lambda _, lbl=label: self.reset_track(lbl))

            track_sound_layout.addWidget(canvas)
            track_sound_layout.addWidget(play_btn)
            track_sound_layout.addWidget(reset_btn)
            track_sound_layout.addWidget(download_btn)

            track_layout.addLayout(track_sound_layout)

            layout.addLayout(track_layout)
            self.waveform_canvases.append(canvas)
            self.download_buttons.append(download_btn)
            self.reset_buttons.append(reset_btn)
            self.play_buttons[label] = play_btn

    def plot_waveform(self):
        y, sr = librosa.load(self.file_path, sr=44100)
        time = np.linspace(0, len(y) / sr, len(y))  # Eje de tiempo
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

        for i, (canvas, label, download_btn, play_btn, reset_btn) in enumerate(
            zip(
                self.waveform_canvases,
                self.track_labels,
                self.download_buttons,
                self.play_buttons,
                self.reset_buttons,
            )
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
            download_btn.setEnabled(True)
            reset_btn.setEnabled(True)
            self.play_buttons[label].setEnabled(True)

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

    def extract_metadata(self):
        temp_thumbnails_dir = os.path.join(os.path.dirname(__file__), "temp_thumbnails")
        os.makedirs(temp_thumbnails_dir, exist_ok=True)
        if self.file_path.endswith(".mp4"):
            video = mp.VideoFileClip(self.file_path)
            if video.duration:
                minutes = int(video.duration // 60)
                seconds = int(video.duration % 60)
                self.track_duration = f"{minutes}:{seconds:02d}"
                thumbnail_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "temp_thumbnails",
                    "thumbnail.png",
                )
                video.save_frame(thumbnail_path, t=1)
                self.thumbnail_path = thumbnail_path
        else:
            y, sr = librosa.load(self.file_path, sr=44100)
            duration = len(y) / sr
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            self.track_duration = f"{minutes}:{seconds:02d}"

    def toggle_play_pause(self, track_label):
        if track_label not in self.audio_outputs:
            self.audio_outputs[track_label] = QMediaPlayer()
            self.audio_outputs[track_label].setSource(
                QUrl.fromLocalFile(
                    os.path.join(
                        os.path.realpath(__file__), "temp_tracks", f"{track_label}.wav"
                    )
                )
            )

        if self.currently_playing and self.currently_playing != track_label:
            self.players[self.currently_playing].pause()
            self.play_buttons[self.currently_playing].setIcon(self.play_icon)

        if track_label not in self.players:
            audio_output = QAudioOutput()
            audio_output.setVolume(0.5)

            player = QMediaPlayer()
            player.setAudioOutput(audio_output)

            # Guardamos ambos objetos en la instancia de la clase para que persistan
            self.players[track_label] = player
            self.audio_outputs[track_label] = audio_output

            # Verificar si el archivo existe antes de cargarlo
            file_path = self.separated_files.get(track_label, "")
            if not os.path.exists(file_path):
                print(f"Error: No se encontró el archivo {file_path}")
                return

            player.setSource(QUrl.fromLocalFile(file_path))

        player = self.players[track_label]
        play_btn = self.play_buttons[track_label]
        self.currently_playing = track_label

        if player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            player.pause()
            play_btn.setIcon(self.play_icon)
        else:
            player.play()
            play_btn.setIcon(self.pause_icon)
            self.media_player_main.stop()
            self.play_main_button.setIcon(self.play_icon)

    def reset_track(self, track_label):
        if track_label in self.players:
            player = self.players[track_label]
            player.setPosition(0)

    def toggle_play_pause_main(self):
        if self.currently_playing:
            self.players[self.currently_playing].pause()
            self.play_buttons[self.currently_playing].setIcon(self.play_icon)
            self.currently_playing = None

        if (
            self.media_player_main.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        ):
            self.media_player_main.pause()
            self.play_main_button.setIcon(self.play_icon)
        else:
            self.media_player_main.play()
            self.play_main_button.setIcon(self.pause_icon)

    def reset_track_main(self):
        self.media_player_main.setPosition(0)
