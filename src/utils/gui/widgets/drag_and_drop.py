from moviepy.editor import VideoFileClip
from mutagen.wave import WAVE
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QFileDialog, QLabel, QMessageBox, QVBoxLayout, QWidget


class DragDropWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 100)
        self.setStyleSheet(
            "border: 2px dashed gray; border-radius: 10px; padding: 20px;"
        )

        self.label = QLabel("Drag and drop a file here\n(.mp4, .wav)", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(
                url.toLocalFile().lower().endswith((".wav", ".mp4")) for url in urls
            ):
                event.acceptProposedAction()
            else:
                self.label.setText("Invalid file type, only .wav and .mp4")
        else:
            self.label.setText("Invalid drop, only .wav and .mp4 files allowed")

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith((".wav", ".mp4")):
                if self.validate_duration(file_path):
                    if self.parent():
                        self.parent().parent().go_second_window(file_path)
                else:
                    self.label.setText("File exceeds 5 minutes limit.")
            else:
                self.label.setText("Invalid file, only .wav and .mp4 allowed")

    def validate_duration(self, file_path):
        try:
            if file_path.lower().endswith(".wav"):
                audio = WAVE(file_path)
                duration = audio.info.length
            elif file_path.lower().endswith(".mp4"):
                video = VideoFileClip(file_path)
                duration = video.duration
                video.close()
            else:
                return False

            return duration <= 300
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file: {e}")
            return False


class FileSelector(QWidget):
    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select File", filter="Audio/Video Files (*.wav *.mp4)"
        )
        if file:
            if self.validate_duration(file):
                self.go_second_window(file)
            else:
                QMessageBox.warning(
                    self, "Invalid File", "File exceeds 5 minutes limit."
                )

    def validate_duration(self, file_path):
        try:
            if file_path.lower().endswith(".wav"):
                audio = WAVE(file_path)
                duration = audio.info.length
            elif file_path.lower().endswith(".mp4"):
                video = VideoFileClip(file_path)
                duration = video.duration
                video.close()
            else:
                return False

            return duration <= 300
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file: {e}")
            return False
