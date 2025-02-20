import os

from moviepy.editor import VideoFileClip
from mutagen.wave import WAVE
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QVBoxLayout, QWidget
from src.utils.gui.SecondWindow import SecondWindow
from src.models import deep_sampler, scunet, u_net
from src.utils.gui.widgets.drag_and_drop import DragDropWidget
from src.utils.gui.widgets.toolbar import Toolbar


class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.models = {
            "DeepSampler": deep_sampler.DeepSampler(),
            "SCUNet": scunet.SCUNet(),
            "UNet": u_net.SimpleUNet(),
        }
        self.setWindowTitle("DinoSampler")
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        logo_path = os.path.join(scriptDir, "assets", "dinosampler_logo.png")
        self.setWindowIcon(QIcon(logo_path))

        self.selected_model = "UNet"

        self.toolbar = Toolbar(self, self.selected_model)
        self.toolbar.create_toolbar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.dragDropWidget = DragDropWidget(self)
        layout.addWidget(self.dragDropWidget)

        self.toolbar.change_style("Fusion")

    def new_instance(self):
        global new_app
        new_app = MainApp()
        new_app.show()

    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select File", filter="Audio Files (*.wav *.mp4)"
        )
        if file:
            if not self.validate_duration(file):
                QMessageBox.warning(
                    self, "Invalid File", "El archivo supera los 5 minutos."
                )
                return
            self.go_second_window(file)

    def validate_duration(self, file_path):
        if file_path.lower().endswith(".mp4"):
            clip = VideoFileClip(file_path)
            duration = clip.duration
            clip.close()
        elif file_path.lower().endswith(".wav"):
            audio = WAVE(file_path)
            duration = audio.info.length
        else:
            return False
        return duration <= 300

    def go_second_window(self, file):
        selected_model_instance = self.models[self.selected_model]
        new_app = SecondWindow(file, self, selected_model_instance)
        new_app.show()
        self.close()
