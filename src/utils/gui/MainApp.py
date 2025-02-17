import os
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QIcon
from SecondWindow import SecondWindow
from widgets.drag_and_drop import DragDropWidget
from widgets.toolbar import Toolbar

class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()

        self.setWindowTitle("DinoSampler")
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        logo_path = os.path.join(scriptDir, "assets", "dinosampler_logo.png")
        self.setWindowIcon(QIcon(logo_path))

        self.selected_model = None

        self.toolbar = Toolbar(self, self.selected_model)
        self.toolbar.create_toolbar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.dragDropWidget = DragDropWidget(self)
        layout.addWidget(self.dragDropWidget)

        self.toolbar.change_style('Fusion')

    def new_instance(self):
        global new_app
        new_app = MainApp()
        new_app.show()

    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", filter="Audio Files (*.wav *.mp4)")
        if file:
            self.go_second_window(file)

    def go_second_window(self, file):
        new_app = SecondWindow(file, self, self.selected_model)
        new_app.show()
        self.close()
