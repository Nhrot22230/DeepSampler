from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

class DragDropWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 100)
        self.setStyleSheet("border: 2px dashed gray; border-radius: 10px; padding: 20px;")

        self.label = QLabel("Drag and drop a file here", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith((".wav", ".mp4")) for url in urls):
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
                if self.parent():
                    self.parent().parent().go_second_window(file_path)
            else:
                self.label.setText("Invalid file, only .wav and .mp4 allowed")