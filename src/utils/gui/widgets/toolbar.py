from PyQt6.QtWidgets import QWidget, QPushButton, QComboBox, QLabel, QHBoxLayout

class ToolbarWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        layout = QHBoxLayout()

        # Botón "New"
        self.newButton = QPushButton("New")
        self.newButton.clicked.connect(parent.new_instance)
        layout.addWidget(self.newButton)

        # Botón "Open"
        self.openButton = QPushButton("Open")
        self.openButton.clicked.connect(parent.open_file)
        layout.addWidget(self.openButton)

        # Selector de modelo
        self.loadModelLbl = QLabel("Load Model: ")
        layout.addWidget(self.loadModelLbl)

        self.loadModelButton = QComboBox()
        self.loadModelButton.addItems(["UNet", "SCUNet", "Dino Sampler"])
        layout.addWidget(self.loadModelButton)

        # Ajustar diseño
        layout.addStretch(1)  # Empuja los elementos a la izquierda
        self.setLayout(layout)
