import os
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QDateTime, Qt, QTimer
from PyQt6.QtGui import QIcon
from SecondWindow import SecondWindow
from widgets.drag_and_drop import DragDropWidget
from widgets.toolbar import ToolbarWidget

class MainApp(QDialog):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())
        styleComboBox.setCurrentIndex(QStyleFactory.keys().index('Fusion'))

        styleLabel = QLabel("&Theme:")

        styleLabel.setBuddy(styleComboBox)

        self.useStylePaletteCheckBox = QCheckBox("&Use theme's standard palette")
        self.useStylePaletteCheckBox.setChecked(True)

        self.createTopGroupBox()
        self.createDragDropArea()

        styleComboBox.textActivated.connect(self.changeStyle)
        self.useStylePaletteCheckBox.toggled.connect(self.changePalette)

        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)
        topLayout.addWidget(self.useStylePaletteCheckBox)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.topGroupBox, 1, 0)
        mainLayout.addWidget(self.dragDropWidget, 3, 0, 1, 2)
        self.setLayout(mainLayout)

        self.setWindowTitle("DinoSampler")
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        logo_path = os.path.join(scriptDir, "assets", "dinosampler_logo.png")
        self.setWindowIcon(QIcon(logo_path))
        self.changeStyle('Fusion')


    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)

    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) // 100)

    def createTopGroupBox(self):
        self.topGroupBox = QGroupBox()

        newButton = QPushButton("New")
        openButton = QPushButton("Open")

        loadModelLbl = QLabel("Load Model: ")
        loadModelButton = QComboBox()
        loadModelButton.addItem("UNet")
        loadModelButton.addItem("SCUNet")
        loadModelButton.addItem("Dino Sampler")

        newButton.clicked.connect(self.new_instance)
        openButton.clicked.connect(self.open_file)

        loadModelLbl.setBuddy(loadModelButton)

        layout = QHBoxLayout()
        layout.addWidget(newButton)
        layout.addWidget(openButton)
        layout.addStretch(1)
        layout.addWidget(loadModelLbl)
        layout.addWidget(loadModelButton)
        layout.addStretch(1)
        self.topGroupBox.setLayout(layout)

    def createDragDropArea(self):
        self.dragDropWidget = DragDropWidget(self)

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)

        timer = QTimer(self)
        timer.timeout.connect(self.advanceProgressBar)
        timer.start(1000)


    def new_instance(self):
        global new_app
        new_app = MainApp()
        new_app.show()
        self.close()

    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", filter="Audio Files (*.wav *.mp4)")
        if(file):
            self.go_second_window(file)

    def go_second_window(self, file):
        global new_app
        new_app = SecondWindow(file)
        new_app.show()
        self.close()
