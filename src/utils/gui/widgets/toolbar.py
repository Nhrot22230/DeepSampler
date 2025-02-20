from PyQt6.QtCore import QObject
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (QApplication, QMenu, QStyleFactory, QToolBar,
                             QToolButton,QFileDialog, QMessageBox)
import torch

class Toolbar(QObject):
    def __init__(self, parent, selected_model):
        super().__init__(parent)
        self.parent = parent
        self.originalPalette = QApplication.palette()
        self.selected_model = selected_model

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.parent.addToolBar(toolbar)

        new_action = QAction("New", self.parent)
        new_action.triggered.connect(self.parent.new_instance)
        toolbar.addAction(new_action)

        open_action = QAction("Open", self.parent)
        open_action.triggered.connect(self.parent.open_file)
        toolbar.addAction(open_action)

        self.create_model_menu(toolbar)
        toolbar.addSeparator()
        self.create_theme_menu(toolbar)

        return toolbar

    def create_theme_menu(self, toolbar):
        theme_menu = QMenu("Theme", self.parent)
        styles = QStyleFactory.keys()
        for style in styles:
            action = QAction(style, self.parent)
            action.triggered.connect(lambda checked, s=style: self.change_style(s))
            theme_menu.addAction(action)

        # checkbox de theme
        self.use_palette_action = QAction(
            "Use Theme's Standard Palette", self.parent, checkable=True
        )
        self.use_palette_action.setChecked(False)
        self.use_palette_action.triggered.connect(self.change_palette)
        theme_menu.addAction(self.use_palette_action)

        # boton de theme
        theme_button = QToolButton(self.parent)
        theme_button.setText("Theme")
        theme_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        theme_button.setMenu(theme_menu)
        toolbar.addWidget(theme_button)

    def create_model_menu(self, toolbar):
        model_menu = QMenu("Load Model", self.parent)
        self.modelGroup = QActionGroup(self.parent)
        self.modelGroup.setExclusive(True)

        # Lista de nombres de modelos
        for model_name in ["UNet", "SCUNet", "DinoSampler"]:
            action = QAction(model_name, self.parent)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, name=model_name: self.select_model(name))
            self.modelGroup.addAction(action)
            model_menu.addAction(action)
            if self.selected_model == model_name:
                action.setChecked(True)

        model_button = QToolButton(self.parent)
        model_button.setText("Load Model")
        model_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        model_button.setMenu(model_menu)
        toolbar.addWidget(model_button)



    def select_unet(self):
        self.selected_model = "UNet"
        print("Selected model: UNet")

    def select_scunet(self):
        self.selected_model = "SCUNet"
        print("Selected model: SCUNet")

    def select_dino_sampler(self):
        self.selected_model = "DinoSampler"
        print("Selected model: DinoSampler")

    def select_model(self, model_name):

        # Actualiza la selección del modelo
        self.selected_model = model_name
        self.parent.selected_model = model_name
        print(f"Selected model: {model_name}")

        # Abre el diálogo para seleccionar el archivo de pesos (.pth)
        file_name, _ = QFileDialog.getOpenFileName(
            self.parent,
            f"Select weights file for {model_name}",
            "",
            "PyTorch Files (*.pth)"
        )
        if file_name:
            try:
                # Cargar el state_dict (diccionario de parámetros) desde el archivo
                state_dict = torch.load(file_name, map_location=torch.device("cpu"))
                # Actualizar los parámetros del modelo correspondiente en el diccionario.
                self.parent.models[model_name].load_state_dict(state_dict)
                QMessageBox.information(
                    self.parent, "Success",
                    f"Weights loaded successfully for {model_name}."
                )
            except Exception as e:
                QMessageBox.critical(
                    self.parent, "Error",
                    f"Weights loaded successfully for {model_name}."
                )


    def change_style(self, styleName):
        QApplication.setStyle(styleName)

    def change_palette(self):
        if self.use_palette_action.isChecked():
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)
