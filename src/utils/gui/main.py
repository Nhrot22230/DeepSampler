from MainApp import MainApp
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main = MainApp()
    main.show()
    app.exec()
