import os
from PyQt6.QtWidgets import QApplication
from MainApp import MainApp

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main = MainApp()
    main.show()
    app.exec()