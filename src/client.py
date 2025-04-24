import sys
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QStackedWidget, QStackedLayout


class MainWindows(QMainWindow):
    def __init__(self):
        super().__init__()
        label = QLabel("Hello, World!")
        self.setWindowTitle("Hello World App")
        self.setGeometry(200, 200, 200, 50)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindows()
    window.show()
    sys.exit(app.exec())