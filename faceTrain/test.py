import runpy
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QWidget



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        self.setStyleSheet("background-color: grey")

        self.scanning_button = QPushButton("start scanning")
        self.scanning_button.clicked.connect(self.scanning)

        self.face_train_gen_button = QPushButton("add student")
        self.face_train_gen_button.clicked.connect(self.face_train_gen)

        self.setFixedSize(800, 600)
        self.scanning_button.setStyleSheet("color: green;")

        # Устанавливаем центральный виджет Window.
        #self.setCentralWidget(self.scanning_button)
        #self.setCentralWidget(self.face_train_button)

        layout = QVBoxLayout()
        layout.addWidget(self.scanning_button)
        layout.addWidget(self.face_train_gen_button)

        container = QWidget()
        container.setLayout(layout)

        # Устанавливаем центральный виджет Window.
        self.setCentralWidget(container)

    def scanning(self):
        runpy.run_module('main')
    def face_train_gen(self):
        runpy.run_module('face_train_gen')

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()