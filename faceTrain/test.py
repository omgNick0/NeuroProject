import runpy
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QWidget



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        self.scanning_button = QPushButton("start scanning")
        self.scanning_button.clicked.connect(self.scanning)

        self.face_train_button = QPushButton("start training")
        self.face_train_button.clicked.connect(self.face_train)

        self.face_gen_button = QPushButton("generate dataset")
        self.face_gen_button.clicked.connect(self.face_gen)

        self.setFixedSize(800, 600)
        # Устанавливаем центральный виджет Window.
        #self.setCentralWidget(self.scanning_button)
        #self.setCentralWidget(self.face_train_button)

        layout = QVBoxLayout()
        layout.addWidget(self.scanning_button)
        layout.addWidget(self.face_train_button)
        layout.addWidget(self.face_gen_button)

        container = QWidget()
        container.setLayout(layout)

        # Устанавливаем центральный виджет Window.
        self.setCentralWidget(container)

    def scanning(self):
        runpy.run_module('main')
    def face_train(self):
        runpy.run_module('face_train')
    def face_gen(self):
        runpy.run_module('face_gen')

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()