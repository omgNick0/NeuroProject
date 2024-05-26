# подключаем библиотеку компьютерного зрения
import cv2
# библиотека для вызова системных функций
import os
# для обучения нейросетей
import numpy as np
# встроенная библиотека для работы с изображениями
from PIL import Image
import runpy
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QWidget
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        self.flag = False
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 352)
        Dialog.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";")
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.apply = QtWidgets.QPushButton(parent=Dialog)
        self.apply.setObjectName("apply")
        self.verticalLayout.addWidget(self.apply)
        self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.apply.clicked.connect(self.fill_file)

        self.name = QtWidgets.QLabel(parent=Dialog)
        self.name.setObjectName("name")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.name)

        self.surname = QtWidgets.QLabel(parent=Dialog)
        self.surname.setObjectName("surname")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.surname)

        self.group = QtWidgets.QLabel(parent=Dialog)
        self.group.setObjectName("group")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.group)

        self.name_line = QtWidgets.QLineEdit(parent=Dialog)
        self.name_line.setObjectName("name_line")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.name_line)

        self.surname_line = QtWidgets.QLineEdit(parent=Dialog)
        self.surname_line.setObjectName("surname_line")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.surname_line)

        self.group_line = QtWidgets.QLineEdit(parent=Dialog)
        self.group_line.setObjectName("group_line")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.group_line)
        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.apply.setText(_translate("Dialog", "Ok"))
        self.name.setText(_translate("Dialog", "Имя: "))
        self.surname.setText(_translate("Dialog", "Фамилия: "))
        self.group.setText(_translate("Dialog", "Группа: "))
    def fill_file(self):
        name = str(abs(hash(self.group_line.text())+ hash(self.name_line.text()) + hash(self.surname_line.text()))%1000)
        with open("Students.txt", "a") as file:
            file.write(name + ',')
            file.write(self.group_line.text() + ',')
            file.write(self.name_line.text() + ' ')
            file.write(self.surname_line.text() + '\n')
        # получаем путь к этому скрипту
        path = os.path.dirname(os.path.abspath(__file__))
        # указываем, что мы будем искать лица по примитивам Хаара
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # счётчик изображений
        i = 0
        # расстояния от распознанного лица до рамки
        offset = 50
        '''
        with open("Students.txt", "r") as file:
            for line in file:
                print(line.split(':'))
                key = line.split(':')[0].replace('\n', '')
                value = line.split(':')[1].replace('\n', '')
                print(key, value)
        '''
        # получаем доступ к камере
        video = cv2.VideoCapture(0)
        # запускаем цикл
        while True:
            # берём видеопоток
            ret, im = video.read()
            # переводим всё в ч/б для простоты
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # настраиваем параметры распознавания и получаем лицо с камеры
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
            # обрабатываем лица
            if (len(faces) > 0):
                for (x, y, w, h) in faces:
                    # увеличиваем счётчик кадров
                    i = i + 1
                    # записываем файл на диск
                    cv2.imwrite("dataSet/face-" + name + '.' + str(i) + ".jpg",
                                gray[y - offset:y + h + offset, x - offset:x + w + offset])
                    # формируем размеры окна для вывода лица
                    cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
                    # показываем очередной кадр, который мы запомнили
                    cv2.imshow('im', im[y - offset:y + h + offset, x - offset:x + w + offset])
                    # делаем паузу
                    cv2.waitKey(100)
            # если у нас хватает кадров
            if i >= 50:  # 50 фото
                # освобождаем камеру
                video.release()
                # удаляем все созданные окна
                cv2.destroyAllWindows()
                # останавливаем цикл
                break

        # создаём новый распознаватель лиц
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # указываем, что мы будем искать лица по примитивам Хаара
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # путь к датасету с фотографиями пользователей
        dataPath = path + r'/dataSet'
        # получаем путь к картинкам
        image_paths = [os.path.join(dataPath, f) for f in os.listdir(dataPath)]
        # списки картинок и подписей на старте пустые
        images = []
        labels = []
        # перебираем все картинки в датасете
        for image_path in image_paths:
            # читаем картинку и сразу переводим в ч/б
            image_pil = Image.open(image_path).convert('L')
            # переводим картинку в numpy-массив
            image = np.array(image_pil, 'uint8')
            # получаем id пользователя из имени файла
            nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
            # определяем лицо на картинке
            faces = faceCascade.detectMultiScale(image)
            # если лицо найдено
            for (x, y, w, h) in faces:
                # добавляем его к списку картинок
                images.append(image[y: y + h, x: x + w])
                # добавляем id пользователя в список подписей
                labels.append(nbr)
                # выводим текущую картинку на экран
                cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
                # делаем паузу
                # cv2.waitKey(100)
        # обучаем модель распознавания на наших картинках и учим сопоставлять её лица и подписи к ним
        recognizer.train(images, np.array(labels))
        # сохраняем модель
        recognizer.save(path + r'/trainer/trainer.yml')
        # удаляем из памяти все созданные окна
        cv2.destroyAllWindows()

class Ui_List(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")

        self.listView = QtWidgets.QListWidget(parent=Dialog)
        self.listView.setObjectName("listView")
        self.gridLayout.addWidget(self.listView, 0, 0, 1, 1)
        self.read_file()
        arr = self.read_file()
        self.listView.addItems(arr)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))

    def read_file(self):
        arr = []
        with open("Students.txt", "r") as file:
            for line in file:
                arr.append(line.replace('\n', '').split(',')[1] + '        ' + line.replace('\n', '').split(',')[2])
                print(arr)
        return arr

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(308, 219)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setStyleSheet("font: 25 16pt \"Cascadia Code Light\"")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        self.scanning = QtWidgets.QPushButton(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scanning.sizePolicy().hasHeightForWidth())
        self.scanning.setSizePolicy(sizePolicy)
        self.scanning.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(170, 0, 0);")
        self.scanning.setObjectName("scanning")
        self.scanning.clicked.connect(self.scanning_face)

        self.verticalLayout.addWidget(self.scanning)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        self.add_stud = QtWidgets.QPushButton(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.add_stud.sizePolicy().hasHeightForWidth())
        self.add_stud.setSizePolicy(sizePolicy)
        self.add_stud.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(0, 0, 127);")
        self.add_stud.setObjectName("add_stud")
        self.add_stud.clicked.connect(self.enter_stud_info)


        self.verticalLayout.addWidget(self.add_stud)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem1)

        self.show_list = QtWidgets.QPushButton(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_list.sizePolicy().hasHeightForWidth())
        self.show_list.setSizePolicy(sizePolicy)
        self.show_list.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(170, 0, 0);")
        self.show_list.setObjectName("show_list")
        self.show_list.clicked.connect(self.show_stud_list)

        self.verticalLayout.addWidget(self.show_list)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.scanning.clicked.connect(self.scanning.setFocus) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.scanning.setText(_translate("MainWindow", "Режим сканирования"))
        self.add_stud.setText(_translate("MainWindow", "Добавить студента"))
        self.show_list.setText(_translate("MainWindow", "Показать список студентов"))


    def scanning_face(self):
        # получаем путь к этому скрипту
        path = os.path.dirname(os.path.abspath(__file__))
        # создаём новый распознаватель лиц
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # добавляем в него модель, которую мы обучили на прошлых этапах
        recognizer.read(path + r'/trainer/trainer.yml')
        # указываем, что мы будем искать лица по примитивам Хаара
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # получаем доступ к камере
        cam = cv2.VideoCapture(0)
        # настраиваем шрифт для вывода подписей
        font = cv2.FONT_HERSHEY_SIMPLEX

        """TODO: How many FPS ???"""

        # запускаем цикл
        while cv2.waitKey(1) < 0:
            # получаем видеопоток
            ret, im = cam.read()
            # переводим его в ч/б
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # определяем лица на видео
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)

            if (len(faces) == 0):
                cv2.imshow('Face recognition', im)
            else:
                # перебираем все найденные лица
                for (x, y, w, h) in faces:
                    # получаем id пользователя
                    nbr_predicted, coord = recognizer.predict(gray[y:y + h, x:x + w])
                    # рисуем прямоугольник вокруг лица
                    cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
                    # если мы знаем id пользователя
                    if nbr_predicted == 1:
                        # подставляем вместо него имя человека
                        nbr_predicted = 'serg'

                    if nbr_predicted == 2:
                        nbr_predicted = 'Kirill'

                    # добавляем текст к рамке
                    cv2.putText(im, str(nbr_predicted), (x, y + h), font, 1.1, (0, 255, 0))
                    # выводим окно с изображением с камеры
                    cv2.imshow('Face recognition', im)
                    # делаем паузу
                    # cv2.waitKey(10)
        # освобождаем камеру
        cam.release()
        # удаляем все созданные окна
        cv2.destroyAllWindows()

    def enter_stud_info(self):
        mess = QtWidgets.QDialog()
        ui = Ui_Dialog()
        ui.setupUi(mess)
        mess.show()
        #mess.enterEvent(self.face_train_gen())
        mess.exec()
        #mess.customEvent(self.face_train_gen())

    def show_stud_list(self):
        list = QtWidgets.QDialog()
        ui = Ui_List()
        ui.setupUi(list)
        list.show()
        list.exec()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
