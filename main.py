from PySide6 import QtCore, QtWidgets, QtWebEngineWidgets
from PySide6.QtGui import *
from PySide6.QtCore import *
import pyautogui
import serial

import json

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        ser = serial.Serial("COM4", 9600)
        self.ser = ser

        self.get_variables()

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(895, 422)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.forward = QtWidgets.QPushButton('forward', self.centralwidget)
        self.forward.setGeometry(QtCore.QRect(200, 50, 100, 40))
        self.forward.setObjectName("forward")
        self.forward.setCheckable(True)
        self.forward.clicked.connect(self.the_forward_button_was_clicked)

        #self.backward = QtWidgets.QPushButton('backward', self.centralwidget)
        #self.backward.setGeometry(QtCore.QRect(200,200, 100, 40))
        #self.backward.setObjectName("backward")
        #self.backward.setCheckable(True)
        #self.j = 1
        #self.backward.clicked.connect(self.the_backward_button_was_clicked)

        self.left = QtWidgets.QPushButton('left', self.centralwidget)
        self.left.setGeometry(QtCore.QRect(100, 125, 100, 40))
        self.left.setObjectName("left")
        self.left.setCheckable(True)
        self.left.clicked.connect(self.the_left_button_was_clicked)

        self.right = QtWidgets.QPushButton('right', self.centralwidget)
        self.right.setGeometry(QtCore.QRect(300, 125, 100, 40))
        self.right.setObjectName("right")
        self.right.setCheckable(True)
        self.right.clicked.connect(self.the_right_button_was_clicked)
        

        url = 'http://192.168.86.26:8080'
        self.view = QtWebEngineWidgets.QWebEngineView(self.centralwidget)
        self.view.load(url)
        self.view.setGeometry(QtCore.QRect(450, 25, 1000, 1000))

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def get_variables(self):
        file = open("variables.json")
        data = json.load(file)
        file.close()
        self.i = data['i']
        self.k = data['k']
        self.l = data['l']
    
    def update_variables(self):
        file = open("variables.json","w")
        updated_data = {"i": self.i, "k": self.k, "l":self.l}
        json.dump(updated_data, file)
        file.close()
        

    def save_image(self,image : str, count : int):
        im = pyautogui.screenshot()
        count = str(count)
        img_name= image + "_" + count +'.png'
        path = './images/'+ image + '/' + img_name
        im.save(path)

    def the_forward_button_was_clicked(self,file1):
        self.save_image("forward", self.i)
        self.i = self.i + 1
        self.update_variables()
        self.ser.write('1'.encode())


    def the_backward_button_was_clicked(self):
        self.save_image("backward", self.j)
        self.j = self.j + 1
        self.ser.write('2'.encode())

    def the_left_button_was_clicked(self,file2):
        self.save_image("left", self.k)
        self.k = self.k + 1
        self.update_variables()
        self.ser.write('3'.encode())

    def the_right_button_was_clicked(self,file3):
        self.save_image("right", self.l)
        self.l = self.l + 1
        self.update_variables()
        self.ser.write('4'.encode())

        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.exitbtn.setText(_translate("MainWindow", "Exit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())