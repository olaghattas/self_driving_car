import cv2
from cv2 import imread
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
from PySide6 import QtCore, QtWidgets, QtWebEngineWidgets
from PySide6.QtGui import *
from PySide6.QtCore import *
import pickle
import schedule
import time
# import sys

def infrence(img):
    # get image
    #ser = serial.Serial("COM4", 9600)
    #img = pyautogui.screenshot()

    # preprocess
    img = img[400:900,700:1650]
    # blur to remove details
    img = cv2.blur(img,(10,10))
    retval, img = cv2.threshold(img,170,255, cv2.THRESH_BINARY)
    # resize to improve performance
    img = cv2.resize(img, (25, 25))
    
    # convert to array
    image_as_array = np.ndarray.flatten(np.array(img))
    print("ima", image_as_array.shape) 
    with open('model_pkl' , 'rb') as f: lr = pickle.load(f)
    command = lr.predict([image_as_array])
    if command ==  'forward':print('forward') #ser.write('1'.encode())
    elif command == 'left':print('left') #ser.write('3'.encode())
    else:print('right') #ser.write('4'.encode())

# def setupUi(MainWindow):

#     MainWindow.setObjectName("MainWindow")
#     MainWindow.resize(895, 422)
#     centralwidget = QtWidgets.QWidget(MainWindow)
#     centralwidget.setObjectName("centralwidget")
#     url = 'http://192.168.86.26:8080'
#     view = QtWebEngineWidgets.QWebEngineView(centralwidget)
#     view.load(url)
#     view.setGeometry(QtCore.QRect(450, 25, 1000, 1000))
    
#     MainWindow.setCentralWidget(centralwidget)
#     QtCore.QMetaObject.connectSlotsByName(MainWindow)


# if __name__ == "__main__":

    # app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    # setupUi(MainWindow)
    # MainWindow.show()
    # sys.exit(app.exec())
  
        img = cv2.imread("./right_72.png",0)
        infrence(img)
    

        

