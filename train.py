import cv2
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
import pyautogui
import serial



#def train():
X = []
y = []

files_name = [f for f in listdir('img') if isfile(join('img', f))]
for name in files_name:
    try:
        # load the image
        img = cv2.imread(join('img', name),0)
        img = img[400:900,700:1650]
        # blur to remove details
        img = cv2.blur(img,(10,10))
        retval, img = cv2.threshold(img,170,255, cv2.THRESH_BINARY)
        # resize to improve performance
        img = cv2.resize(img, (25, 25))
        
        # convert to array
        image_as_array = np.ndarray.flatten(np.array(img))
        # add our image to the dataset
        X.append(image_as_array)
        # retrive the direction from the filename
        y.append(name.split('_')[0])
    except Exception as inst:
        print(name)
        print(inst)


# split for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


clf = MLPClassifier(solver='lbfgs', alpha=0.1, random_state=1 ,hidden_layer_sizes=50, max_iter=2000)
clf.fit(X_train, y_train)
print('score: ', clf.score(X_train, y_train))
print('score: ', clf.score(X_test, y_test))

from sklearn.model_selection import cross_val_predict
from sklearn import metrics

predicted =cross_val_predict(clf, X, y, cv=3, verbose=2, n_jobs=8)
print('CV: ', metrics.accuracy_score(y, predicted))
print(clf.predict(X_test[5]))
print(y_test[5])        
    


def infrence(clf,ser):
    # get image
    ser = serial.Serial("COM4", 9600)
    img = pyautogui.screenshot()

    # preprocess
    img = img[400:900,700:1650]
    # blur to remove details
    img = cv2.blur(img,(10,10))
    retval, img = cv2.threshold(img,170,255, cv2.THRESH_BINARY)
    # resize to improve performance
    img = cv2.resize(img, (25, 25))
    
    # convert to array
    image_as_array = np.ndarray.flatten(np.array(img))
    
    command = clf.predict(image_as_array)
    if command == 1: ser.write('1'.encode())
    elif command == 2: ser.write('2'.encode())
    else: ser.write('3'.encode())

def setupUi(MainWindow):

    MainWindow.setObjectName("MainWindow")
    MainWindow.resize(895, 422)
    centralwidget = QtWidgets.QWidget(MainWindow)
    centralwidget.setObjectName("centralwidget")
    url = 'http://192.168.86.26:8080'
    view = QtWebEngineWidgets.QWebEngineView(centralwidget)
    view.load(url)
    view.setGeometry(QtCore.QRect(450, 25, 1000, 1000))
    
    MainWindow.setCentralWidget(centralwidget)
    QtCore.QMetaObject.connectSlotsByName(MainWindow)


#if __name__ == "__main__":#
    # import sys
    # app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    # setupUi(MainWindow)
    # MainWindow.show()
    # sys.exit(app.exec())
   # clf = train()

