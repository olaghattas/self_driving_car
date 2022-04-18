import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from PIL import Image
import pickle
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
np.random.seed(0)

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
        image_as_array = np.array(img)
        # add our image to the dataset
        X.append(image_as_array)
        # retrive the direction from the filename
        y.append(name.split('_')[0])
    except Exception as inst:
        print(name)
        print(inst)

X = np.asarray(X)
y = np.asarray(y)



# split for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# scale the data
X_train = X_train.reshape(X_train.shape[0],25,25,1)
X_train = X_train/255.0
X_test = X_test.reshape(X_test.shape[0],25,25,1)
X_test = X_test/255.0
print(X_train.shape)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

import tensorflow as tf
import numpy as np
from tensorflow import keras
model = keras.Sequential([keras.layers.Conv2D(64,(3,3),activation='relu',input_shape= (25,25,1)),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Flatten(),
                          keras.layers.Dense(units = 128,activation = 'relu'),
                          keras.layers.Dropout(0.25),
                          keras.layers.Dense(units = 225,activation = 'relu'),
                          keras.layers.Dropout(0.25),
                          keras.layers.Dense(3,activation='softmax')
                          ])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.summary()

#train the model
model.fit(X_train,y_train,epochs = 10)
print(model.predict(X_test)[8])
test_loss, test_acc = model.evaluate(X_test, y_test)

print('test loss: {}, test accuracy: {}'.format(test_loss, test_acc) )

#evaluate one example
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
random_idx = np.random.choice(len(X_test))
x_sample = X_test[random_idx]
y_sample_true = y_test[random_idx]
y_sample_pred_class = y_pred_classes[random_idx]

plt.title('predicted:{}, True: {}'.format(y_sample_pred_class, y_sample_true), fontsize=16)
plt.imshow(x_sample.reshape(25,25), cmap='gray')
print(random_idx)

confusion_mtx = confusion_matrix(y_test, y_pred_classes)

#plot
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(confusion_mtx, annot = True, fmt = 'd', ax=ax,cmap = 'Blues')
ax.set_xlabel('predicted label')
ax.set_ylabel('true label')
ax.set_title('confusion matrix')

with open('model_pkl', 'wb') as files: pickle.dump(model, files)
