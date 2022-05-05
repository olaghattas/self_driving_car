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

import seaborn as sns
np.random.seed(0)

X = []
y = []

folder_name = 'image_bar'
files_name = [f for f in listdir(folder_name) if isfile(join(folder_name , f))]
for name in files_name:
    try:
        # load the image
        img = cv2.imread(join(folder_name , name),0)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
# scale the data
X_train = X_train.reshape(X_train.shape[0],25,25,1)
X_train = X_train/255.0
X_val = X_val.reshape(X_train.shape[0],25,25,1)
X_val = X_val/255.0
X_test = X_test.reshape(X_test.shape[0],25,25,1)
X_test = X_test/255.0
print(X_train.shape)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.fit_transform(y_val)
y_test = le.fit_transform(y_test)



from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model


inputShape=(25,25,1)
input = Input(inputShape)

x = Conv2D(64,(3,3),strides = (1,1),name='layer_conv1')(input)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool1')(x)


x = Conv2D(64,(3,3),strides = (1,1),name='layer_conv2')(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool2')(x)

x = Conv2D(32,(3,3),strides = (1,1),name='conv3')(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool3')(x)

x = Flatten()(x)
x = Dense(128,activation = 'relu',name='fc0')(x)
x = Dropout(0.25)(x)
x = Dense(64,activation = 'relu',name='fc1')(x)
x = Dropout(0.25)(x)
x = Dense(3,activation = 'softmax',name='fc2')(x)

model = Model(inputs = input,outputs = x,name='Predict')

model.summary()


# compile the model
model.compile(loss='categorical_crossentropy',optimizer= 'adam', metrics=['accuracy'])
history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=10)

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("on valid data")
pred1=model.evaluate(X_val,y_val)
print("accuaracy", str(pred1[1]*100))
print("Total loss",str(pred1[0]*100))

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




from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_train[10].reshape(1,25,25,1))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

plt.imshow(X_train[10][:,:,0]);

display_activation(activations, 8, 8, 1)
display_activation(activations, 8, 8, 3)
display_activation(activations, 8, 8, 7)


with open('model_pkl_ola', 'wb') as files: pickle.dump(model, files)