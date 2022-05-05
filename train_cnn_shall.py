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
# from keras.layers import Input 
# from keras.models import Model 

# data = Input(shape=(2,3)) 

# model = keras.Sequential()
# input_layer = keras.layers.Conv2D(64,(3,3),activation='relu',input_shape= (25,25,1))
# model.add(input_layer)
# model.keras.layers.MaxPooling2D(2,2),
# hidden_layer_1 = keras.layers.Conv2D(64,(3,3),activation = 'relu'),
# model.add(hidden_layer_1)
# model.keras.layers.MaxPooling2D(2,2),
# model.keras.layers.Flatten(),
# linear_1 = keras.layers.Dense(units = 128,activation = 'relu')
# model.add(linear_1)
# linear_2 = keras.layers.Dense(3,activation='softmax')
# model.add(linear_2)

model= keras.Sequential([keras.layers.Conv2D(64,(3,3),activation='relu',input_shape= (25,25,1)),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Flatten(),
                          keras.layers.Dense(units = 128,activation = 'relu'),
                          keras.layers.Dense(3,activation='softmax')
                          ])


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.summary()

#train the model
history = model.fit(X_train,y_train,epochs = 10, batch_size = 5)
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

with open('model_pkl_2', 'wb') as files: pickle.dump(model, files)

layer_names = [layer.name for layer in model.layers]
layer_outputs = [layer.output for layer in model.layers]
feature_map_model = keras.models.Model(input=model.input, output=layer_outputs)

img = cv2.imread('forward_218.png',0)
img = img[400:900,700:1650]
# blur to remove details
img = cv2.blur(img,(10,10))
retval, img = cv2.threshold(img,170,255, cv2.THRESH_BINARY)
# resize to improve performance
img = cv2.resize(img, (25, 25))

# convert to array
image_as_array = np.array(img)
# add our image to the dataset
X1 = image_as_array
# retrive the direction from the filename
y1 = name.split('_')[0]

import matplotlib.pyplot as plt
plt.imshow(X1)
plt.show()

# layer_outputs = [layer.output for layer in model.layers[:4]]
# activation_model = keras.models.Model(inputs = model.input, outputs = layer_outputs)
# activations = activation_model.predict(X1)

# layer_names = []
  
# for layer in model.layers[:8]:
#   layer_names.append(layer.name)
# print(layer_names)

# layer_outputs = [layer.output for layer in model.layers]

# feature_map_model = tf.keras.models.Model(input=model.input, output=layer_outputs)
# feature_maps = feature_map_model.predict(X1)

# for layer_name, feature_map in zip(layer_names, feature_maps):print(f"The shape of the {layer_name} is =======>> {feature_map.shape}")

# # Steps to generate feature maps:-
# # We need to generate feature maps of only convolution layers and not dense layers and hence we
# #  will generate feature maps of layers that have “dimension=4″.

# for layer_name, feature_map in zip(layer_names, feature_maps): if len(feature_map.shape) == 4

# # Number of feature images/dimensions in a feature map of a layer 
   
# for layer_name, feature_map in zip(layer_names, feature_maps): if len(feature_map.shape) == 4
#     k = feature_map.shape[-1]  
#     #iterating over a feature map of a particular layer to separate all feature images.    
#     for i in range(k):
#         feature_image = feature_map[0, :, :, i]
# feature_image-= feature_image.mean()
# feature_image/= feature_image.std ()
# feature_image*=  64
# feature_image+= 128
# feature_image= np.clip(x, 0, 255).astype('uint8')

# for layer_name, feature_map in zip(layer_names, feature_maps):  
#     if len(feature_map.shape) == 4:
#       k = feature_map.shape[-1]  
#       size=feature_map.shape[1]
#       for i in range(k):
#         feature_image = feature_map[0, :, :, i]
#         feature_image-= feature_image.mean()
#         feature_image/= feature_image.std ()
#         feature_image*=  64
#         feature_image+= 128
#         feature_image= np.clip(x, 0, 255).astype('uint8')
#         image_belt[:, i * size : (i + 1) * size] = feature_image    

#     scale = 20. / k
#     plt.figure( figsize=(scale * k, scale) )
#     plt.title ( layer_name )
#     plt.grid  ( False )
#     plt.imshow( image_belt, aspect='auto')

#     import matplotlib.pyplot as plt
  
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# accuracy = history_dict['accuracy']
# val_accuracy = history_dict['val_accuracy']
  
# epochs = range(1, len(loss_values) + 1)
# fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# #
# # Plot the model accuracy vs Epochs
# #
# ax[0].plot(epochs, accuracy, 'bo', label='Training accuracy')
# ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
# ax[0].set_title('Training & Validation Accuracy', fontsize=16)
# ax[0].set_xlabel('Epochs', fontsize=16)
# ax[0].set_ylabel('Accuracy', fontsize=16)
# ax[0].legend()
# #
# # Plot the loss vs Epochs
# #
# ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
# ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
# ax[1].set_title('Training & Validation Loss', fontsize=16)
# ax[1].set_xlabel('Epochs', fontsize=16)
# ax[1].set_ylabel('Loss', fontsize=16)
# ax[1].legend()



# Initial layers are more interpretable and retain the majority of the features in the input image. 
# As the level of the layer increases, features become less interpretable, they become more abstract and they
#  identify features specific to the class leaving behind the general features of the image.