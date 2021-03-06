import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pickle

X = []
y = []

files_name = [f for f in listdir('image_bar') if isfile(join('image_bar', f))]
for name in files_name:
    try:
        # load the image
        img = cv2.imread(join('image_bar', name),0)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


model = MLPClassifier(solver='lbfgs', alpha=0.1, random_state=1 ,hidden_layer_sizes=15, max_iter=2000)
#train the model
history = model.fit(X_train,y_train)





from sklearn.metrics import accuracy_score


y_pred = history.predict(X_test)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
with open('model_pkl', 'wb') as files: pickle.dump(history, files)

