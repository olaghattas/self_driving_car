from flask import Flask, request, jsonify
from cv2 import imread
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pickle
import cv2
app = Flask(__name__)


@app.route('/getmsg/', methods=['GET'])
def respond():
    # Retrieve the name from the url parameter /getmsg/?name=
    name = request.args.get("name", None)

    # For debugging
    print(f"Received: {name}")

    response = {}

    # Check if the user sent a name at all
    if not name:
        response["ERROR"] = "No name found. Please send a name."
    # Check if the user entered a number
    elif str(name).isdigit():
        response["ERROR"] = "The name can't be numeric. Please send a string."
    else:
        response["MESSAGE"] = f"Welcome {name} to our awesome API!"

    # Return the response in json format
    return jsonify(response)


@app.route('/predict/', methods=['POST'])
def predict():
    param = request.form.get('img')
    print(param)

    # You can add the test cases you made in the previous function, but in our case here you are just testing the POST functionality
    if param:
        # preprocess
        img = param[400:900,700:1650]
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
        commands = {"foward":'1','left':'3', 'right': '4'}
        return jsonify({
            'command': commands[command]
        })

    else:
        return jsonify({
            "ERROR": "No name found. Please send a name."
        })


@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to our medium-greeting-api!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)