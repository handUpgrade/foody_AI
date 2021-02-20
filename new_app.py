import flask
import numpy as np
from scipy import misc
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import io
import os
import sys
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


app = Flask(__name__)

img_path = 'shin.jpg'
img = image.load_img(img_path, target_size = (64,64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
@app.route('/')
def hello():
    return "hello"


@app.route('/predict', methods=["POST"])
def make_prediction():
    if request.method =="POST":
        file = request.files[x]
        if not file: 
            return "image가 없음"

        img = misc.imread(file)
        img = img[:, :, :3]
        img = img.reshape(1, -1)

        prediction = model.predict(img)
        label = str(np.squeeze(prediction))

        if label =='shin':
            print("shin이 나옴")

        return {'image': label}

if __name__== '__main__':
    model = load_model('food_final.h5')
    app.run(debug=True)