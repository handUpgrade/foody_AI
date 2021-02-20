import os
import sys
import io
import flask
from flask import redirect, url_for, request, render_template, Response, jsonify, redirect
from flask import Flask, render_template
from flask_restful import Api
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image


# app = flask.Flask(__name__)
app = Flask(__name__)
api = Api(app)

def get_model():
    global model
    model = load_model('food_final.h5')
    print("*model loaded!")



def prepare_image(image, target_size):

    cvt_image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
    im_pil = Image.fromarray(cvt_image)
    
    # resize the array (image) then PIL image
    im_resized = im_pil.resize((64, 64))

    img_array = img_to_array(im_resized)
    
    image_array_expanded = np.expand_dims(img_array, axis = 0)
    return image_array_expanded
    
    # image = image.resize(target_size, refcheck=False)
    # # image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)  
    # # image = img_to_array(image)
    
    # image = np.array(image)
    # image = np.expand_dims(image, axis=0)

    # return image

print('loading keras')
get_model()

@app.route("/test", methods=['POST'])
def predict():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format(이미지 읽기)
            # image = np.fromstring(flask.request.files["image"].read())
            # print(type(image))
            # plt.imshow(image)
            # plt.show()
            # print (image.format, image.size, image.mode)
             
            image = np.fromstring(flask.request.files["image"].read(),dtype=np.uint8)         
            # image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            # image = image[90:400, 10:500]

        
            cap = cv2.VideoCapture(0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
             
            # 내부가 파란색으로 채워진 사각형을 그립니다. 
            
            # cv2.rectangle(image,(10,90), (290, 410), (0, 0, 255), 3)
            image = image[90:410, 10:290]
            # cv2.imshow('after', image)
            
            # image = Image.open(io.BytesIO(image))
            # print (image.format, image.size, image.mode)
            # print(type(image))

            # image = Image.open(image)
        
            # preprocess the image and prepare it for classification
            #opencv
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = np.array(image)

            #이미지 전처리
            # image = cv2.resize(image, (64,64), interpolation = cv2.INTER_AREA)
            # image = image.reshape(-1, 100, 100, 1)
            prepared_image = prepare_image(image, target_size=(64, 64))
          
            
            # print(prepared_image)
            
            # classify the input image and then initialize the list
            # of predictions to return to the client

            #예측
            preds = model.predict(prepared_image).tolist()         
            results = np.argmax(preds, axis=1)
            data["prediction"]=[]
            
            # print(preds)
            for i in results:
                # print (type(pre_ans))
                # pre_ans_str = ''
                print(i)
                if i  == 0: 
                    pre_ans_str = "fire"
                    food_id = 1

                elif i == 1:
                    pre_ans_str = "potato"
                    food_id = 2
                    print(type(food_id))
                elif i == 2: 
                    pre_ans_str = "shin"

                else: pre_ans_str = "인식 불가"
                print(str(pre_ans_str))
            r={"label": pre_ans_str}
            data["prediction"].append(r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()       
        
        # return {'label':pre_ans_str}
        # return {'label': preds}
            

            # indicate that the request was a success
        # data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data) 



if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))

    app.run(debug=True)
