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
    model = load_model('june_model_2.h5')
    print("*model loaded!")



def prepare_image(image, target_size):

    cvt_image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
    im_pil = Image.fromarray(cvt_image)
    
    im_resized = im_pil.resize((64, 64))

    img_array = img_to_array(im_resized)
    
    image_array_expanded = np.expand_dims(img_array, axis = 0)
    return image_array_expanded
    

print('loading keras')
get_model()

@app.route("/predictPhoto", methods=['POST'])
def predict():
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = np.fromstring(flask.request.files["image"].read(),dtype=np.uint8)         
            # image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (292, 512), interpolation=cv2.INTER_AREA)
            # image = image[90:400, 10:500]
            # 내부가 파란색으로 채워진 사각형을 그립니다. 
            
            cv2.rectangle(image,(10,90), (290, 410), (0, 0, 255), 3)
            # cv2.imshow('before', image)
            image = image[90:410, 10:290]
            # cv2.imshow('after', image)
           
            #이미지 전처리
          
            prepared_image = prepare_image(image, target_size=(64, 64))
          
            

            #예측
            preds = model.predict(prepared_image).tolist()         
            results = np.argmax(preds, axis=1)
            
            # print(preds)
            for i in results:
                # print (type(pre_ans))
                pre_ans_str = ''
                print(i)
                if i  == 0: 
                    pre_ans_str = 3
                    print("불닭볶음면2")

                elif i == 1:
                    pre_ans_str = 17
                    print("허니버터칩 프랑스 고메버터 맛")
                    
                elif i == 2: 
                    pre_ans_str = 1
                    print("자가비3")
                    
                elif i == 3:
                    pre_ans_str = 4
                    print("청정원 홍초 석류")

                elif i == 4: 
                    pre_ans_str = 5
                    print("브라우니")

                elif i == 5:
                    pre_ans_str = 2
                    print("신라면")
                    
                elif i == 6: 
                    pre_ans_str = "동원참치"
                    # food_id = 7

                elif i == 7: 
                    pre_ans_str = "월드콘"
                     
                # elif i == 8: 
                #     pre_ans_str = "인식 불가"
                #     food_id = 9
                else: 
                    pre_ans_str = 0
                    print("인식불가")
                      
                
                print(str(pre_ans_str))
            r={"label": pre_ans_str}
            
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()       
        
        # return {'label':pre_ans_str}
        # return {'label': preds}
            

         
        # data["success"] = True

    return flask.jsonify(r) 



if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))

    app.run(debug=True)
