from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

caltech_dir = "./dataset/test/testing/*.*"
# caltech_dir = 'http://image.nongshim.com/non/pro/03_product.jpg'
image_w = 64
image_h = 64

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir)
# print(files)
# files = glob.glob(caltech_dir)
# print(caltech_dir)
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    # print(img)
    data = np.asarray(img)
    # print(data)
    filenames.append(f)
    X.append(data)
    # print(f)
X = np.array(X)
# print("X = " + str(X))
# model = load_model('./model/multi_img_classification.model')
model = load_model('./model/june_batch_16.h5')
prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0

#이 비교는 그냥 파일들이 있으면 해당 파일과 비교. 카테고리와 함께 비교해서 진행하는 것은 _4 파일.
for i in prediction:
    # pre_ans = i.argmax()  # 예측 레이블
    pre_ans = i.argmax()
    print(i)
    print(pre_ans)
    pre_ans_str = ''
    # if pre_ans == 0: pre_ans_str = "불닭볶음면"
    # elif pre_ans == 1: pre_ans_str = "자가비"
    # elif pre_ans == 2: pre_ans_str = "신라면"
    if pre_ans == 0: pre_ans_str = "firechicken"
    elif pre_ans == 1: pre_ans_str = "honeyButter"
    elif pre_ans == 2: pre_ans_str = "jagabi"
    elif pre_ans == 3: pre_ans_str = "marketO"
    elif pre_ans == 4: pre_ans_str = "red(pomegranate)"
    elif pre_ans == 5: pre_ans_str = "shin"
    elif pre_ans == 6: pre_ans_str = "tuna(dongwon)"
    elif pre_ans == 7: pre_ans_str = "worldcon"
    elif pre_ans == 8: pre_ans_str = "nothing"
    # else: pre_ans_str = "nothing"
    if i[0] >= 0.8: print(filenames[cnt].split("\\")[1]+" = "+pre_ans_str)
    if i[1] >= 0.8: print(filenames[cnt].split("\\")[1]+" = "+pre_ans_str)
    if i[2] >= 0.8: print(filenames[cnt].split("\\")[1]+" = "+pre_ans_str)
    if i[3] >= 0.8: print(filenames[cnt].split("\\")[1]+" = "+pre_ans_str)
    if i[4] >= 0.8: print(filenames[cnt].split("\\")[1]+" = "+pre_ans_str)
    if i[5] >= 0.8: print(filenames[cnt].split("\\")[1]+" = "+pre_ans_str)
    if i[6] >= 0.8: print(filenames[cnt].split("\\")[1]+" = "+pre_ans_str)
    if i[7] >= 0.8: print(filenames[cnt].split("\\")[1]+" = "+pre_ans_str)
    if i[8] >= 0.001: print(filenames[cnt].split("\\")[1]+" = "+pre_ans_str)

      
    # else: print(pre_ans_str)
    cnt += 1