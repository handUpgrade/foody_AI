import tensorflow
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

file_name_freq = 0
file_name_freq += 1
#data 불러오기
data_gen = ImageDataGenerator(rescale = 1./255, 
                              rotation_range=90,
                              brightness_range=[0.2, 1.0],
                              width_shift_range=0.3,
                              height_shift_range=0.3,
                              shear_range=0.5,
                              zoom_range=[0.8, 1.3],
                              horizontal_flip=True,
                              vertical_flip= True,
                              fill_mode='nearest')


img = load_img('./dataset/train/tuna(dongwon)/new/f.jpg')
x = img_to_array(img)
x = x.reshape((1,)+ x.shape)

i = 0 
# save_to_dir = "./dataset/train/potato".split('/')[0] + "/" + './dataset/train/potato'.split("/")[1]
save_to_dir ="./dataset/train/tuna(dongwon)/new/"
print(save_to_dir)

if not save_to_dir == "./dataset/train/":
    # for batch in data_gen.flow(x, batch_size=1, save_to_dir=save_to_dir, save_prefix='plus_'+str(file_name_freq),save_format='jpg'):
    for batch in data_gen.flow(x, batch_size=1, save_to_dir=save_to_dir, save_prefix='plus_'+str(file_name_freq),save_format='jpg'):    
        i += 1
        if i > 10:
            break