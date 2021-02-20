from over import *

classes = ["firechicken",
                "honeyButter",
                "jagabi",
                "marketO",
                "red(pomegranate)",
                "shin",
                "tuna(dongwon)",
                "worldcon",
                "nothing"]

nb_classes = len(classes)

image_w = 64
image_h = 64

X_train, X_test, Y_train, Y_test = np.load("./june_16_2.npy")
print("ok")
# print('Xtrain_shape', X_train.shape)
print(X_train.shape)
print(X_train.shape[0])
print(X_test.shape)

#일반화
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

#모델
model = Sequential()
model.add(Convolution2D(32, (3, 3), padding='same', activation='relu', 
                        input_shape=X_train.shape[1:])) 

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3),  activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  
model.add(Convolution2D(64, 3, 3,  activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Convolution2D(64, 3, 3, activation='relu'))#새로추가
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Convolution2D(64, 3, 3, activation='relu'))#새로추가1(sample)
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.4))
  
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes,activation = 'softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#0.9781 'binary_crossentropy' - 0.9692
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 0.9526 -0.9887 *-09869
model_dir = './model'

if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
# model_path = model_dir + '/june_batch_32.h5' # 제일 최근
model_path = model_dir + '/sample.h5'
# model_path = model_dir + '/graphic_compare.h5'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
model.summary()

# history = model.fit(X_train, Y_train, batch_size=32, epochs=50, validation_data=(X_test, Y_test), callbacks=[checkpoint, early_stopping])
history = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.4, callbacks=[checkpoint, early_stopping])
# history = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[checkpoint, early_stopping])
print("정확도 : %.4f" % (model.evaluate(X_test, Y_test)[1]))
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))



plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_oss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()