import numpy as np
import matplotlib.pyplot as plt
    
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import time
import random

def plot_loss_curve(history):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()   

def train_model(X_train,y_train, X_test, y_test,learning_rate, kernel_size, filter_size,dense_size, strides):

    model = Sequential([
                Input(shape=(300,300,3), name='input_layer'),

                Conv2D(filter_size,  kernel_size=kernel_size, strides=strides, activation='relu', name='conv_layer1'),
                BatchNormalization(),
                MaxPooling2D(pool_size=2),
                
                # Conv2D(filter_size,  kernel_size=kernel_size, strides=strides, activation='relu', name='conv_layer2'),
                # MaxPooling2D(pool_size=2),

                # Conv2D(filter_size*2,  kernel_size=kernel_size, strides=strides, activation='relu', name='conv_layer3'),
                # MaxPooling2D(pool_size=2),

                # Conv2D(16,  kernel_size=3, strides = 2, activation='relu', name='conv_layer2'),
                # MaxPooling2D(pool_size=2),
                # Dropout(0.5),   

                # Conv2D(16,  kernel_size=3, activation='relu', name='conv_layer2'),
                # MaxPooling2D(pool_size=2),

                Flatten(),
                Dense(dense_size, activation='relu', name='hidden_layer'),
                Dropout(0.3),   

                Dense(3, activation='softmax', name='output_layer')
            ])

    #learning rate 변경해보기
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) 
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=5)
    # plot_loss_curve(history.history)

    # print("train loss=", history.history['loss'][-1])
    # print("validation loss=", history.history['val_loss'][-1])    
    
    model.save('model-%s-%s-%s-%s-%s.model'%(learning_rate, kernel_size, filter_size,dense_size, strides))
    
    return model


def train_model2(X_train,y_train, X_test, y_test):
    model = Sequential([
                Input(shape=(300,300,3), name='input_layer'),

                Conv2D(128,  kernel_size=3, activation='relu', name='conv_layer1'),
                MaxPooling2D(pool_size=2),
                Dropout(0.5),   
                # Conv2D(16,  kernel_size=3, strides = 2, activation='relu', name='conv_layer2'),
                # MaxPooling2D(pool_size=2),
                # Dropout(0.5),   

                # Conv2D(16,  kernel_size=3, activation='relu', name='conv_layer2'),
                # MaxPooling2D(pool_size=2),

                Flatten(),
                # Dense(16, activation='relu', name='hidden_layer'),
                Dense(3, activation='softmax', name='output_layer')
            ])

    #learning rate 변경해보기
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=5, epochs=5)
    # plot_loss_curve(history.history)

    # print("train loss=", history.history['loss'][-1])
    # print("validation loss=", history.history['val_loss'][-1])    
    
    model.save('model-2.model')

    
    return model

def predict_image_sample(model, X_test, y_test, test_id=-1):
    if test_id < 0:
        from random import randrange
        test_sample_id = randrange(X_test.shape[1])
    else:
        test_sample_id = test_id
        
    test_image = X_test[test_sample_id]
    
    plt.imshow(test_image)
    
    test_image = test_image.reshape(1,300,300,3)

    y_actual = np.argmax(y_test[test_sample_id])
    
    print("y_actual number=", y_actual)
    
    y_pred = model.predict(test_image)
    print("y_pred=", y_pred)
    y_pred = np.argmax(y_pred, axis=1)[0]
    print("y_pred number=", y_pred)
    
    if y_pred != y_actual:
        print("sample %d is wrong!" %test_sample_id)
        with open("wrong_samples.txt", "a") as errfile:
            print("%d"%test_sample_id, file=errfile)
    else:
        print("sample %d is correct!" %test_sample_id)

if __name__ == '__main__':
    start = time.time()

    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    np.save('X_train_norm',X_train)
    np.save('X_test_norm',X_test)


    # # #test용 train data = 10000개, test data = 5000개
    # X_train = np.load('X_train_temp.npy')
    # X_test = np.load('X_test_temp.npy')[:3000]
    # y_train = np.load('y_train_temp.npy')
    # y_test = np.load('y_test_temp.npy')[:3000]

    # print(X_train.shape)
    # print(y_train.shape)
    # print(y_train)

    # C = list(zip(X_train,y_train))
    # random.shuffle(C)
    # X_train,y_train = zip(*C)
    # X_train = np.array(list(X_train))
    # y_train = np.array(list(y_train))
    # print(X_train.shape)
    # print(y_train.shape)
    # print(y_train)

    # print("load shuffle time:", (time.time()-start))
    # learning_rate = 0.00001
    # kernel_size = [3]
    # filter_size = [32,64,128]
    # dense_size = [1]
    # strides = 1
    
    # for j in kernel_size:
    #     for k in filter_size:
    #         for d in dense_size:
    #             print("model learning rate %s, kernel %s, filter %s,densse %s, strides %s" % (learning_rate, j, k,d,strides))
    #             model = train_model(X_train,y_train, X_test, y_test,learning_rate, j, k,d,strides)


    # for i in range(2):
    #     model = train_model(X_train,y_train, X_test, y_test,0.01, 3, 32, i+1, i+1)

  
    # predict_image_sample(model, X_test, y_test)

    # model2 = train_model2(X_train,y_train, X_test, y_test)
    # predict_image_sample(model2, X_test, y_test)


    # loaded_model = load_model("image_classification.model")
    # print("load model time:", (time.time()-start))

    # for i in range(500):
    #    predict_image_sample(loaded_model, X_test, y_test)
    