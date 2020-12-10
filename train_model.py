import numpy as np
import matplotlib.pyplot as plt
    
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import time

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

def train_model(X_train,y_train, X_test, y_test):

    model = Sequential([
                Input(shape=(300,300,3), name='input_layer'),

                Conv2D(32,  kernel_size=3, strides = 1, activation='relu', name='conv_layer1'),
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

    model.summary()    
    
    #learning rate 변경해보기
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=20, epochs=3)
    # plot_loss_curve(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])    
    
    model.save('model-201612066.model')
    
    return model
def train_loaded_model(file_name):
    model = load_model(file_name)

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

    # X_train = np.load('X_train.npy')
    # X_test = np.load('X_test.npy')
    # y_train = np.load('y_train.npy')
    # y_test = np.load('y_test.npy')

    # #test용 train data = 10000개, test data = 5000개
    X_train = np.load('X_train_temp.npy')
    X_test = np.load('X_test_temp.npy')
    y_train = np.load('y_train_temp.npy')
    y_test = np.load('y_test_temp.npy')
    print(X_test)
    print("load data time:", (time.time()-start))
    
    model = train_model(X_train,y_train, X_test, y_test)
    predict_image_sample(model, X_test, y_test)

    loaded_model = load_model("image_classification.model")
    print("load model time:", (time.time()-start))

    # for i in range(500):
    #    predict_image_sample(loaded_model, X_test, y_test)
    