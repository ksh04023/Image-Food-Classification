import numpy as np
import matplotlib.pyplot as plt
    
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import load_model
from sklearn.model_selection import train_test_split
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

def train_model():

    model = Sequential([
                Input(shape=(300,300,3), name='input_layer'),
                
                # n_filters * (filter_size + 1) = 32*(9+1) = 320
                Conv2D(32, kernel_size=3, activation='relu', name='conv_layer1'),
                #Conv2D(64, kernel_size=3, activation='relu', name='conv_layer1'),
                
                #Dropout(0.5)
                MaxPooling2D(pool_size=2),
                Flatten(),
                #Dense(20, activation='softmax', name='output_layer')
                Dense(3, activation='softmax', name='output_layer')
            ])

    model.summary()    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=3)
    plot_loss_curve(history.history)
    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])    
    
    model.save('image_classification.model')
    
    return model

def predict_image_sample(model, X_test, y_test, test_id=-1):
    if test_id < 0:
        from random import randrange
        test_sample_id = randrange(13500)
    else:
        test_sample_id = test_id
        
    test_image = X_test[test_sample_id]
    
    plt.imshow(test_image)
    
    test_image = test_image.reshape(1,300,300,3)

    y_actual = y_test[test_sample_id]
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

def split_dataset(X, y, test_proportion):
    ratio = int(X.shape[0]/test_proportion) #should be int
    X_train = X[ratio:,:]
    X_test =  X[:ratio,:]
    y_train = y[ratio:,:]
    y_test =  y[:ratio,:]
    return X_train, X_test, y_train, y_test

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == '__main__':
    start = time.time()
    loaded_images = np.load('image_vectors.npy')
    print((time.time()-start))

    loaded_tags = np.load('tags.npy')
    loaded_tags = to_categorical(loaded_tags)

    start = time.time()
    # X_train, X_test, y_train, y_test = train_test_split(loaded_images, loaded_tags, test_size=0.3, random_state=42)
    X,y = unison_shuffled_copies(loaded_images,loaded_tags)
    print(X.shape)
    print(y.shape)
    print((time.time()-start))
    
    X_train, X_test, y_train, y_test = split_dataset(X,y,30)

    # model = train_model()
    # predict_image_sample(model, X_test, y_test)

    loaded_model = load_model("image_classification.model")
    for i in range(500):
       predict_image_sample(loaded_model, X_test, y_test)
    