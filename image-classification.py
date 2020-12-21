import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import keras
import tensorflow_datasets as tfds
import pathlib
import time
import cv2
from tensorflow.keras.models import Sequential, load_model,save_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report

import random
print(tf.__version__)

def get_dataset(base_dir):
    train_ds = image_dataset_from_directory(
        base_dir,
        label_mode='categorical',
        class_names=['food','interior','exterior'],
        color_mode='rgb',
        image_size=(300,300),
        shuffle=True,
        batch_size = 500,
        seed=42,
        validation_split=0.3,
        subset='training'
    )
    validation_ds = image_dataset_from_directory(
        base_dir,
        label_mode='categorical',
        class_names=['food','interior','exterior'],
        color_mode='rgb',
        image_size=(300,300),
        shuffle=True,
        batch_size = 500,
        seed=42,
        validation_split=0.3,
        subset='validation'
    )
    return train_ds, validation_ds

def train_model(train_ds, validation_ds):
    model = Sequential([
        Input(shape=(300,300,3), name='input_layer'),
        Conv2D(16,  kernel_size=3,  activation='relu', padding= 'valid', name='conv_layer1'),
        Conv2D(32,  kernel_size=3,  activation='relu', padding= 'valid', name='conv_layer2'),
        MaxPooling2D(pool_size=2,strides=2),
        BatchNormalization(),
        Dropout(0.3),

        Conv2D(64,  kernel_size=3,  activation='relu', padding= 'valid', name='conv_layer3'),
        Conv2D(128,  kernel_size=3,  activation='relu', padding='valid', name='conv_layer4'),
        MaxPooling2D(pool_size=2,strides=2),
        BatchNormalization(),
        Dropout(0.3),

        Flatten(),
        Dense(32, activation='relu', name='hidden_layer1'),
        Dropout(0.3),
        Dense(16, activation='relu', name='hidden_layer2'),
        Dropout(0.3),
        Dense(3, activation='softmax', name='output_layer')
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    callbacks = [EarlyStopping(monitor="val_accuracy", patience=5),
                 ModelCheckpoint(filepath="my_best_model.model",monitor="val_accuracy",save_best_only=True)]

    history = model.fit(train_ds, validation_data=validation_ds, batch_size=10, epochs=50, callbacks=callbacks)
    plot_loss_curve(history.history)

    return model

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

def plot_accuracy_curve_(history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

def test_for_presentation():
    for images, labels in validation_ds.take(1):
        pred = model.predict(images)
    labels = np.array(labels)
    images = np.array(images).astype(np.uint8)

    y_real = labels.argmax(axis=1)
    y_predicted = pred.argmax(axis=1)
    target_names = ['food','interior','exterior']

    #precision, recall, f1-score
    print(classification_report(y_real, y_predicted,target_names=target_names))

    true_flag = 0
    false_flag = 0
    for i in range(y_predicted.shape[0]):

        y = labels[i].argmax()
        y_pred = pred[i].argmax()

        if y != y_pred:
            false_flag += 1
            if false_flag < 4:
                print("X:", i)
                print("실제:",y,"예측:", y_pred)
                plt.title(i)
                plt.imshow(images[i])
                plt.show()
        else:
            true_flag += 1

            if true_flag < 4:
                print("O:", i)
                print("실제:",y,"예측:", y_pred)
                plt.title(i)
                plt.imshow(images[i])
                plt.show()

    print("맞은개수: ", true_flag, "틀린개수: ", false_flag)



#1 image load
base_dir =  "/content/images"
train_ds, validation_ds = get_dataset(base_dir)

#2 load model
model = load_model('model-201612066')

#3 train model
model = train_model(train_ds, validation_ds)

#4 evalutate
results = model.evaluate(validation_ds, verbose=1)

#5 test for presentation
#6 analyze precision, recall, f1 score
test_for_presentation()


