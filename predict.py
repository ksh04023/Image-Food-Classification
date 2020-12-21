import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model

from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
from sklearn.metrics import classification_report

def get_dataset(base_dir):
    train_ds = image_dataset_from_directory(
        base_dir,
        label_mode='categorical',
        class_names=['food', 'interior', 'exterior'],
        color_mode='rgb',
        image_size=(300, 300),
        shuffle=True,
        seed=42,
        validation_split=0.3,
        subset='training'
    )
    validation_ds = image_dataset_from_directory(
        base_dir,
        label_mode='categorical',
        class_names=['food', 'interior', 'exterior'],
        color_mode='rgb',
        image_size=(300, 300),
        shuffle=True,
        seed=42,
        validation_split=0.3,
        subset='validation'
    )
    return train_ds, validation_ds

def classification_performance_eval(y, y_predict):
    metrics = np.zeros((3, 3))
    for y, yp in zip(y, y_predict):
        metrics[y][yp] += 1
    print(metrics)
    return metrics
# model = load_model('model86.hdf5')
# model.save('model-201612066')
model = load_model('model-201612066')
base_dir = "./images"

train_ds, validation_ds = get_dataset(base_dir)

# score = model.evaluate(validation_ds,verbose=1, batch_size=128)
# print(score)

for images, labels in validation_ds.take(-1):
    pred = model.predict(images)


labels = np.array(labels)
images = np.array(images).astype(np.uint8)

y_real = labels.argmax(axis=1)
y_predicted = pred.argmax(axis=1)
target_names = ['food','interior','exterior']
print(classification_report(y_real, y_predicted,target_names=target_names))

print(y_real)
print(y_predicted)
true_flag = 0
for i in range(y_predicted.shape[0]):
    # print(images[i])
    # print(type(images[i]))
    # plt.imshow(images[i], vmin=0, vmax=1)
    # plt.show()
    # print(pred[i])
    y = labels[i].argmax()
    y_pred = pred[i].argmax()

    if y != y_pred:
        print("X:", i)
        print("실제:",y,"예측:", y_pred)
        plt.title(i)
        plt.imshow(images[i])
        plt.show()
    else:
        if true_flag < 3:
            true_flag += 1
            print("O:", i)
            print("실제:",y,"예측:", y_pred)
            plt.title(i)
            plt.imshow(images[i])
            plt.show()




    # plt.imshow(np.array(images[0]),vmin=0, vmax=1)
    # plt.show()

