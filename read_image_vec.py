import csv
import numpy as np
import ast
import os
from cv2 import cv2
from matplotlib import pyplot as plt

def csvWriter(file_name, nparray):
    example = nparray.tolist()
    with open(file_name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(example)

def csvReader():
    file = "imageVecList.csv"
    vec = []
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i in reader:
            arraytemp = []
            for j in i:
                temp_reader = csv.reader(j.split('[',']'), delimiter=',')
                for k in temp_reader:
                    print(type(k))
                    k = np.array(k)
                    print(k[0])
            vec.append(arraytemp)

    return np.array(vec)

def readNp(fileName):
    loaded_array = np.load('image_vectors.npy')
    return loaded_array

if __name__ == "__main__": 
    loaded_array = np.load('image_vectors.npy')
    loaded_tags = np.load('tags.npy')
    tempImage = loaded_array[0]
    print(loaded_tags.shape)
    