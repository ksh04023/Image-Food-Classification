import os
import numpy as np
from cv2 import cv2
import time

filePath = 'C:\\Users\\ksh04\\Projects\\PythonProjects\\DataScience_Assignment\\FinalProject\\images'

image_files = []
tags = []
flag = 0

for r,d,f in os.walk(filePath):
    for file in f:
        image = cv2.imread(os.path.join(r, file))
        image_resized = cv2.resize(image,(300,300),cv2.INTER_NEAREST)
        image_files.append(image_resized)

        #food = 0, interior = 1, exterior = 2
        if 'food' in file:
            tags.append(0)
        elif 'interior' in file:
            tags.append(1)
        else:
            tags.append(2)

start = time.time()
images = np.array(image_files)
end = time.time()
print((end-start)/1000/60)
tags = np.array(tags)

np.save('image_vectors',images)
np.save('tags',tags)
