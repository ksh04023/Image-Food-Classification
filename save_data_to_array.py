import os
import numpy as np
from cv2 import cv2

filePath = '/FinalProject/Image-Food-Classification/images'

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

images = np.array(image_files)
tags = np.array(tags)

X_train, X_test, y_train, y_test = train_test_split(images, tags, test_size=0.3, random_state=42)
np.save('X_train',X_train)
np.save('X_test',X_test)
np.save('y_train',y_train)
np.save('y_test',y_test)