import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from cv2 import cv2

def predict_image_sample(model, X_test, y_test, test_id=-1):
    if test_id < 0:
        from random import randrange
        test_sample_id = randrange(X_test.shape[1])
    else:
        test_sample_id = test_id
        
    test_image = X_test[test_sample_id]
    


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
    
    cv2.imshow('image',test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
        
    X_test = np.load('X_test_temp.npy')
    y_test = np.load('y_test_temp.npy')
    loaded_model = load_model("image_classification.model")

    predict_image_sample(loaded_model,X_test,y_test,test_id = 238)
    predict_image_sample(loaded_model,X_test,y_test,test_id = 238)