import cv2
# import keras
import tensorflow as tf
import numpy as np
from tensorflow_core.python.keras.api._v2.keras.models import load_model
from tensorflow_core.python.keras.api._v2.keras.preprocessing import image
from tensorflow_core.python.keras.api._v2.keras.utils import CustomObjectScope
# from tensorflow.python.keras._impl.keras.utils.generic_utils import CustomObjectScope
# from tensorflow.python.keras._impl.keras.applications import mobilenet
# from tensorflow.python.keras._impl.keras.models import load_model
from tensorflow_core.python.keras.api._v2.keras.models import load_model
import keras
from keras.applications import MobileNet
import tensorflow_core
import pickle
import sklearn
from sklearn import svm
# import tensorflow_core
# Load pretrained model
model_face = load_model('./facenet_keras.h5')
model = pickle.load(open('./model_SVM_final_last_17.pkl', 'rb'))
print('run')
IMAGE_SIZE=160
def extract_feature(img):
    y = model_face.predict(prepare_image(img))
    return y

def prepare_image(img):
    img_array = image.img_to_array(img)
    img_array /= 255
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims

def predict(img):
    """
    Predict face crop from frame
    :param img:
    :return: If boss is appear when open the code IDE
    """
    try:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        probs = model.predict(extract_feature(img))
        if probs==0:
            is_boss= 1
            return is_boss
        print(probs)  
    except:
        return False
