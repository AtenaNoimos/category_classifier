import tensorflow as tf
import keras
import configparser
from keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from keras.layers import Layer, BatchNormalization, Dense, GlobalAveragePooling2D, Dropout, Convolution2D, Input
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os
IMG_SIZE = 299
predicted_class = ["car", "close-up", "dashboard", "graycard", "others"]  


class classifier_inference():
    def __init__(self, config_file = "config.ini"):
        """ A class to be used for the inferencing
            Args:
                config_file: a file that contains constants and paths """
        
        self.CLASSES = ["car", "dashboard", "graycard", "others","close-up"]  # Class definations
        # read config file
        config = configparser.ConfigParser()
        config_file = os.path.join(os.path.dirname(__file__), "config.ini")
        config.read(config_file)
        print(config)
        self.IMG_SIZE= int(config["MODEL_PARAMETERS"]["IMG_SIZE"]) #the input imag_size (depends onf the backbone model)   
        CheckPoint_Dir = str(config["MODEL_PARAMETERS"]["CheckPoint_dir"]) 
        self.CheckPoint_Dir = os.path.join(os.path.dirname(__file__), CheckPoint_Dir)
        self.laod_model()
        
    def laod_model(self):
        """ Model architecture 
        Returns:
            - model: the model architecture  """
        InputLayer = Input(shape = (self.IMG_SIZE, self.IMG_SIZE,3), name='input')
        main_input = (InputLayer)
        x = BatchNormalization()(main_input) # normalized data within the model
        x = EfficientNetB0(include_top=False, weights='imagenet')(x) # predefined convolutional neural network which will be initialized with 'imagenet'
        x = GlobalAveragePooling2D()(x)  #  It applies average pooling on the spatial dimensions until each spatial dimension is one, and leaves other dimensions unchanged.:  (samples, h, w, c) would  be output as (samples, 1, 1, 1)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.2)(x)
        clas = Dense(256, activation="relu")(x)
        clas = Dropout(0.2)(clas)
        clas = Dense(5, activation='softmax', name='clas')(clas)
        self._model = keras.Model(inputs=InputLayer,outputs=clas)
        self._model.load_weights(self.CheckPoint_Dir)
        return self._model
        
    def preprocess(self, image:Image.Image):
        image = cv2.resize(image, (self.IMG_SIZE,self.IMG_SIZE), interpolation = cv2.INTER_AREA)
        image = np.array(image/255)
        image = image.reshape(1,self.IMG_SIZE,self.IMG_SIZE,3)
        return image
        
    def predict (self, image: np.ndarray):
        predictions = self._model.predict(image)
        pred_class = predicted_class[np.argmax(predictions)]
        return pred_class

    
    

    

