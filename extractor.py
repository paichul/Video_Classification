from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input, Cropping2D, Reshape
import numpy as np
import cv2
from PIL import Image

class Extractor():
    def __init__(self):
        base_model = InceptionV3(weights='imagenet',
                                 include_top=True)

        self.model = Model(inputs = base_model.input,
                           outputs=base_model.get_layer('avg_pool').output)

    def extract(self, image_path):
        # feature vector size 2048
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = self.model.predict(x)
        features = features[0]
        return features

    def extract_multires(self, image_path):
        # feature vector size 4096
        img = image.load_img(image_path, target_size=(299, 299))
        img1 = img.crop((50, 50, 200, 200))
        img1 = img1.resize((299,299), Image.ANTIALIAS)
                
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        x1 = image.img_to_array(img1)
        x1 = np.expand_dims(x1, axis=0)
        x1 = preprocess_input(x1)

        features = self.model.predict(x)
        features1 = self.model.predict(x1)
        features = features[0]
        features1 = features1[0]

        #return features + features1
        return np.append(features, features1)
