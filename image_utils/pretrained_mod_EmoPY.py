# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:57:03 2019

@author: ress
"""

from keras.layers import Input, Conv2D
from keras.models import load_model, Model
import numpy as np
from cv2 import imread

# path to the model file
model_path = 'conv_model_012.hdf5'
# dimensions of our images
img_width, img_height = 640, 490
# path to the image
image_path = 'S042_004_00000012.png'

#loading of EmoPY pretrained model 012 with target emotions [fear, anger, disgust]
pretrained_model = load_model(model_path)
pretrained_model.summary()

#loading and reshaping of our image
image = imread(image_path, 0)
image = image.reshape(img_width, 49, 10)
print(image.shape)
    
# Disassemble layers
layers = [l for l in pretrained_model.layers]

x_input = Input(shape=(img_width, 49, 10))
x = x_input

for i in range(1, len(layers) - 1):
    print(layers[i](x).shape)
    x = layers[i](x)

#our model
model = Model(input=x_input, output=x)
model.summary()

image = np.expand_dims(image, axis=0)
print(image.shape)
    
print(model.predict(image, steps=1))