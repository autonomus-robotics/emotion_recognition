# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:25:22 2019

@author: ress
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import spec_augment_tensorflow 
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pylab
from keras import models, layers
from AudioDataGenerator import AudioDataGenerator

path_x = "D:/DB_audio/"
rate = []
epochs = 10
batch_size = 16
num_of_files = 0
count = 0
num_of_augs = 1
max_shape = 0
num_of_class = 5

def build_model():
    model_weights = np.load('sound8.npy',encoding = 'latin1').item()
    print(type(model_weights))
    model = models.Sequential()
    model.add(layers.InputLayer(batch_input_shape=(1, None, 1)))
    
    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},
    
                        {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                         'kernel_size': 32, 'conv_strides': 2,
                         'pool_size': 8, 'pool_strides': 8},
    
                        {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                         'kernel_size': 16, 'conv_strides': 2},
    
                        {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                         'kernel_size': 8, 'conv_strides': 2},
    
                        {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                         'kernel_size': 4, 'conv_strides': 2,
                         'pool_size': 4, 'pool_strides': 4},
    
                        {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                         'kernel_size': 4, 'conv_strides': 2},
    
                        {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                         'kernel_size': 4, 'conv_strides': 2},
    
                        {'name': 'conv8_2', 'num_filters': 401, 'padding': 1,
                         'kernel_size': 8, 'conv_strides': 2},
                        ]
    
    for x in filter_parameters:
        model.add(layers.ZeroPadding1D(padding=x['padding']))
        model.add(layers.Conv1D(x['num_filters'],
                                kernel_size=x['kernel_size'],
                                strides=x['conv_strides'],
                                padding='valid'))
        weights = model_weights[x['name']]['weights'].reshape(model.layers[-1].get_weights()[0].shape)
        biases = model_weights[x['name']]['biases']
    
        model.layers[-1].set_weights([weights, biases])
    
        if 'conv8' not in x['name']:
            gamma = model_weights[x['name']]['gamma']
            beta = model_weights[x['name']]['beta']
            mean = model_weights[x['name']]['mean']
            var = model_weights[x['name']]['var']
            
            model.add(layers.BatchNormalization())
            model.layers[-1].set_weights([gamma, beta, mean, var])
            model.add(layers.Activation('relu'))
            
        if 'pool_size' in x:
            model.add(layers.MaxPooling1D(pool_size=x['pool_size'],
                                          strides=x['pool_strides'],
                                          padding='valid'))
    return model
            
filenames = glob.glob(os.path.join(path_x, '*/*.wav'))
#model.summary()

for filename in filenames:

    X, sample_rate = librosa.load(filename)
    save_path = 'test.jpg'

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    melspec = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=256, hop_length=128, fmax=8000)
    
    shape = melspec.shape[0]*melspec.shape[1]
    print(melspec.shape, shape)
    
    for i in range(num_of_augs):
        warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=melspec)
        #print(warped_masked_spectrogram)
        #spec_augment_tensorflow.visualization_spectrogram(melspec, 'before')
        #spec_augment_tensorflow.visualization_spectrogram(warped_masked_spectrogram, 'after')
        
        warped_masked_spectrogram = warped_masked_spectrogram.reshape((1, shape, 1))
        model = build_model()
        model.predict(warped_masked_spectrogram)
        
        librosa.display.specshow(librosa.power_to_db(melspec, ref=np.max))
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()
        
    melspec = melspec.reshape((1, shape, 1))
    model = build_model()
    model.predict(melspec)