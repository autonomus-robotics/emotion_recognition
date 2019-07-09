# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:52:20 2019

@author: ress
"""

import os, glob
import numpy as np
import librosa
import librosa.display
import spec_augment_tensorflow 
from keras import models, layers 
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

path_x = "D:/raw\\"
epochs = 10
batch_size = 16
num_of_files = 0
counter = 0
num_of_augs = 2
max_shape = 0

def au_proc(filename, max_shape):
    
    result = np.zeros( (num_of_augs + 1, max_shape) )
    X, sample_rate = librosa.load(filename)
    melspec = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=256, hop_length=128, fmax=8000)   
    shape = melspec.shape[0]*melspec.shape[1]
 
    for i in range (num_of_augs):
        warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=melspec)     
        warped_masked_spectrogram = warped_masked_spectrogram.reshape(shape)
        for j in range (shape):
            result[i+1][j] = warped_masked_spectrogram[j]
    
    melspec = melspec.reshape(shape)
    for j in range (shape):
        result[0][j] = melspec[j]
    return result

def generator(filenames, labels, batch_size, max_shape):
    count = 0
    lab_count = 0
    x_train = np.zeros((batch_size, max_shape))
    pre_arr = np.zeros((num_of_augs + 1, max_shape))
    for filename in (filenames): 
        pre_arr = au_proc(filename, max_shape)
        for i in range (num_of_augs + 1):
            for j in range (max_shape):
                x_train[count][j] = pre_arr[i][j]
            lab_count = lab_count + 1
            count = count + 1
            if count == batch_size:
                yield x_train.reshape(batch_size, max_shape, 1), labels[lab_count-batch_size:lab_count]
                count = 0
                x_train[:] = 0
        print('Осталось {} файлов')

filenames = glob.glob(os.path.join(path_x, '*/*.wav'))
folders = glob.glob(os.path.join(path_x, '*/'))
num_of_class = len(folders)

#print (filenames)

num_of_files = len(filenames)
num_of_files = num_of_files * (num_of_augs + 1)

labels = np.empty(num_of_files, dtype=object)
labels_arr = np.zeros((num_of_files, num_of_class))

for filename in filenames:
    X, sample_rate = librosa.load(filename)
    melspec = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=256, hop_length=128, fmax=8000) 
    shape = melspec.shape[0]*melspec.shape[1]
    if shape > max_shape: 
        max_shape = shape
    for i in range (num_of_augs + 1):
        labels[counter] = filename[len(path_x):len(path_x) + 2]
        counter = counter + 1
        
max_shape = max_shape + 20
print (max_shape)

labelencoder = LabelEncoder()
labels_arr = labelencoder.fit_transform(labels)
labels_arr = labels_arr.reshape(-1, 1) 
onehotencoder = OneHotEncoder()
labels_arr = onehotencoder.fit_transform(labels_arr).toarray()
labels_arr_end = np.zeros((num_of_files, 1))
for i in range (labels_arr.shape[0]):
    for j in range (labels_arr.shape[1]):
        if labels_arr[i][j] == 1:
            labels_arr_end[i] = j

input_shape = (max_shape, 1)
model = models.Sequential()
model.add(layers.LSTM(16, activation='relu', input_shape=input_shape))
model.add(layers.Dense(16, activation='relu', input_shape=input_shape))
model.add(layers.Dense(num_of_class, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(generator(filenames, labels_arr_end, batch_size, max_shape),
                    steps_per_epoch = num_of_files/batch_size,
                    epochs = epochs)

pickle.dump(model, open('au_aug_model', 'wb'))
model.save('au_em_rec_lstm.h5')
model.save_weights('au_em_rec_lstm_weights.h5')