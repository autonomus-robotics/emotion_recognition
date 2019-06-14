# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:47:14 2019

@author: ress
"""

import scipy.io.wavfile as wav
import os, glob, random
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import spec_augment_tensorflow 

def amplitude(time, data):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time, data, linewidth=0.02, alpha=1, color='#000000')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

import pylab

path_x = "D:/DB_audio/"
rate = []
x_train = np.zeros( (3300, 130, 700) )
k=-1

for filename in glob.glob(os.path.join(path_x, '*.wav')):
    rate_value, x_train_value = wav.read(filename)
            
    X, sample_rate = librosa.load(filename)
    melspec = librosa.feature.melspectrogram(X, sr=sample_rate)
    #librosa.display.specshow(melspec, x_coords=None, y_coords=None, x_axis=None, y_axis=None, sr=22050, hop_length=512, fmin=None, fmax=None, bins_per_octave=12, ax=None, **kwargs)
    save_path = 'test.jpg'

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    melspec = librosa.feature.melspectrogram(y=X,
                                             sr=sample_rate,
                                             n_mels=256,
                                             hop_length=128,
                                             fmax=8000)
    l = melspec.shape[0]
    ll = melspec.shape[1]
    k = k + 1
    for j in range(l):
        for jj in range(ll):
            x_train[k][j][jj] = melspec[j][jj]
    
    for pp in range(10):
        warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=melspec)
        #print(warped_masked_spectrogram)
        #spec_augment_tensorflow.visualization_spectrogram(melspec, 'before')
        #spec_augment_tensorflow.visualization_spectrogram(warped_masked_spectrogram, 'after')
        
        k = k + 1
        for j in range(l):
            for jj in range(ll):
                x_train[k][j][jj] = warped_masked_spectrogram[j][jj]
        
        librosa.display.specshow(librosa.power_to_db(melspec, ref=np.max))
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()
  
  
x_train = x_train.reshape(1, 3300, 91000)
print(len(x_train))
#print(rate)
#print(data)

labels = np.zeros( (1, 3300, 1) )
for i in range(3300):
    j =  random.randint(0, 0)
    labels[0][i][j] = 1
    continue

from AudioDataGenerator import AudioDataGenerator

datagen = AudioDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                shift=.2,
                horizontal_flip=True,
                zca_whitening=True)

from keras import models, layers

print()
print (len(x_train))
print ()

input_shape = (3300, 91000)
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=input_shape))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7, activation='softmax')) #7 классов

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, labels, batch_size=16),
                    steps_per_epoch = 3300/16,
                    epochs = 10)