# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:52:43 2019

@author: ress
"""

import os, glob
import librosa
import librosa.display
import pickle
import numpy as np
import time
from datetime import timedelta
start_time = time.monotonic()

def mfcc(file_path, max_pad_len=400):
    wave, sr = librosa.load(file_path)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=sr) #16000)
    #print(mfcc.shape)
    if mfcc.shape[1]<400:
        pad_width = max_pad_len - mfcc.shape[1]
    else:
        pad_width = 0
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

path_x = "D:/wav\\"
mfcc_path = path_x + 'mfcc.pickle'
labels_path = path_x + 'labels.pickle'
folders = glob.glob(os.path.join(path_x, '*/'))
num_of_files = len(folders)
print(num_of_files)

dataset_mfcc = [0] * num_of_files
labels = [0] * num_of_files
i = 0

for filename in folders:
    path = glob.glob(os.path.join(filename, '*/*.wav'))
    dataset_mfcc[i] = mfcc(path[0])
    labels[i] = filename[len(path_x):filename.find('\\', len(path_x)+1, len(filename))]
    print(labels[i])
    #print(dataset_mfcc[i])
    i = i + 1
    #print(filename, mfcc.shape, mfcc.shape[1])

with open(mfcc_path, 'wb') as file:
    pickle.dump(dataset_mfcc, file, pickle.HIGHEST_PROTOCOL)
            
with open(labels_path, 'wb') as file:
    pickle.dump(labels, file, pickle.HIGHEST_PROTOCOL)
        
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))