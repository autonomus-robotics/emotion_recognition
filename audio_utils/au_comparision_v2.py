# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:02:31 2019

@author: ress
"""

import librosa
import librosa.display
import os, glob
import pickle

path = "D:/raw_compare\\"
filenames = glob.glob(os.path.join(path, '*/*.wav'))
num_of_files = len(filenames)
print(num_of_files)
dataset_mfcc = [0] * num_of_files
labels = [0] * num_of_files
i = 0

for filename in filenames:
    y2, sr2 = librosa.load(filename) 

    mfcc = librosa.feature.mfcc(y2, sr2)
    dataset_mfcc[i] = mfcc
    labels[i] = filename[len(path):filename.find('\\', len(path)+1, len(filename))]
    print(labels[i])
    #print(dataset_mfcc[i])
    i = i + 1
    #print(filename, mfcc.shape, mfcc.shape[1])

with open('mfcc.pickle', 'wb') as file:
    pickle.dump(dataset_mfcc, file, pickle.HIGHEST_PROTOCOL)
    
with open('labels.pickle', 'wb') as file:
    pickle.dump(labels, file, pickle.HIGHEST_PROTOCOL)

with open('mfcc.pickle', 'rb') as file:
    dataset_mfcc_2 = pickle.load(file)
    
with open('labels.pickle', 'rb') as file:
    lebelss = pickle.load(file)
