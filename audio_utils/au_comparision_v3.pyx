# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:01:42 2019

@author: ress
"""

import librosa
from dtw import accelerated_dtw
import pickle
import cython
import time
from datetime import timedelta

#cdef str FILENAME = "output.wav"

cdef list dataset_mfcc_2
cdef list labels

cdef float di
cdef float y[97373]
cdef int sr
cdef float cost
cdef float acc_cost
cdef float mfcc1
cdef int num_min
cdef int i
cdef tuple path

def compare(str FILENAME):
    cdef float min_dist = 100000
    cdef float start_time = time.monotonic()
    with open('mfcc.pickle', 'rb') as file:
        dataset_mfcc_2 = pickle.load(file)
    with open('labels.pickle', 'rb') as file:
        labels = pickle.load(file)
    #print ('1')
    y, sr = librosa.load(FILENAME)
    mfcc1 = librosa.feature.mfcc(y, sr)   #Computing MFCC values
    
    for i in range (len(dataset_mfcc_2)):
        di, cost, acc_cost, path = accelerated_dtw(mfcc1.T, dataset_mfcc_2[i].T, dist='euclidean')
        #print(labels[i], " : ", di)   # 0 for similar audios 
        if di<min_dist:
            min_dist = di
            num_min = labels[i]
    #print()
    print(num_min, " - ", di)
    
    cdef float end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    
    
compare('output.wav')