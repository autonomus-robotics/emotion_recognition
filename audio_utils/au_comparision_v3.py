# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:02:31 2019

@author: ress
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
from dtw import dtw
from numpy.linalg import norm
import pickle
import pyaudio
import wave
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
FILENAME = "output.wav"

def au_rec():
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("* recording")
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("* done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

#au_rec()

min_dist = 100000

with open('mfcc.pickle', 'rb') as file:
    dataset_mfcc_2 = pickle.load(file)
with open('labels.pickle', 'rb') as file:
    labels = pickle.load(file)

y1, sr1 = librosa.load(FILENAME)
mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values

euclidean_norm = lambda x, y: norm(x - y, ord=1) 
dist = np.zeros((len(dataset_mfcc_2)))

for i in range (len(dataset_mfcc_2)):  
    mfcc2 = dataset_mfcc_2[i]
    dist[i], cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=euclidean_norm)
    #dist, cost, path = dtw(mfcc1.T, mfcc2.T)
    print("The normalized distance between the two : ",dist[i])   # 0 for similar audios 
    if dist[i]<min_dist:
        min_dist = dist[i]
        num_min = i

print(dist)
print(labels[num_min])
