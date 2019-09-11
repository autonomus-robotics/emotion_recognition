# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:42:10 2019

@author: ress
"""

import os, glob
import librosa
import librosa.display
from dtw import accelerated_dtw
import pickle
import pyaudio
import wave
import numpy as np
from logmmse import logmmse
import time
from datetime import timedelta
start_time = time.monotonic()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 4

def au_rec(FILENAME):
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
        
def mfcc(file_path, max_pad_len=400):
    wave1, sr = librosa.load(file_path)
    wave = logmmse(wave1, sr)
    librosa.output.write_wav(file_path, wave, sr)
    wave, sr = librosa.load(file_path)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=sr) #16000)
    #print(mfcc.shape)
    if mfcc.shape[1]<400:
        pad_width = max_pad_len - mfcc.shape[1]
    else:
        pad_width = 0
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

path_x = "D:/SPIIRAS_VOICE_DB\\1_2\\"
mfcc_path = 'mfcc15_27_logmmse.pickle'
labels_path = 'labels15_27_logmmse.pickle'
folders = glob.glob(os.path.join(path_x, '*/'))
num_of_files = len(folders)

n = '0'
while n=='0':
    print ('write a number: 1 - to recognize person; 2 - to add person')
    decision = input()
    if decision == '1':
        au_rec("output.wav")
        
        min_dist = 100000
        
        with open(mfcc_path, 'rb') as file:
            dataset_mfcc_2 = pickle.load(file)
        with open(labels_path, 'rb') as file:
            labels = pickle.load(file)
        
        mfcc1 = mfcc("output.wav")
        
        dist = np.zeros((len(dataset_mfcc_2)))
        
        for i in range (len(dataset_mfcc_2)):  
            mfcc2 = dataset_mfcc_2[i]
            dist[i], cost, acc_cost, path = accelerated_dtw(mfcc1.T, mfcc2.T, dist='euclidean')
            #dist, cost, path = dtw(mfcc1.T, mfcc2.T)
            print(labels[i], " : ", dist[i])   # 0 for similar audios 
            if dist[i]<min_dist:
                min_dist = dist[i]
                num_min = i
        
        if (min_dist>35000):
            print("this is guest")
        else:
            print()
            print(labels[num_min], " - ", dist[num_min])
            print()
        
        print('write 0 to continue, write anything except 0 to exit')
        n = input()

    if decision == '2':
        num_of_files = num_of_files + 1
        new_path = path_x + str(num_of_files) 
        if not os.path.exists(new_path):
            os.makedirs(new_path)
                
        print ('write full name of new person')
        full_name = input()
                
        new_path = new_path + '\\' + full_name + '.wav'
        au_rec(new_path)
        
        with open(mfcc_path, 'rb') as file:
            dataset_mfcc = pickle.load(file)
                    
        dataset_mfcc.append(mfcc(new_path))
                
        with open(labels_path, 'rb') as file:
            labels = pickle.load(file)
                
        labels.append(full_name)
                
        with open(mfcc_path, 'wb') as file:
            pickle.dump(dataset_mfcc, file, pickle.HIGHEST_PROTOCOL)
                    
        with open(labels_path, 'wb') as file:
            pickle.dump(labels, file, pickle.HIGHEST_PROTOCOL)
        
        print('write 0 to continue, write anything except 0 to exit')
        n = input()

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))