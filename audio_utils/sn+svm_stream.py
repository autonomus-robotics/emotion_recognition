# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:11:44 2019

@author: OlgaS
"""

import numpy as np
import pyaudio
from sklearn import svm
import pickle
from keras.models import load_model
import time
from datetime import timedelta

support_vm = svm.SVC(kernel='linear', C=1)
num = 5760
data = np.empty((num, 1024))
labels = np.empty((num))

pic_data = open('au_aug_ampl_features_from_sn_1_vector.pickle', 'rb')
pic_label = open('au_aug_ampl_labels_from_sn_1_vector.pickle', 'rb')

for i in range (num):
    data[i] = pickle.load(pic_data)
    labels[i] = pickle.load(pic_label)

support_vm.fit(data, labels)
model = load_model('my_soundnet.h5')

class Audio:  # TODO: Docstring
    def __init__(self, seconds, rate=22050, chunk=1024):
        self.rate = rate
        self.chunk = chunk

        self.audio_len = int(rate / chunk * seconds)
        self.audio = [[]] * self.audio_len

        self.p = pyaudio.PyAudio()
        self.stream = self.open_stream()
    
    def open_stream(self):
        stream = self.p.open(format=pyaudio.paFloat32,
                             channels=1,
                             rate=self.rate,
                             input=True,
                             frames_per_buffer=self.chunk)
        return stream

    def update_stream(self):  # TODO: asyncio maybe
        while True:
            try:
                byte_data = self.stream.read(self.chunk)
                data = np.frombuffer(byte_data, dtype=np.float32)
                # print(any(x != 0 for x in data))

                data = self.denoise_array(data)
                self.array_to_mfcc(data, self.rate)
                # noinspection PyTypeChecker
                self.audio.append(1)
                del self.audio[0]
                return self.audio

            except IOError as e:
                print(e)
                self.stream = self.open_stream()

    @staticmethod
    def denoise_array(array):  # TODO: Denoise
        return array

    @staticmethod
    def array_to_mfcc(array, rate=44100):
        #melspec = librosa.feature.melspectrogram(y=array, sr=rate, n_mels=256, hop_length=128, fmax=8000)
       # melspec = melspec.reshape((1, melspec.shape[0]*melspec.shape[1], 1))
        #data = array
        
        start_time = time.monotonic()
        data = array * 256.0
        data = np.reshape(data, (1, -1, 1))
        pred =  model.predict(data)
        
        pred = pred.reshape(1, pred.shape[2])
        end_time = time.monotonic()
        print(timedelta(seconds=end_time - start_time))
        print(support_vm.predict(pred))


if __name__ == "__main__":
    audio = Audio(3)
    print(audio.p.get_default_input_device_info())
    while True:
        audio.update_stream()
        