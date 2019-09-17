# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:55:36 2019

@author: ress
"""

import design
from PyQt5 import QtWidgets
import sys

import os, glob
import librosa
import librosa.display
import acc_dtw
import pickle
import pyaudio
import wave
import numpy as np
import multiprocessing as mp
import firwin

class MainWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):
   def __init__(self):
      super().__init__()
      self.setupUi(self)  # Это нужно для инициализации нашего дизайна
      self.Add_Button.clicked.connect(self.add_person)
      self.Recognize_Button.clicked.connect(self.recognize_person)
      self.RECORD_SECONDS = 4
      self.path_x = "D:/SPIIRAS_VOICE_DB\\1_2\\"
      #self.mfcc_path = 'mfcc_spiiras.pickle'
      #self.labels_path = 'labels_spiiras.pickle'
      self.mfcc_path = 'mfcc_spiiras_FIR.pickle'
      self.labels_path = 'labels_spiiras_FIR.pickle'
      self.CHUNK = 1024
      self.FORMAT = pyaudio.paInt16
      self.CHANNELS = 2
      self.RATE = 44100
      self.RECORD_SECONDS = 4
      
      self.M = 100 #number of taps in filter
      self.fc = 0.25 #i.e. normalised cutoff frequency 1/4 of sampling rate i.e. 25Hz
      self.lp = firwin.build_filter(self.M, self.fc, window=firwin.blackman)
      self.shift = np.cos(2*np.pi*0.5*np.arange(self.M + 1))
      self.hp = self.lp*self.shift
      
           
   def au_rec(self, FILENAME):
        w = QtWidgets.QWidget()
        w.resize(300,1)
        w.setWindowTitle('Recording...')
        
        p = pyaudio.PyAudio()
    
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
      
        w.show()
        
        frames = []
    
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)
    
        w.close()
    
        stream.stop_stream()
        stream.close()
        p.terminate()
    
        wf = wave.open(FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
   def mfcc(self, file_path):
        wave1, sr = librosa.load(file_path)
        wave = np.convolve(wave1, self.hp)
        #self.textEdit.append(wave.shape)
        wave = wave[::3]
        mfcc = librosa.feature.mfcc(wave, sr=sr) #16000)
        #print(mfcc.shape)
        return mfcc
    
   def add_person(self):
      self.textEdit.clear()
      folders = glob.glob(os.path.join(self.path_x, '*/'))
      num_of_files = len(folders)
      num_of_files += 1
      new_path = self.path_x + str(num_of_files) 
      if not os.path.exists(new_path):
          os.makedirs(new_path)
                
      full_name = self.FIO_Edit.toPlainText()
      if full_name=='':
          return self.textEdit.append('Write fullname of person in widget near buttons \nAdding is interrupted')
                
      new_path = new_path + '\\' + full_name + '.wav'
      self.au_rec(new_path)
               
      with open(self.mfcc_path, 'rb') as file:
          dataset_mfcc = pickle.load(file)
                    
      dataset_mfcc.append(self.mfcc(new_path))
                
      with open(self.labels_path, 'rb') as file:
          labels = pickle.load(file)
                
      labels.append(full_name)
                
      with open(self.mfcc_path, 'wb') as file:
          pickle.dump(dataset_mfcc, file, pickle.HIGHEST_PROTOCOL)
                    
      with open(self.labels_path, 'wb') as file:
          pickle.dump(labels, file, pickle.HIGHEST_PROTOCOL)       
        
      self.textEdit.append('Done.')
      self.FIO_Edit.clear()
    
   def recognize_person(self):
      self.textEdit.clear()
      self.FIO_Edit.clear()
      self.au_rec("output.wav")
        
      with open(self.mfcc_path, 'rb') as file:
          dataset_mfcc_2 = pickle.load(file)
      with open(self.labels_path, 'rb') as file:
          labels = pickle.load(file)
      
      mfcc1 = self.mfcc("output.wav")
      min_shape_mfcc = min((mfcc.shape[1]) for mfcc in dataset_mfcc_2)
      if (mfcc1.shape[1]<min_shape_mfcc):
          min_shape_mfcc = mfcc1.shape[1]
      
      pool = mp.Pool(processes=mp.cpu_count())
      pool_test_dataset = ((mfcc1[:, :min_shape_mfcc],
                             mfcc2[:, :min_shape_mfcc]) for mfcc2 in dataset_mfcc_2)
      distances = pool.map(acc_dtw.accelerated_dtw, pool_test_dataset)
      num_min = np.argmin(distances)
      
      strr = str(labels[num_min]) + " - " + str(distances[num_min])
      self.textEdit.append(strr)
      pool.close()
      
if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = MainWindow()  # Создаём объект класса MainWindow
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение