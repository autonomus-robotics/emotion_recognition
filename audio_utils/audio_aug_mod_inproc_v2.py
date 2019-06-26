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
epochs = 10
batch_size = 16
num_of_files = -1
count = -1
num_of_augs = 10
max_shape = 0

for filename in glob.glob(os.path.join(path_x, '*_/*.wav')):
    num_of_files = num_of_files + 1
num_of_files = num_of_files + num_of_files * num_of_augs
#print(count_of_files)
x_train = [0] * num_of_files 

for filename in glob.glob(os.path.join(path_x, '*_/*.wav')):
    count = count + 1
    rate_value, x_train_value = wav.read(filename)
            
    X, sample_rate = librosa.load(filename)
    save_path = 'test.jpg'

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    melspec = librosa.feature.melspectrogram(y=X,
                                             sr=sample_rate,
                                             n_mels=256,
                                             hop_length=128,
                                             fmax=8000)
        
    shape = melspec.shape[0]*melspec.shape[1]
    print(len(x_train), melspec.shape, shape)
    if shape > max_shape: 
        max_shape = shape
    x_train[count] = [0] * shape
    shape = 0
    for i in range (melspec.shape[0]):
        for j in range (melspec.shape[1]):
            x_train[count][shape] = melspec[i][j]
            shape - shape + 1
    count = count + 1
    #print(len(x_train[count]), melspec.shape) 
    
    for i in range(num_of_augs):
        warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=melspec)
        print(warped_masked_spectrogram)
        spec_augment_tensorflow.visualization_spectrogram(melspec, 'before')
        spec_augment_tensorflow.visualization_spectrogram(warped_masked_spectrogram, 'after')
        
        shape = warped_masked_spectrogram.shape[0]*warped_masked_spectrogram.shape[1]
        if shape > max_shape: 
            max_shape = shape
        x_train[count] = [0] * shape
        shape = 0
        for i in range (warped_masked_spectrogram.shape[0]):
            for j in range (warped_masked_spectrogram.shape[1]):
                x_train[count][shape] = warped_masked_spectrogram[i][j]
                shape - shape + 1
        count = count + 1
        
        librosa.display.specshow(librosa.power_to_db(melspec, ref=np.max))
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()

x_train = x_train.reshape(1, num_of_files, max_shape)

labels = np.zeros( (1, num_of_files, 1) )
for i in range(num_of_files):
    labels[0][i][0] = random.randint(0,9)
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

input_shape = (num_of_files, max_shape)
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=input_shape))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7, activation='softmax')) #7 классов

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, labels, batch_size=batch_size),
                    steps_per_epoch = num_of_files/batch_size,
                    epochs = epochs)