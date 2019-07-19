# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:52:20 2019

@author: ress
"""

import os, glob
from keras import models, layers
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator

batch_size = 16
epochs = 10
path_x = "D:/raw\\"
height_width = 224
train_data_dir = path_x + 'train\\'
val_data_dir = path_x + 'val\\'

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(height_width, height_width, 3))
base_model.summary()

filenames = glob.glob(os.path.join(train_data_dir, '*/*.jpg'))
nb_train_samples = len(filenames)
filenames = glob.glob(os.path.join(val_data_dir, '*/*.jpg'))
nb_val_samples = len(filenames)
folders = glob.glob(os.path.join(train_data_dir, '*/'))
num_of_class = len(folders)

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(height_width, height_width),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = datagen.flow_from_directory(
    val_data_dir,
    target_size=(height_width, height_width),
    batch_size=batch_size,
    class_mode='categorical')

base_model.trainable = False

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(num_of_class, activation='softmax'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator,
    samples_per_epoch = nb_train_samples,
    epochs = epochs,
    validation_data = val_generator,
    nb_val_samples = nb_val_samples)

model.save('au_aug_model.h5')
model.save_weights('au_aug_model_weights.h5')