from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image
from keras.optimizers import RMSprop


#---------- НАСТРОЙКИ ----------#
# Каталог с данными для обучения
train_dir = 'X:\\work_dir\\img\\train\\'
# Каталог с данными для проверки
val_dir = 'X:\\work_dir\\img\\validation\\'
# Каталог с данными для тестирования
test_dir = 'X:\\work_dir\\img\\test\\'
# Размеры изображения
img_width, img_height = 600, 600

# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)

# Количество эпох   ## 30
epochs = 20
# Размер мини-выборки   ## 16
batch_size = 16
# Количество изображений для обучения
nb_train_samples = 87360
# Количество изображений для проверки
nb_validation_samples = 18720
# Количество изображений для тестирования
nb_test_samples = 18720



#---------- Создаем сверточную нейронную сеть ----------#

# Архитектура сети
#
# 1. Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., функция активации ReLU.
# 2. Слой подвыборки, выбор максимального значения из квадрата 2х2
# 3. Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., функция активации ReLU.
# 4. Слой подвыборки, выбор максимального значения из квадрата 2х2
# 5. Слой свертки, размер ядра 3х3, количество карт признаков - 64 шт., функция активации ReLU.
# 6. Слой подвыборки, выбор максимального значения из квадрата 2х2
# 7. Слой преобразования из двумерного в одномерное представление
# 8. Полносвязный слой, 64 нейрона, функция активации ReLU.
# 9. Слой Dropout.
# 10. Выходной слой, 1 нейрон, функция активации sigmoid
# Слои с 1 по 6 используются для выделения важных признаков в изображении, а слои с 7 по 10 - для классификации.


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))



# Компилируем нейронную сеть
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



#---------- Создаем генератор изображений ----------#

# Генератор изображений создается на основе класса ImageDataGenerator.
# Генератор делит значения всех пикселов изображения на 255.
datagen = ImageDataGenerator(rescale=1. / 255)

# Генератор данных для обучения на основе изображений из каталога
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Генератор данных для проверки на основе изображений из каталога
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Генератор данных для тестирования на основе изображений из каталога
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')



#---------- Обучаем модель с использованием генераторов ----------#

# train_generator - генератор данных для обучения
# validation_data - генератор данных для проверки

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)



#---------- Оцениваем качество работы сети с помощью генератора ----------#

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))