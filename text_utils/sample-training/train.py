# organize imports
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# the output of prepare_data.py is to be used here
db = 'dataset/texts.csv.out'
testsize = 0.33
max_train_set = 100000
max_test_set = max_train_set

# seed for reproducing same results
seed = 9
np.random.seed(seed)

# load dataset
dataset = np.loadtxt(db, delimiter=',', skiprows=0)

# split into input and output variables
X = dataset[:,0:len(dataset[0]) - 1]
Y = dataset[:,len(dataset[0]) - 1]

# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=testsize, random_state=seed)

if len(X_train) > max_train_set:
  X_train = X_train[0:max_train_set]
  Y_train = Y_train[0:max_train_set]

if len(X_test) > max_test_set:
  X_test = X_train[0:max_test_set]
  Y_test = Y_train[0:max_test_set]

# create the model
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu', kernel_initializer="uniform"))
model.add(Dense(6, activation='relu', kernel_initializer="uniform"))
model.add(Dense(1, activation='sigmoid', kernel_initializer="uniform"))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=5, verbose=0)
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
