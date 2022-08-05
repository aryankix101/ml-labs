from tensorflow import keras
import tensorflow as tf
import pandas as pd 
from sklearn.preprocessing import normalize 
from tensorflow.keras import layers 
from tensorflow.keras import utils as np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import numpy as np 
import matplotlib.pyplot as plt

def create_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='random_normal', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(10, activation='softmax'))
    return model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = create_model()
optimizer = keras.optimizers.Adam(lr=0.01)
#Error calculations
model.compile(optimizer=optimizer, metrics=['accuracy'], loss="BinaryCrossentropy")
#Where backprop/forwardprop occur
model_history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))


predict_label = np.argmax(model.predict(x_test), axis=1)
y_label = np.argmax(y_test, axis=1)

print("Accuracy: ", np.sum(y_label==predict_label)/len(model.predict(x_test)) * 100 )

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()