from tensorflow import keras
import tensorflow as tf
import pandas as pd 
import pandas as pd 
from sklearn.preprocessing import normalize 
from tensorflow.keras import layers 
from tensorflow.keras import utils as np_utils
import numpy as np 


df = pd.read_csv('./iris.csv')
df.loc[df["variety"]=="Setosa","variety"]=0
df.loc[df["variety"]=="Versicolor","variety"]=1
df.loc[df["variety"]=="Virginica","variety"]=2
df = df.iloc[np.random.permutation(len(df))]
var_t = 4
X=df.iloc[:,1:var_t].values
y=df.iloc[:,var_t].values
X_normalized = normalize(X, axis=0)
train_len = int(len(df)*0.7)
test_len = int(len(df)*0.3)

X_train = X_normalized[:train_len]
X_test = X_normalized[test_len:]
y_train = y[:train_len]
y_test = y[test_len:]
y_train = np_utils.to_categorical(y_train, num_classes=3)
y_test = np_utils.to_categorical(y_test, num_classes=3)
model = tf.keras.Sequential()
model.add(layers.Dense(1000, input_dim=3,activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, metrics=['accuracy'], loss="BinaryCrossentropy")

model.fit(X_train,y_train,validation_data=(X_test,y_test), batch_size=10, epochs=10)

predict_label = np.argmax(model.predict(X_test), axis=1)
y_label = np.argmax(y_test, axis=1)

print("Accuracy: ", np.sum(y_label==predict_label)/len(model.predict(X_test)) * 100 )