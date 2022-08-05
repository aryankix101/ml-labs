from sklearn.datasets import make_regression
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot


X, y = make_regression(n_samples=1000, n_features=20, random_state=1)
train_size = 500
trainX, testX = X[:train_size, :], X[train_size:, :]
trainy, testy = y[:train_size], y[train_size:]

model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(1, activation='linear'))
opt = SGD(lr=0.01, momentum=1.0, clipnorm=1.0)
model.compile(loss='mean_squared_error', optimizer=opt)

history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=150)
train_mse = model.evaluate(trainX, trainy)
test_mse = model.evaluate(testX, testy)

print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
pyplot.title('Average Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()