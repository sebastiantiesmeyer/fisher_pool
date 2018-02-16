import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(3,3,input_shape=(56,56,1)))
model.add(keras.layers.pooling.MaxPool2D(2))
model.add(keras.layers.Conv2D(3,3))
model.add(keras.layers.pooling.MaxPool2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='SGD',loss='categorical_crossentropy')

def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=x_train.repeat(2,1).repeat(2,2).reshape([-1,56,56,1])
    x_test=x_test.repeat(2,1).repeat(2,2).reshape([-1,56,56,1])
    x_train=x_train+np.random.random(x_train.shape)
    x_test=x_test+np.random.random(x_test.shape)
    t_train = np.zeros([y_train.shape[0],10])
    t_train[np.arange(t_train.shape[0]),y_train]=1
    t_test = np.zeros([y_test.shape[0],10])
    t_test[np.arange(t_test.shape[0]),y_test]=1

    return((x_train, t_train), (x_test, t_test))
    


model.fit(x_train,t_train, epochs=2,validation_data = (x_test,t_test))
