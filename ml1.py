# https://developers.google.com/codelabs/tensorflow-1-helloworld
from datetime import datetime as dt
d1 = dt.now()
import tensorflow as tf
import numpy as np
from tensorflow import keras
d2 = dt.now()

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
ys = np.array([-2, 1, 4, 7, 10, 13], dtype=float)
d3 = dt.now()

model.fit(xs, ys, epochs=100)
print(model.predict([10]))
d4 = dt.now()

print((d2-d1).total_seconds())
print((d3-d2).total_seconds())
print((d4-d3).total_seconds())
print((d4-d1).total_seconds())
