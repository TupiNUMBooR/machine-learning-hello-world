# https://developers.google.com/codelabs/tensorflow-2-computervision
from datetime import datetime as dt

d1 = dt.now()

import tensorflow as tf
import numpy as np
from tensorflow import keras

d2 = dt.now()
print("imported in", (d2 - d1).total_seconds())


def look(label, image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    print(label)
    print(image)
    plt.show()


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        acc=0.98
        if logs.get('accuracy') > acc:
            print(f"Reached {acc}% accuracy so cancelling training!")
            self.model.stop_training = True


mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# look(test_labels[0], test_images[0])
training_images = training_images / 255
test_images = test_images / 255

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
d3 = dt.now()
print("model compiled in", (d3 - d2).total_seconds())

model.fit(training_images, training_labels, epochs=300, callbacks=[myCallback()])
d4 = dt.now()
print("model fit in", (d4 - d3).total_seconds())

model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print(classifications[0])
d5 = dt.now()
print("model tested in", (d5 - d4).total_seconds())
