from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def getModel(input_dimension=4608, output_dimension=10):
    model = tf.keras.Sequential()
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    model.add(Dense(300, activation=tf.nn.relu,
                    input_shape=(input_dimension,)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation=tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation=tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(Dense(output_dimension, activation=tf.nn.softmax))
    opt = tf.keras.optimizers.Adam()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt, metrics=['sparse_categorical_accuracy'])
    return model