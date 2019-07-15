# Classifier
from keras.models import Sequential
from keras.layers import *


def getClassifierModel(input_dimension=4608, output_dimension=10,name = "Audo_DNN"):
    model = Sequential()
    model.add(Dense(300, activation='relu',
                    input_shape=(input_dimension,)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dimension, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['sparse_categorical_accuracy'])
    model.name = name
    return model
