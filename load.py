import numpy as np
from keras import models
from tensorflow.keras.models import model_from_json
from imageio import imsave, imread, imwrite
import tensorflow as tf

def init():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')
    print('Model Loaded from Disk')

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    graph = tf.compat.v1.get_default_graph()

    return loaded_model, graph