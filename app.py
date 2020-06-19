from flask import Flask, render_template, request
# necessities for dealing with images
from imageio import imsave, imread, imwrite
# good ol' numpy
import numpy as np
# importing models from keras
from keras import models
# for regular expressions, system-level data, operating system functions
import re, sys, os, base64

from load import *

app = Flask(__name__)
global model, graph

model, graph = init()

@app.route('/')
def index():
    return render_template('index.html')

def convert_image(img_data):
    imgstr = re.search(r'base64,(.*)',img_data).group(1)
    with open('output.png', 'wb') as f:
        f.write(base64.decodebytes(imgstr))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    image_data = request.get_data()
    convert_image(image_data)
    print('debug')
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imwrite(x, (28,28))
    x = x.reshape(1,28,28,1)
    print('debug2')
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        print('debug3')
        response = np.array_str(np.argmax(out, axis=1))
        return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True)