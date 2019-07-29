# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import scipy as sp

#keras
import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenetv2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
import itertools

#matplotlib
import matplotlib.pyplot as plt

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#prediction dictionary
pred_dict = {0: 'Actinic keratoses',
             1: 'Basal cell carcinoma',
             2: 'Benign keratosis-like lesions ',
             3: 'Dermatofibroma',
             4: 'Melanocytic nevi',
             5: 'Melanoma',
             6: 'Vascular lesions'}

get_label = lambda lst: np.array([pred_dict[x] for x in lst])

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/HAM1000_best_model.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

def get_last_conv_model(model):
  last_conv_layer_name = [layer for layer in model.layers if "conv" in layer.name.lower()][-1].name
  print(f"Found last conv layer to be: {last_conv_layer_name}")
  last_conv_model      = Model(model.input, model.get_layer(last_conv_layer_name).output)
  return last_conv_model

def get_cam(img, model, last_conv_model):
  img = img[np.newaxis,:,:]
  #Obtaining class_weights
  gap_layers_inds = np.argwhere([True if "global_average_pooling2d" in layer.name.lower() else False for layer in model.layers ])
  gap_layer = model.layers[gap_layers_inds.flatten()[-1] + 1]
  gap_layer_weights = gap_layer.get_weights()[0]
  pred_probas      = model.predict(img)
  pred_class       = np.argmax(pred_probas.flatten())
  pred_class_proba = pred_probas.flatten()[pred_class]
  class_weights = gap_layer_weights[:,pred_class]

  #Extracting last conv layer
  conv_out = last_conv_model.predict(img).squeeze()
  h, w = img.shape[1]/conv_out.shape[0], img.shape[2]/conv_out.shape[1]

  conv_out = sp.ndimage.zoom(conv_out, (h, w, 1), order=1)

  return np.dot(conv_out, class_weights), pred_class, pred_class_proba

last_conv_model = get_last_conv_model(model)
last_conv_model._make_predict_function()

# make a prediction using test-time augmentation
def tta_prediction(datagen, model, x, steps=5):
  # prepare iterator

  it = datagen.flow(x=x,
                    batch_size=1,
                    shuffle=False)
  predictions = []
  for i in range(steps):
      # make predictions for each augmented image
      yhats = model.predict_generator(it, steps=it.n//it.batch_size, verbose=0)
      predictions.append(yhats)
  pred = np.mean(predictions, axis=0)
  return np.argmax(pred, axis=-1), np.max(pred)


tta_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=7,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        x_test = load_img(file_path, target_size=(224, 224))
        x_test = img_to_array(x_test)  # this is a Numpy array with shape (3, 224, 224)
        x_test = preprocess_input(x_test)
        # pred_class = tta_prediction(tta_datagen, model, x_test[np.newaxis,:,:])
        cam, pred_class, pred_class_proba = get_cam(x_test, model, last_conv_model)
        pred_class, pred_class_proba = tta_prediction(tta_datagen, model, x_test[np.newaxis,:,:], steps=5)
        f, ax = plt.subplots(1, 1)
        ax.imshow(x_test)
        ax.imshow(cam, cmap='jet', alpha=0.3)
        cam_file_path = file_path.split('.')[0]+"_cam.png"
        f.savefig(cam_file_path)
        result = pred_dict[pred_class[0]] # Convert to string
        return f"{result} with {pred_class_proba*100:.4f}% probability"
    return None


if __name__ == '__main__':
    app.run(debug=True)
