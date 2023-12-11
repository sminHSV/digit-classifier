from flask import Flask, render_template, request, jsonify
import urllib
import tensorflow as tf
import numpy as np
import PIL.ImageOps

app = Flask(__name__)

model = tf.keras.models.load_model('saved_models/model.keras')
activation_model = tf.keras.models.Model(model.inputs, [layer.output for layer in model.layers])

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    image = urllib.request.urlopen(data)
    f = open('image.png', 'wb')
    f.write(image.file.read())
    f.close()
    image = tf.keras.preprocessing.image.load_img('image.png', target_size=(28, 28), color_mode='grayscale', interpolation='bilinear')
    image = PIL.ImageOps.invert(image)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.array([image])
    output = activation_model.predict(image)
    pred = np.argmax(output[-1][0])
    feature_maps_1 = output[4][0,:,:,:]
    feature_maps_2 = output[6][0,:,:,:]
    return jsonify({"result": int(pred)})