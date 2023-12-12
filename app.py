from flask import Flask, render_template, request, jsonify
import urllib
import tensorflow as tf
import numpy as np
import PIL.ImageOps
from PIL import Image
import io
import base64

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

    input = output[0][0][:, :, 0]
    feature_maps_1 = output[4][0]
    feature_maps_2 = output[6][0]

    feature_maps_1 = np.transpose(feature_maps_1, (2, 0, 1))
    feature_maps_2 = np.transpose(feature_maps_2, (2, 0, 1))

    def encode_image(array):
        image = tf.keras.preprocessing.image.array_to_img([array], data_format='channels_first')
        image.save('image.png')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        string = 'data:image/png;base64,' + string
        return string
    
    images_1 = []
    images_2 = []
    for i in range(6):
        images_1.append(encode_image(feature_maps_1[i]))

    for i in range(16):
        images_2.append(encode_image(feature_maps_2[i]))

    input = encode_image(input)

    return jsonify({
        "result": int(pred),
        "input": input,
        "feature_maps_1": images_1,
        "feature_maps_2": images_2
    })