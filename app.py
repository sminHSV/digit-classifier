from flask import Flask, render_template, request, jsonify, send_from_directory
import urllib
import tensorflow as tf
import numpy as np
from PIL import ImageOps
from PIL import Image
import io
import base64
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

SWAGGER_URL = '/doc'
API_URL = '/spec'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Digit Classifier"
    }
)
app.register_blueprint(swaggerui_blueprint)

model = tf.keras.models.load_model('saved_models/model.keras')
activation_model = tf.keras.models.Model(model.inputs, [layer.output for layer in model.layers])

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/spec')
def spec():
    return send_from_directory(app.root_path, 'swagger.json')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive request data
    data = request.get_data()
    # Open image
    im = Image.open(io.BytesIO(data))
    # Convert to grayscale
    im = im.convert('L') 
    # Invert image
    im = ImageOps.invert(im)
    # Crop image
    bbox = im.getbbox()
    if bbox is not None:
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        left = bbox[0] + (bbox[2] - bbox[0] - size) // 2
        upper = bbox[1] + (bbox[3] - bbox[1] - size) // 2
        right = left + size
        bottom = upper + size
        im = im.crop((left, upper, right, bottom))
    # Resize image
    im = im.resize((28, 28), resample=Image.Resampling.BICUBIC)
    # Convert to numpy array
    im = np.asarray(im, dtype=np.float32)
    # Normalize image
    im /= 255.0

    # Make prediction
    output = activation_model.predict(np.array([im]))
    pred = np.argmax(output[-1][0])

    # Get feature maps
    input = output[0][0]
    fm_1 = output[4][0]
    fm_2 = output[6][0]
    
    # Channel First
    input = np.transpose(input, (2, 0, 1))
    fm_1 = np.transpose(fm_1, (2, 0, 1))
    fm_2 = np.transpose(fm_2, (2, 0, 1))

    # Encode images as base64 Data URLs
    def encode_image_array(array):
        im = tf.keras.preprocessing.image.array_to_img(
            [array], data_format='channels_first')
        buffer = io.BytesIO()
        im.save(buffer, format='PNG')
        string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        string = 'data:im/png;base64,' + string
        return string

    return render_template(
        'results.html',
        pred=pred,
        input=encode_image_array(input[0]),
        layer_1=[encode_image_array(im) for im in fm_1],
        layer_2=[encode_image_array(im) for im in fm_2]
    )

print(app.url_map)