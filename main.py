import os
from flask import Flask, jsonify, request
from classification_model import classify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return 'API CNN Food Classification'


@app.route("/classify", methods=['POST'])
def classify_image():
    file = request.files['image_src']
    file.save("./saveImage.jpg")
    image = Image.open("./saveImage.jpg")
    predict, value = classify(image)
    return jsonify({"predict": predict, "value": str(value)}), 200


def not_found(error):
    return jsonify({'message': str(error)}), 404


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
