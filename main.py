import os
from flask import Flask, jsonify, request
from classification_model import classify
from flask_cors import CORS
import cv2
from PIL import Image

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return 'API CNN Food Classification'


@app.route("/classify", methods=['POST'])
def classify_image():
    f = request.files['image_src']
    f.save("./saveImage.jpg")
    #request_data = request.json
    #if request_data is None:
       # return '{"message": "Invalid Request"}'
   # base64_image = request_data.get("image_src")
    #if base64_image is None:
       # return '{"message": "Send a image please!"}'
    #image = cv2.imread("./saveImage.jpg", cv2.IMREAD_COLOR)
    im = Image.open("./saveImage.jpg")

    predict, value = classify(im)
    # return jsonify({"predict": predict, "value": str(value)}), 200
    return jsonify({"result": predict}), 200


def not_found(error):
    return jsonify({'message': str(error)}), 404


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
