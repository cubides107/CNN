import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import io
import base64
import cv2

model = load_model(f"{os.path.dirname(os.path.abspath(__file__))}/models/model.h5")


def classify(img):
    img = tf.image.resize(img, [100, 100])
    #test_image = tf.keras.utils.img_to_array(img)
    test_image = np.expand_dims(img, axis=0)
    result = model.predict(test_image)
    max_value = max(result[0])
    print(max_value)
    print(result)
    if result[0][0] == max_value:
        return 'Donut', max_value
    elif result[0][1] == max_value:
        return 'Pizza', max_value
    elif result[0][2] == max_value:
        return 'Burger', max_value
    elif result[0][3] == max_value:
        return 'Potato', max_value


def base64_to_image(base64_image: str):
    # con un tamaño de 100x100
    # img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_image, "utf-8"))))
    # return img
    img = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
