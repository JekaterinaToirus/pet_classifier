import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = None


def load_model_once():
    global model

    if model is None:
        model = tf.keras.models.load_model("cats_vs_dogs.h5")


def predict_image(img_path):
    global model

    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return {
            "label": "Dog",
            "confidence": float(prediction)
        }
    else:
        return {
            "label": "Cat",
            "confidence": float(1 - prediction)
        }