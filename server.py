from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/tf_model")
labels = open("model/labels.txt").read().splitlines()

def preprocess(img):
    img = img.resize((224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.expand_dims(arr, 0)
    return arr / 255.0

@app.post("/")
def analyze():
    data = request.get_json()
    image_url = data.get("imageUrl")
    
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    
    arr = preprocess(img)
    preds = model.predict(arr)[0]
    
    top_indices = preds.argsort()[-3:][::-1]
    tags = [labels[i] for i in top_indices]

    return jsonify({"tags": tags})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
