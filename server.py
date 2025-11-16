from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# Dummy labels
labels = [
    "garbage",
    "plastic_bottles",
    "paper_waste",
    "people_washing_clothes",
    "crowd",
    "water_foam",
    "pollution",
    "dirty_water",
    "greenery",
    "normal"
]

def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = arr.reshape(1, -1)  # flatten
    return arr

@app.post("/")
def analyze():
    data = request.get_json()
    image_url = data.get("imageUrl")

    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        _ = preprocess(img)

        # Random predictions (since no real model yet)
        random_scores = np.random.rand(len(labels))
        top_indices = random_scores.argsort()[-3:][::-1]
        tags = [labels[i] for i in top_indices]

        return jsonify({"tags": tags})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
