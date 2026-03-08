from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

from tensorflow.keras.models import load_model

model = load_model("malaria_detection_model.h5", compile=False, safe_mode=False)

def predict_image(img):

    img = img.resize((64,64))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    print("Prediction:", prediction)

    if prediction[0][0] > 0.5:
        return "Uninfected Cell"
    else:
        return "Malaria Detected (Parasitized)"


@app.route("/", methods=["GET","POST"])
def index():

    result = ""

    if request.method == "POST":

        file = request.files["image"]
        img = Image.open(file)

        result = predict_image(img)

    return render_template("index.html", result=result)


if __name__ == "__main__":

    app.run(debug=True)

