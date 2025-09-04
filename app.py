from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
from PIL import Image
import io

from trainer import make_predictions
from recognizer import preprocess

app = Flask(__name__, template_folder="templates")
CORS(app)  # allow frontend requests (from same or different host)

# Load trained parameters
with open("model_params.pkl", "rb") as f:
    W1, b1, W2, b2 = pickle.load(f)

@app.route("/")
def home():
    # serve your frontend
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"].read()
    img = Image.open(io.BytesIO(file)).convert("L")

    processed = preprocess(img)
    prediction = make_predictions(processed, W1, b1, W2, b2)

    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    # host="0.0.0.0" → lets you access from phone on same WiFi
    app.run(debug=True, host="0.0.0.0", port=5000)
