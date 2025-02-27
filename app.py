from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os
from flask_cors import CORS
from OrloskyPupilDetector import process_video

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)

# Load the trained model
MODEL_PATH = "artifacts/model_stress.pickle"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)
    cls = model_data["cls"]

def inference_stress(data):
    id_ = data.get("id", "Unknown")
    records = data.get("data", [])

    if not records:
        return {"error": "No data provided"}

    df = pd.DataFrame(records)
    if "TEMP" not in df or "HR" not in df:
        return {"error": "Missing required fields: HR, TEMP"}

    TEMP = df["TEMP"].values
    if np.any((TEMP < 26) | (TEMP > 38)):
        return {"id": id_, "stress": "Temperature should be between 26 and 38"}

    X = df.drop(columns=["datetime"], errors='ignore').values
    P = cls.predict(X)
    avg_stress = np.sum(P) / (len(P) * 2)

    if avg_stress < 0.25:
        status = "Excellent"
    elif avg_stress < 0.75:
        status = "Adequate"
    else:
        status = "Needs Improvement"

    return {"id": id_, "stress": status, "average_stress": f"{avg_stress * 100:.2f}%"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/eyeCoordinates', methods=['GET'])
def get_eye_coordinates():
    video_path = "D:/MyProjects/PyCharm/EyeTracker/videos/cropped_video.mp4"  
    eye_coordinates = process_video(video_path, 1)  
    return jsonify(eye_coordinates)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        result = inference_stress(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)