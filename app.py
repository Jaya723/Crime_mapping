from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    best_model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    with open('kmeans_model.pkl', 'rb') as kmeans_file:
        kmeans_model = pickle.load(kmeans_file)
    print("All models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: A required model file was not found: {e}")
    best_model = None
    scaler = None
    encoders = None
    kmeans_model = None

API_KEY = os.environ.get("ML_API_KEY", "supersecret")
app = Flask(__name__)

CORS(app, resources={
     r"/predict": {"origins": "http://localhost:3000", "methods": ["POST"]}})


def predict_risk(state: str, district: str, year: int, latitude: float, longitude: float):
    """
    Predicts the crime risk level for a new, unseen location and year.
    This function uses the saved models and doesn't rely on historical data lookup.
    """
    if not best_model or not scaler or not encoders or not kmeans_model:
        return {"error": "Prediction models are not loaded. Server may be misconfigured."}

    try:
        state_enc = encoders['State'].transform([state])[0]
        district_enc = encoders['District'].transform([district])[0]

        new_location_data = pd.DataFrame(
            [{'Latitude': latitude, 'Longitude': longitude}])
        geo_cluster = kmeans_model.predict(new_location_data)[0]

        input_data = pd.DataFrame({
            'Year': [year],
            'State_Enc': [state_enc],
            'District_Enc': [district_enc],
            'Geo_Cluster': [geo_cluster]
        })

        scaled_input_data = pd.DataFrame(scaler.transform(
            input_data), columns=input_data.columns)

        pred_class = best_model.predict(scaled_input_data)[0]
        proba = best_model.predict_proba(scaled_input_data)[0]

        classes = best_model.classes_
        prob_dict = {str(cls): float(proba[i])
                     for i, cls in enumerate(classes)}

        return {
            "State": state,
            "District": district,
            "Year": year,
            "Latitude": latitude,
            "Longitude": longitude,
            "Predicted Risk Level": str(pred_class),
            "Probabilities": prob_dict
        }
    except ValueError as e:
        return {"error": f"Encoding failed for State or District. New values must be in the training set: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during prediction: {str(e)}"}


@app.route("/predict", methods=["POST"])
def predict():
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    state = data.get("state")
    district = data.get("district")
    year = data.get("year")
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not all([state, district, year, latitude, longitude]):
        return jsonify({"error": "Missing required fields: state, district, year, latitude, longitude"}), 400

    try:
        year = int(year)
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        return jsonify({"error": "Year, latitude, and longitude must be numbers"}), 400

    result = predict_risk(state, district, year, latitude, longitude)
    return jsonify(result)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crime Prediction API is running!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
