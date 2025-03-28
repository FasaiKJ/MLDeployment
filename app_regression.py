# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

with open("regression_model.pkl", "rb") as f:
    model = pickle.load(f)

feature_columns = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
    'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]

@app.route("/")
def home():
    return "Housing Price Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "input_features" not in data:
            return jsonify({"error": "'input_features' key is required"}), 400

        input_df = pd.DataFrame(data["input_features"])

        binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
        for col in binary_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].map({"yes": 1, "no": 0})

        input_df = pd.get_dummies(input_df, columns=["furnishingstatus"], drop_first=True)

        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        predictions = model.predict(input_df).tolist()

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001)
