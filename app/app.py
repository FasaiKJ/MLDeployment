from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Input validation
    if "features" not in data:
        return jsonify({"error": "'features' key is required"}), 400
    if not isinstance(data["features"], list):
        return jsonify({"error": "'features' must be a list"}), 400

    try:
        features = np.array(data["features"])
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != 4:
            return jsonify({"error": "Each input must have 4 values"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    predictions = model.predict(features).tolist()

    try:
        confidences = model.predict_proba(features).max(axis=1).tolist()
    except AttributeError:
        confidences = [None] * len(predictions)

    # Return confidence if single input
    if features.shape[0] == 1:
        return jsonify({
            "prediction": predictions[0],
            "confidence": confidences[0]
        })
    else:
        return jsonify({
            "predictions": predictions,
            "confidences": confidences
        })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)
