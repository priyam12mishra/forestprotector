from flask import Flask, request, jsonify
from predict import predict_fire

app = Flask(__name__)

@app.route('/')
def home():
    return "Forest Fire Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required_features = ["temp", "RH", "wind", "rain"]
    
    # Check if all required features are present
    if not all(feature in data for feature in required_features):
        return jsonify({"error": f"Missing features. Required: {required_features}"}), 400
    
    # Get feature values
    temp = data['temp']
    RH = data['RH']
    wind = data['wind']
    rain = data['rain']
    
    # Make prediction
    prediction, probability = predict_fire(temp, RH, wind, rain)
    
    result = {
        "prediction": "Fire" if prediction == 1 else "No Fire",
        "probability": round(probability, 2)
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
