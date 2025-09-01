import joblib
import numpy as np
import os

def predict_fire(temp, RH, wind, rain):
    # Load model and scaler
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
    model = joblib.load(os.path.join(base_dir, "forest_fire_model.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
    
    # Prepare sample
    sample = np.array([[temp, RH, wind, rain]])
    sample_scaled = scaler.transform(sample)
    
    # Predict
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]
    
    return prediction, probability

if __name__ == "__main__":
    # Example
    temp = 25
    RH = 40
    wind = 5
    rain = 0
    
    pred, prob = predict_fire(temp, RH, wind, rain)
    print("Prediction:", "🔥 Fire" if pred == 1 else "✅ No Fire")
    print("Fire Probability:", round(prob, 2))
