import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import load_data, preprocess_data

def train_and_save_model():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Ensure saved_models folder exists
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    # Save model and scaler
    joblib.dump(model, os.path.join(save_dir, "forest_fire_model.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
