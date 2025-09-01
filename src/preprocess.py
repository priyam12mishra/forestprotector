import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath=None):
    """
    Load forest fires dataset
    """
    if filepath is None:
        # Automatically find dataset relative to this file
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "forestfires.csv")
    
    df = pd.read_csv(filepath)
    
    # Create binary target: Fire (1) if area > 0 else No Fire (0)
    df['fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)
    
    # Features to use
    features = ["temp", "RH", "wind", "rain"]
    X = df[features]
    y = df['fire']
    
    return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Split and scale data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
