import pickle
import numpy as np

# Load previous model (if available)
try:
    with open("previous_model.pkl", "rb") as f:
        previous_model = pickle.load(f)
except FileNotFoundError:
    previous_model = None

# Load current model
with open("downloaded_xgboost.pkl", "rb") as f:
    current_model = pickle.load(f)

def test_model_consistency():
    if previous_model:
        X_test = np.array([[1, 2], [2, 3], [3, 4]])  # Example input
        prev_preds = previous_model.predict(X_test)
        curr_preds = current_model.predict(X_test)
        assert all(prev_preds == curr_preds), "New model should not introduce unexpected changes"