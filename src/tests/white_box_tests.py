import pickle

# Load model
with open("downloaded_xgboost.pkl", "rb") as f:
    model = pickle.load(f)

def test_model_structure():
    assert hasattr(model, "feature_importances_"), "Model should have feature importance attribute"

def test_model_parameters():
    params = model.get_xgb_params()
    assert "max_depth" in params, "XGBoost model should have max_depth parameter"