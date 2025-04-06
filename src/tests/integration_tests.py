import pytest
import numpy as np
import os
import pickle
import xgboost as xgb
import sys
sys.path.append('../')

from data.csc_data import CSCDataset
from functions.constants import FEATURES
from functions.feature_fns import add_steering_velocity, add_steering_acceleration
from torch.utils.data import DataLoader
from data.df_preprocessing import (
    CompositeDP, FeatureAdderDP, RollingMeanFeatureAdderDP,
    LagAdderDP, FeatureRemoverDP
)


@pytest.fixture(scope="module")
def dataset():
    # Use same preprocessing as training
    preprocessor = CompositeDP([
        FeatureAdderDP([
            add_steering_velocity,
            add_steering_acceleration
        ]),
        RollingMeanFeatureAdderDP(["steerFiltered"], window=5),
        LagAdderDP("steerFiltered", lag=1),
        LagAdderDP("steerFiltered", lag=2),
        LagAdderDP("steerFiltered", lag=3),
        FeatureRemoverDP(['t']),
        FeatureRemoverDP(['steerFiltered'])
    ])
    return CSCDataset(
        dataset="AUDI_Q3_2ND_GEN",
        features=FEATURES,
        download=True,
        train_preprocessor=preprocessor,
        label_preprocessor=None,
        is_prediction=True
    )

@pytest.fixture(scope="module")
def model():
    model_path = "src/notebooks/xgboost_saved_models/torque_pred_xgb_model.pkl"
    assert os.path.exists(model_path), "Model file not found!"

    # rename pickle file to json
    model_path = model_path.replace('.pkl', '.json')
    os.rename(model_path, model_path)
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

def test_model_predicts_shape_correct(dataset, model):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    for X, _ in loader:
        X_np = X.numpy()

        # shape: (batch_size * 600, 9)
        X_np = X_np.reshape(-1, X_np.shape[2])
        preds = model.predict(X_np)

        assert isinstance(preds, np.ndarray), "Prediction should return NumPy array"
        # print(len(preds), X.shape[0])
        
        # Check the shape of the predictions
        assert len(preds) == X_np.shape[0], "Prediction length should match number of samples"
        assert not np.isnan(preds).any(), "Predictions contain NaN values"

        break  # single batch

def test_prediction_values_are_finite(dataset, model):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for X, _ in loader:
        X_np = X.numpy()

        # shape: (batch_size * 600, 9)
        X_np = X_np.reshape(-1, X_np.shape[2])
        preds = model.predict(X_np)

        assert np.isfinite(preds).all(), "All prediction values should be finite"
        break
    
def test_prediction_distribution_reasonable(dataset, model):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for X, _ in loader:
        X_np = X.numpy().reshape(-1, X.shape[2])
        preds = model.predict(X_np)

        mean = np.mean(preds)
        std_dev = np.std(preds)

        assert -500 < mean < 500, f"Prediction mean {mean} seems off"
        assert std_dev < 300, f"Prediction std deviation {std_dev} is too large"
        break
    
def test_multiple_batches_consistency(dataset, model):
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    batch_count = 0
    for X, _ in loader:
        X_np = X.numpy().reshape(-1, X.shape[2])
        preds = model.predict(X_np)

        assert isinstance(preds, np.ndarray), "Prediction should return NumPy array"
        assert preds.shape[0] == X_np.shape[0], "Mismatch in prediction and input shape"
        assert np.isfinite(preds).all(), "Non-finite values found in predictions"
        batch_count += 1

        if batch_count >= 3:
            break
        

def test_model_with_incomplete_batch(dataset, model):
    batch_size = len(dataset) // 4 + 1  # Ensure incomplete final batch
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for X, _ in loader:
        X_np = X.numpy().reshape(-1, X.shape[2])
        preds = model.predict(X_np)

        assert preds.shape[0] == X_np.shape[0], "Prediction shape mismatch on incomplete batch"
        break
    
def test_prediction_no_zero_variance(dataset, model):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for X, _ in loader:
        X_np = X.numpy().reshape(-1, X.shape[2])
        preds = model.predict(X_np)

        std = np.std(preds)
        assert std > 1e-3, "Predictions have near-zero variance, might be constant output"
        break
