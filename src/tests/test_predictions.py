import pytest
import numpy as np
import os
import pickle
import xgboost as xgb
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from data.csc_data import CSCDataset
from functions.constants import FEATURES
from functions.feature_fns import add_steering_velocity, add_steering_acceleration
from torch.utils.data import DataLoader
from data.df_preprocessing import (
    CompositeDP, FeatureAdderDP, RollingMeanFeatureAdderDP,
    LagAdderDP, FeatureRemoverDP
)


def test_predictions():
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

    dataset = CSCDataset(
        dataset="AUDI_Q3_2ND_GEN",
        features=FEATURES,
        download=True,
        train_preprocessor=preprocessor,
        label_preprocessor=None,
        is_prediction=True
    )

    model_path = "src/notebooks/xgboost_saved_models/torque_pred_xgb_model.pkl"
    assert os.path.exists(model_path), "Model file not found!"

    # rename pickle file to json
    model_path = model_path.replace('.pkl', '.json')
    os.rename(model_path, model_path)
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)


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

if __name__ == "__main__":

  test_predictions()


