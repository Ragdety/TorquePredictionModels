import pytest
import numpy as np
import os
import pickle
import xgboost as xgb
import pandas as pd
import time
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

# ----------- Data Preprocessor Black-box Tests -----------

def test_lag_adder_adds_correct_lags():
    df = pd.DataFrame({"steerFiltered": [1, 2, 3, 4, 5]})
    lag_dp = LagAdderDP("steerFiltered", lag=2)

    result = lag_dp(df.copy())

    # Check if new column was added
    assert "steerFilteredLag2" in result.columns
    # Check if lag values are correctly shifted
    assert pd.isna(result["steerFilteredLag2"].iloc[0])
    assert pd.isna(result["steerFilteredLag2"].iloc[1])
    assert result["steerFilteredLag2"].iloc[2] == 1
    assert result["steerFilteredLag2"].iloc[4] == 3


def test_feature_remover_drops_columns():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "t": [5, 6]
    })

    remover = FeatureRemoverDP(["t"])
    df_cleaned = remover(df)

    assert "t" not in df_cleaned.columns
    assert list(df_cleaned.columns) == ["a", "b"]


def test_composite_dp_applies_all_steps():
    df = pd.DataFrame({"x": [1, 2, 3]})

    def add_double_x(df):
        df["x2"] = df["x"] * 2
        return df

    composite = CompositeDP([
        FeatureAdderDP([add_double_x]),
        FeatureRemoverDP(["x"])
    ])

    df_out = composite(df)

    assert "x" not in df_out.columns
    assert "x2" in df_out.columns
    assert df_out["x2"].iloc[0] == 2


def test_preprocessing_pipeline_preserves_shape():
    df = pd.DataFrame({
        "steerFiltered": np.random.rand(10),
        "vEgo": np.random.rand(10),
        "steeringAngleDeg": np.random.rand(10),
        "t": np.arange(10)
    })

    pipeline = CompositeDP([
        FeatureAdderDP([
            add_steering_velocity,
            add_steering_acceleration
        ]),
        RollingMeanFeatureAdderDP(["steerFiltered"], window=3),
        FeatureRemoverDP(["t", "steerFiltered"])
    ])

    df_out = pipeline(df)

    assert "steerFiltered" not in df_out.columns
    assert "steerFilteredRollingMean" in df_out.columns
    assert "steeringVelocity" in df_out.columns
    assert df_out.shape[0] == df.shape[0]


# ----------- Model Black-box Tests -----------

# 1. Standard input
def test_standard_input_gives_valid_output(dataset, model):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    X, _ = next(iter(loader))
    X_np = X.numpy().reshape(-1, X.shape[2])
    preds = model.predict(X_np)
    
    assert preds.shape[0] == X_np.shape[0]
    assert np.all(np.isfinite(preds)), "Model output contains NaN or Inf"

# 2. Zero input
def test_zero_input_gives_reasonable_output(model):
    X_np = np.zeros((10, 9))
    preds = model.predict(X_np)
    
    assert preds.shape[0] == 10
    assert np.all(np.isfinite(preds))
    assert np.var(preds) >= 0  # Shouldn’t all be identical unless by design

# 3. Edge input
def test_edge_input_extremes(model):
    X_np = np.full((10, 9), 9999.0)
    preds = model.predict(X_np)
    
    assert np.all(np.isfinite(preds))

# 4. Random noise input
def test_random_noise_input(model):
    X_np = np.random.normal(0, 10, (10, 9))
    preds = model.predict(X_np)
    
    assert preds.shape[0] == 10
    assert np.all(np.isfinite(preds))

# 5. Incomplete input
def test_nan_input_handling_gracefully(model):
    X_full = np.random.rand(10, 9)
    X_nan = X_full.copy()
    X_nan[0, 0] = np.nan  # Introduce NaN

    preds = model.predict(X_nan)

    assert preds.shape[0] == 10
    assert np.all(np.isfinite(preds)), "Model prediction failed with NaN input"

# 6. Constant input
def test_constant_input(model):
    X_np = np.full((10, 9), 5.0)
    preds = model.predict(X_np)
    
    assert np.all(np.isfinite(preds))
    assert np.var(preds) < 1.0  # Shouldn’t be highly variable

# 7. Repeated input
def test_repeated_input_consistency(model):
    X_sample = np.random.rand(1, 9)
    X_batch = np.tile(X_sample, (10, 1))
    preds = model.predict(X_batch)
    
    assert np.allclose(preds, preds[0]), "Predictions for repeated inputs should be equal"

# 8. Prediction speed
def test_prediction_speed(model):
    X_np = np.random.rand(1000, 9)  # Larger batch size for speed test
    start_time = time.time()
    model.predict(X_np)
    elapsed_time = time.time() - start_time

    # Instrument the time taken for prediction
    print(f"Prediction time: {elapsed_time:.4f} seconds")
    
    assert elapsed_time < 0.1, "Prediction took too long"


 