import os
import pandas as pd
import pytest
import torch
import sys
sys.path.append('../')

from data.csc_data import CSCDataset
from data.df_preprocessing import (
    CompositeDP, FeatureAdderDP, RollingMeanFeatureAdderDP, 
    LagAdderDP, FeatureRemoverDP
)

# ----------- Fixtures -----------

@pytest.fixture
def mock_dataset_folder(tmp_path):
    # create a mock zip file
    folder_path = tmp_path / "mock_dataset"
    os.makedirs(folder_path, exist_ok=True)

    # Add dummy CSV to the folder
    df = pd.DataFrame({
        't': range(10),
        'steerFiltered': [0.1 * i for i in range(10)],
    })
    df.to_csv(folder_path / "file1.csv", index=False)
    
    return str(folder_path)

@pytest.fixture
def mock_dataframe(tmp_path):
    df = pd.DataFrame({
        't': range(10),
        'steerFiltered': [0.1 * i for i in range(10)],
    })
    path = tmp_path / "mock.csv"
    df.to_csv(path, index=False)
    return tmp_path, ["steerFiltered"], "steerFiltered"

@pytest.fixture
def mock_dataset_dir(tmp_path):
    df = pd.DataFrame({
        't': range(10),
        'steerFiltered': [0.1 * i for i in range(10)],
    })
    dir_path = tmp_path / "dataset"
    os.makedirs(dir_path)
    df.to_csv(dir_path / "file1.csv", index=False)
    df.to_csv(dir_path / "file2.csv", index=False)
    return dir_path

# ----------- Dataset Tests -----------

def test_dataset_length(mock_dataset_dir, mock_dataset_folder):
    dataset = CSCDataset(
        dataset=mock_dataset_folder, 
        features=["steerFiltered"],
        label="steerFiltered",
        download=False
    )
    dataset.dataset_folder_path = mock_dataset_dir
    dataset.dataframes = [pd.read_csv(f) for f in mock_dataset_dir.glob("*.csv")]
    dataset.max_rows = 10
    assert len(dataset) == 2

def test_dataset_getitem_shape(mock_dataset_dir, mock_dataset_folder):
    dataset = CSCDataset(
        dataset=mock_dataset_folder, 
        features=["steerFiltered"],
        label="steerFiltered",
        download=False
    )
    dataset.dataset_folder_path = mock_dataset_dir
    dataset.dataframes = [pd.read_csv(f) for f in mock_dataset_dir.glob("*.csv")]
    dataset.max_rows = 10

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[0] == 10
    assert y.shape[0] == 10

# ----------- Preprocessor Tests -----------
# Only the dp's that are being used at the moment

def test_rolling_mean_dp():
    df = pd.DataFrame({'steerFiltered': range(10)})
    adder = RollingMeanFeatureAdderDP(features=["steerFiltered"], window=2)
    df_transformed = adder(df.copy())
    assert "steerFilteredRollingMean" in df_transformed.columns
    assert not df_transformed["steerFilteredRollingMean"].isnull().all()

def test_lag_adder_dp():
    df = pd.DataFrame({'steerFiltered': range(10)})
    lagger = LagAdderDP("steerFiltered", lag=1)
    df_transformed = lagger(df.copy())
    assert "steerFilteredLag1" in df_transformed.columns

    # Check if the 3rd value equals the 2nd value of the original column (The actual lag)
    assert df_transformed["steerFilteredLag1"].iloc[2] == df["steerFiltered"].iloc[1]
    assert df_transformed["steerFilteredLag1"].isnull().sum() == 1

def test_composite_pipeline():
    df = pd.DataFrame({'steerFiltered': range(10)})
    pipeline = CompositeDP([
        RollingMeanFeatureAdderDP(["steerFiltered"], window=2),
        LagAdderDP("steerFiltered", lag=1),
        FeatureRemoverDP(["steerFiltered"]),
    ])
    df_out = pipeline(df.copy())
    assert "steerFilteredRollingMean" in df_out.columns
    assert "steerFiltered" not in df_out.columns

def test_feature_remover_dp():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    remover = FeatureRemoverDP(["a"])
    df_out = remover(df.copy())
    assert "a" not in df_out.columns
    assert "b" in df_out.columns

def test_feature_adder_dp():
    df = pd.DataFrame({'steerFiltered': range(10)})

    # Dummy feature column name
    dummy_feature_name = "steerFilteredMultBy2"
    feature_name_to_multiply = "steerFiltered"

    def dummy_func_mult_by_2(df, feature_to_multiply=feature_name_to_multiply):
        # Create a new feature by multiplying the original feature by 2 
        df[dummy_feature_name] = df[feature_to_multiply] * 2
        return df

    adder = FeatureAdderDP(feature_funcs=[dummy_func_mult_by_2])
    df_transformed = adder(df.copy())
    assert dummy_feature_name in df_transformed.columns
    assert df_transformed[dummy_feature_name].equals(df[feature_name_to_multiply] * 2)

