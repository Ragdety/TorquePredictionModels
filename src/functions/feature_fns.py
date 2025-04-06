import numpy as np

from scipy.signal import savgol_filter


def savitzky_golay_filter(data, window_size=5, polynomial_order=3):
    """Applies a Savitzky-Golay filter to the data."""
    return savgol_filter(data, window_size, polynomial_order)

def differentiate(data, time):
    """Differentiates data with respect to time."""
    return np.gradient(data, time)

def add_steering_velocity(df):
    """Adds a new feature to the dataframe: steering velocity."""
    
    df = df.copy()
    
   #  assert (df['t'].diff() > 0).all(), "Timestamps are not strictly increasing!"

    # df.loc[:, 'steeringVelocity'] = differentiate(df['steeringAngleDeg'], df['t'])
    df['steeringVelocity'] = df['steeringAngleDeg'].diff() / (df['t'].diff() + 1e-8)

    # TODO: Add/explore preprocessing for steering velocity
    # Replace inf/-inf with the maximum/minimum finite values
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    max_finite = df['steeringVelocity'][np.isfinite(df['steeringVelocity'])].max()
    min_finite = df['steeringVelocity'][np.isfinite(df['steeringVelocity'])].min()
    df['steeringVelocity'] = np.clip(df['steeringVelocity'], min_finite, max_finite)

    # df['steeringVelocity'] = df['steeringVelocity'].rolling(window=5, min_periods=1).mean()
    df.fillna(0, inplace=True)

    # Apply filter to remove noise
    df['steeringVelocity'] = savitzky_golay_filter(df['steeringVelocity'], 5, 2)
    # df['steeringVelocity'] = moving_average(df['steeringVelocity'], 1)

    return df

def add_steering_acceleration(df):
    """Adds a new feature to the dataframe: steering acceleration."""

    df = df.copy()

    # assert (df['t'].diff() > 0).all(), "Timestamps are not strictly increasing!"

    # df.loc[:, 'steeringAcceleration'] = differentiate(df['steeringVelocity'], df['t'])
    df['steeringAcceleration'] = df['steeringVelocity'].diff() / (df['t'].diff() + 1e-8)

    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    max_finite = df['steeringAcceleration'][np.isfinite(df['steeringAcceleration'])].max()
    min_finite = df['steeringVelocity'][np.isfinite(df['steeringAcceleration'])].min()
    df['steeringAcceleration'] = np.clip(df['steeringAcceleration'], min_finite, max_finite)

    # df['steeringAcceleration'] = df['steeringAcceleration'].rolling(window=5, min_periods=1).mean()
    df.fillna(0, inplace=True)

    # Apply filter to remove noise
    df['steeringAcceleration'] = savitzky_golay_filter(df['steeringAcceleration'], 5, 2)
    # df['steeringVelocity'] = moving_average(df['steeringVelocity'], 1)

    return df

def add_ema(df, feature, window=5):
    """Adds an Exponential Moving Average (EMA) to the dataframe."""
    
    df = df.copy()
    
    df[f'{feature}_ema'] = df[feature].ewm(span=window, adjust=False).mean()
    
    return df

