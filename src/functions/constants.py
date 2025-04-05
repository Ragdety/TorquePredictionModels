import torch.nn as nn


# Other
USER_AGENT = "csc/torque"
MAX_CSC_ROWS = 600

# Feature and label names
# Note: t will be used for feature engineering, removed after feature engineering
FEATURES = ['vEgo', 'steeringAngleDeg', 'latAccelSteeringAngle', 't', 'steerFiltered']
LABEL = 'steerFiltered'

# Training constants
TRAIN_PERCENTAGE = 0.8
BATCH_SIZE = 128
N_EPOCHS = 25
LEARNING_RATE = 0.0001
OLD_ARCH = [512, 256, 128, 64, 32]
MODEL_ARCH = [128, 64, 32]
MODEL_OUTPUT_ACTIVATIONS = {
  nn.Tanh: 'TanH',
  nn.Sigmoid: 'Sigmoid'
}
SEQUENCE_LENGTH = 40
IS_LSTM = False
TORQUE_LP_FILTER = False
USE_SCHEDULER = False
ADD_FEATURES = False

NUM_FEATURES = len(FEATURES)

if ADD_FEATURES:
    NUM_FEATURES = 6



# Model metadata
MODEL_NAME = 'torque_predictor'
MODEL_PATH = 'saved_models'

# Data storage
MIN_MAX_SCALER_TYPE = 'MinMaxScaler'
COMPOSITE_TYPE = 'Composite'
SEQUENCES_TYPE = 'Sequences'
ADDER_TYPE = 'Adder'
REMOVER_TYPE = 'Remover'
