from google.cloud import storage
from constants import (
  XGB_MODELS_DIR, 
  EXPECTED_XGB_GCP_MODEL_NAME,
  LOCAL_XGB_MODEL_PATH,
  GCP_BUCKET_NAME
)

# These imports are just for the automatic generation of requirements.txt
# That are part of the jupyter notebook which pipreqs skips
import xgboost as xgb
import shutil
import matplotlib.pyplot as plt
import seaborn as sns


def download_xgb_model():
    # TODO: Add this to environment variables
    BLOB_NAME = f'{XGB_MODELS_DIR}/{EXPECTED_XGB_GCP_MODEL_NAME}.pkl'

    client = storage.Client()
    bucket = client.bucket(GCP_BUCKET_NAME)
    blob = bucket.blob(BLOB_NAME)

    # Download Model
    blob.download_to_filename(LOCAL_XGB_MODEL_PATH)

    print(f"Model downloaded to '{LOCAL_XGB_MODEL_PATH}'")

    return LOCAL_XGB_MODEL_PATH

if __name__ == "__main__":
    download_xgb_model()
