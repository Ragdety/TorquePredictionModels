import pickle

from argparse import ArgumentParser
from download_model import download_xgb_model
from constants import LOCAL_XGB_MODEL_PATH


def download_model(model_path: str=LOCAL_XGB_MODEL_PATH) -> str:
   return download_model(model_path=model_path)

def load_xgb_model(model_path: str=LOCAL_XGB_MODEL_PATH) -> object:
    with open(model_path, 'rb') as file:
      model = pickle.load(file)

    return model

