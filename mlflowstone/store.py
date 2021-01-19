from mlflowstone import Experiment
import hashlib
import json
import logging
import os
import pickle
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Union

import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.models.model import Model
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


class Store:
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri

    @property
    def client(self):
        return MlflowClient(self.tracking_uri)

    def experiment(self, name: str, models_path: Path = None):
        return Experiment(name, self, models_path)
