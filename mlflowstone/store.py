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


class Experiment:
    def __init__(self, name, store: Store, models_path: Path = None):
        self.store = store
        self.client = self.store.client
        self.models_path = models_path
        experiment = self.client.get_experiment_by_name(name)
        if experiment is None:
            self.id = self.client.create_experiment(
                name, artifact_location=str(self.models_path / name))
        else:
            if self.models_path is not None:
                logger.warning(
                    'Models Path passed as parameter is ignored. Experiment already created')
            self.models_path = Path(experiment.artifact_location)
            self.id = experiment.experiment_id

    def start_run(self, name=None):
        return Run.create(name, self)

    def last_parent_run(self):
        run = self.client.search_runs(self.id,
                                      filter_string="tags.mlflow.parentRunId = '-1'",
                                      max_results=1)
        if len(run) > 0:
            run = run[0]
            return Run(run, self)
        else:
            return None

    def last_run_with_name(self, name: str):
        run = self.client.search_runs(self.id,
                                      filter_string=f"tags.mlflow.runName = '{name}'",
                                      max_results=1)
        if len(run) > 0:
            run = run[0]
            return Run(run, self)
        else:
            return None

    def run_exists(self, name: str) -> bool:
        run = self.client.search_runs(self.id,
                                      filter_string=f"tags.mlflow.runName = '{name}'",
                                      max_results=1)
        if len(run) > 0:
            return True
        else:
            return False


class Run:
    @staticmethod
    def create(name: str, experiment: Experiment, parent=None, tags={}):
        if parent is not None:
            tags[MLFLOW_PARENT_RUN_ID] = parent.run.info.run_id
        else:
            tags[MLFLOW_PARENT_RUN_ID] = -1
        if len(name) > 0:
            tags[MLFLOW_RUN_NAME] = name
        run = experiment.client.create_run(experiment.id, tags=tags)
        return Run(run, experiment, parent=parent)

    def __init__(self, run, experiment: Experiment, parent=None, tags={}):
        self.experiment = experiment
        self.client = self.experiment.client
        self.run = run
        self.id = self.run.info.run_id
        self.parent = parent

    def start_run(self, name='', tags={}):
        return Run.create(name, self.experiment, parent=self, tags=tags)

    def end(self):
        self.experiment.client.set_terminated(self.id)

    def log_cross_validation(self, gridsearch: GridSearchCV, model_name: str, tags={}, log_only_best=False):
        best = gridsearch.best_index_

        if log_only_best:
            (self.start_run(str(best))
             .log_cv_run(gridsearch, model_name, best, tags)
             .end())
        else:
            for i in range(len(gridsearch.cv_results_['params'])):
                (self.start_run(str(best))
                     .log_cv_run(gridsearch, model_name, i, tags)
                     .end())
        return self

    def _temp_file(self, filename, extension):
        tempdir = tempfile.TemporaryDirectory().name
        os.mkdir(tempdir)
        timestamp = datetime.now().isoformat().split(".")[
            0].replace(":", ".")
        filename = "%s-%s.%s" % (filename, timestamp, extension)
        return os.path.join(tempdir, filename)

    def log_pickle(self, data, filename: str):
        file_path = self._temp_file(filename, 'pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        self.client.log_artifact(self.id, file_path)

    def log_pandas(self, df: pd.DataFrame, filename):
        csv = self._temp_file(filename, 'csv')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df.to_csv(csv, index=False)

        self.client.log_artifact(self.id, csv)

    def log_cv_run(self, gridsearch: GridSearchCV, model_name: str, run_index: int, tags={}):
        """Logging of cross validation results to mlflow tracking server

        https://gist.github.com/liorshk/9dfcb4a8e744fc15650cbd4c2b0955e5

        Parameters
        -----------
                experiment_name (str): experiment name
                model_name (str): Name of the model
                run_index (int): Index of the run (in Gridsearch)
                tags (dict): Dictionary of extra data and tags (usually features)
        """

        cv_results = gridsearch.cv_results_

        self.client.log_param(self.id, "folds", gridsearch.cv)

        logger.info("Logging parameters")
        params = list(gridsearch.param_grid.keys())
        for param in params:
            self.client.log_param(self.id,
                                  param, cv_results["param_%s" % param][run_index])

        logger.info("Logging metrics")
        for score_name in [score for score in cv_results if "mean_test" in score]:
            self.client.log_metric(self.id,
                                   score_name, cv_results[score_name][run_index])
            self.client.log_metric(self.id,
                                   score_name.replace(
                                       "mean", "std"), cv_results[score_name.replace("mean", "std")][run_index])

        logger.info("Logging model")
        self.log_model(gridsearch.best_estimator_, model_name, mlflow.sklearn)

        logger.info("Logging CV results matrix")
        self.log_pandas(pd.DataFrame(cv_results), 'cv_results')

        logger.info("Logging extra data related to the experiment")
        for k, v in tags.items():
            self.client.set_tag(self.id, k, v)
        return self

    def log_model(self, model, name, flavor, **kwargs):
        with TempDir() as tmp:
            local_path = tmp.path("model")
            run_id = self.id
            mlflow_model = Model(
                run_id=self.id, artifact_path=str(self.experiment.models_path))
            flavor.save_model(model,
                              path=local_path, mlflow_model=mlflow_model, **kwargs)
            self.client.log_artifacts(self.id, local_path)
            try:
                self.client._record_logged_model(self.id, mlflow_model)
            except MlflowException:
                # We need to swallow all mlflow exceptions to maintain backwards compatibility with
                # older tracking servers. Only print out a warning for now.
                logger.warning(
                    "Logging model metadata to the tracking server has failed, possibly due older "
                    "server version. The model artifacts have been logged successfully under %s. "
                    "In addition to exporting model artifacts, MLflow clients 1.7.0 and above "
                    "attempt to record model metadata to the  tracking store. If logging to a "
                    "mlflow server via REST, consider  upgrading the server version to MLflow "
                    "1.7.0 or above.",
                    mlflow.get_artifact_uri(),
                )
        return self

    def childs(self) -> list:
        runs = self.client.search_runs(self.experiment.id,
                                       filter_string=f"tags.mlflow.parentRunId = '{self.id}'")
        return [Run(run, self.experiment) for run in runs]

    def list_artifacts(self, full_path=True) -> list:
        artifacts = self.client.list_artifacts(self.id)
        if full_path:
            return [FileInfo(self.full_path(fi), fi.is_dir, fi.file_size) for fi in artifacts]

        return artifacts

    def full_path(self, filename: Union[str, FileInfo]) -> Union[Path, str]:
        if isinstance(filename, FileInfo):
            filename = filename.path
        return self.experiment.models_path / self.id / 'artifacts' / filename
