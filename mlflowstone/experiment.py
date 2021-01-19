
import logging
from pathlib import Path
from typing import Optional

from mlflowstone.run import Run

logger = logging.getLogger(__name__)


class Experiment:
    def __init__(self, name, store: 'Store', models_path: Path = None):
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

    def start_run(self, name=None) -> Run:
        return Run.create(name, self)

    def last_parent_run(self) -> Optional[Run]:
        run = self.client.search_runs(self.id,
                                      filter_string="tags.mlflow.parentRunId = '-1'",
                                      max_results=1)
        if len(run) > 0:
            run = run[0]
            return Run(run, self)
        else:
            return None

    def last_run_with_name(self, name: str) -> Optional[Run]:
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
