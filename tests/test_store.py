from pathlib import Path

import mlflow
import mlflowstone as mlflows
import pandas as pd
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV


class TestStore():

    def test_cross_validation(self):
        iris = datasets.load_iris()
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(iris.data, iris.target)

        PATH = Path(__file__).resolve().parent

        store = mlflows.Store(f'sqlite:///{PATH}/test.sqlite')

        (store
         .experiment('TestCV', Path('.').resolve())
         .start_run('Cross Validation')
         .log_cross_validation(clf, 'SVC')
         .end())

        runs = store.experiment('TestCV').last_parent_run().childs()

        assert len(runs) == 4

        artifacts = runs[0].list_artifacts(full_path=True)

        assert len(artifacts) == 4

        results = [f for f in artifacts if str(f.path).endswith('.csv')][0]
        assert pd.read_csv(results.path).shape[0] == 4

    def test_named_runs(self):
        iris = datasets.load_iris()

        svc = svm.SVC()
        svc.fit(iris.data, iris.target)
        #PATH = Path(__file__).resolve().parent
        PATH = Path('/tmp')
        store = mlflows.Store(f'sqlite:///{PATH}/test.sqlite')

        (store
            .experiment('test_named_runs', Path('.').resolve())
            .start_run('Named Runs')
            .log_model(svc, 'LOG', mlflow.sklearn)
            .end())

        run = store.experiment(
            'test_named_runs').last_run_with_name('Invalid name')
        assert run is None

        run = store.experiment(
            'test_named_runs').last_run_with_name('Named Runs')
        assert run is not None

        artifacts = run.list_artifacts(full_path=True)

        assert len(artifacts) == 3
