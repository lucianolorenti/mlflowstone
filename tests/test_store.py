from pathlib import Path

import pandas as pd
from mlwolf.store import MLWolflStore
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

        store = MLWolflStore(f'sqlite:///{PATH}/test.sqlite')

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
