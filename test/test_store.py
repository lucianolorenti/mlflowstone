from pathlib import Path

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

        #PATH = Path(__file__).resolve().parent
        PATH = Path('.').resolve().parent

        store = MLWolflStore(f'sqlite:////home/luciano/pirulo.sqlite')

        (store
         .experiment('TestCV', Path('.').resolve())
         .start_run('Cross Validation')
         .log_cross_validation(clf, 'SVC')
         .end())

        runs = store.experiment('TestCV').last_parent_run().childs()

        assert len(runs) == 4
