from pathlib import Path

from mlwolf.store import MLWolflStore
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)

PATH = Path(__file__).resolve().parent

store = MLWolflStore(f'sqlite:///{PATH}/db.sqlite')


(store.experiment('Test exp', Path('/home/luciano/aa').resolve() / 'models' / 'pirulo')
 .start_run('Cross Validation')
 .log_cross_validation(clf, 'SVC')
    .end())
