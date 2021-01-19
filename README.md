# API over MLFlow tracking API

[![Coverage Status](https://coveralls.io/repos/github/lucianolorenti/mlwolf/badge.svg?branch=main)](https://coveralls.io/github/lucianolorenti/mlwolf?branch=main)


# Mode of usage

## Saving data

```python
import mlflowstone as mlflows
store = mlflows.Store(f'sqlite:///{PATH}/test.sqlite')

run = (store.experiment('Features Experiment', data_path)
        .start_run('LASSO path')
        .log_pandas(train_dataset, 'train_dataset')
        .log_pandas(validation_dataset, 'validation_dataset')
        )

X, y, sw = get_data(train_dataset)
alphas_lasso, coefs_lasso, _ = sklearn.linear_model.lasso_path(X, y)

run.log_pickle((alphas_lasso, coefs_lasso), 'lasso_path')
run.end()
```

## Loading data
```python
run = (store.experiment('Features Experiment')
            .last_run_with_name('LASSO path'))
artifacts = run.list_artifacts()
```