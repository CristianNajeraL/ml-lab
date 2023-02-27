# Regression example with hyperparameter tuning

## Keras

---

## How to run the experiment

- `mlflow experiments create -n individual_runs`
- `mlflow experiments create -n hyper_param_runs`
- `mlflow run -e train --experiment-id <individual_runs_experiment_id> examples/hyperparam`
- `mlflow run -e random --experiment-id <hyperparam_experiment_id> examples/hyperparam`
- `mlflow run -e hyperopt --experiment-id <hyperparam_experiment_id> examples/hyperparam`
- `mlflow ui`