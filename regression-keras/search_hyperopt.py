import click
import numpy as np

from hyperopt import fmin, hp, tpe, rand

import mlflow.projects
from mlflow.tracking import MlflowClient

_inf = np.finfo(np.float64).max


@click.command(
    help="Perform hyperparameter search with Hyperopt library. Optimize dl_train target."
)
@click.option("--max-runs", type=click.INT, default=10, help="Maximum number of runs to evaluate.")
@click.option("--epochs", type=click.INT, default=500, help="Number of epochs")
@click.option("--metric", type=click.STRING, default="rmse", help="Metric to optimize on.")
@click.option("--algo", type=click.STRING, default="tpe.suggest", help="Optimizer algorithm.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator")
@click.argument("training_data")
def train(training_data, max_runs, epochs, metric, algo, seed):
    # create random file to store run ids of the training tasks
    tracking_client = MlflowClient()

    def new_eval(
        nepochs, experiment_id, null_train_loss, null_valid_loss, null_test_loss, return_all=False
    ):
        def eval(params):
            import mlflow.tracking

            lr, momentum = params
            with mlflow.start_run(nested=True) as child_run:
                p = mlflow.projects.run(
                    uri=".",
                    entry_point="train",
                    run_id=child_run.info.run_id,
                    parameters={
                        "training_data": training_data,
                        "epochs": str(nepochs),
                        "learning_rate": str(lr),
                        "momentum": str(momentum),
                        "seed": seed,
                    },
                    experiment_id=experiment_id,
                    synchronous=False,  # Allow the run to fail if a model is not properly created
                )
                succeeded = p.wait()
                mlflow.log_params({"lr": lr, "momentum": momentum})

            if succeeded:
                training_run = tracking_client.get_run(p.run_id)
                metrics = training_run.data.metrics
                # cap the loss at the loss of the null model
                train_loss = min(null_train_loss, metrics["train_{}".format(metric)])
                valid_loss = min(null_valid_loss, metrics["val_{}".format(metric)])
                test_loss = min(null_test_loss, metrics["test_{}".format(metric)])
            else:
                # run failed => return null loss
                tracking_client.set_terminated(p.run_id, "FAILED")
                train_loss = null_train_loss
                valid_loss = null_valid_loss
                test_loss = null_test_loss

            mlflow.log_metrics(
                {
                    "train_{}".format(metric): train_loss,
                    "val_{}".format(metric): valid_loss,
                    "test_{}".format(metric): test_loss,
                }
            )

            if return_all:
                return train_loss, valid_loss, test_loss
            else:
                return valid_loss

        return eval

    space = [
        hp.uniform("lr", 1e-5, 1e-1),
        hp.uniform("momentum", 0.0, 1.0),
    ]

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        # Evaluate null model first.
        train_null_loss, valid_null_loss, test_null_loss = new_eval(
            0, experiment_id, _inf, _inf, _inf, True
        )(params=[0, 0])
        print('train_null_loss', train_null_loss)
        print('valid_null_loss', valid_null_loss)
        print('test_null_loss', test_null_loss)
        best = fmin(
            fn=new_eval(epochs, experiment_id, train_null_loss, valid_null_loss, test_null_loss),
            space=space,
            algo=tpe.suggest if algo == "tpe.suggest" else rand.suggest,
            max_evals=max_runs,
        )
        mlflow.set_tag("best params", str(best))
        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id], "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id)
        )
        best_val_train = _inf
        best_val_valid = _inf
        best_val_test = _inf
        best_run = None
        for r in runs:
            if r.data.metrics["val_rmse"] < best_val_valid:
                best_run = r
                best_val_train = r.data.metrics["train_rmse"]
                best_val_valid = r.data.metrics["val_rmse"]
                best_val_test = r.data.metrics["test_rmse"]
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics(
            {
                "train_{}".format(metric): best_val_train,
                "val_{}".format(metric): best_val_valid,
                "test_{}".format(metric): best_val_test,
            }
        )


if __name__ == "__main__":
    train()
