# Import necessary libraries
import pandas as pd
from pycaret.regression import *
from pycaret.datasets import get_data
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import docker

# Load dataset
data = get_data('insurance')
print(data)

# Initialize PyCaret experiment
exp_name = "my_experiment"
# clf = setup(data, target="charges", experiment_name=exp_name)
clf = setup(data, target='charges',
            log_experiment=True, experiment_name=exp_name, log_plots=True)

# Train models and get metrics
models = ["lr", "knn", "nb", "svm", "rf"]
metrics_list = []
for model in models:
    model = create_model(model)
    metrics = pull()
    metrics_list.append(metrics)

# Log metrics to MLflow
client = MlflowClient()
with mlflow.start_run() as run:
    mlflow.log_param("models", models)
    for i, metrics in enumerate(metrics_list):
        model_name = models[i]
        mlflow.log_metric(f"{model_name} Accuracy", metrics["Accuracy"])
        mlflow.log_metric(f"{model_name} AUC", metrics["AUC"])
    mlflow.log_artifact(run.info.artifact_uri, "pycaret_experiment")

    # Deploy model as PyFunc flavor in a Docker container
    for i, model in enumerate(models):
        saved_model_path = f"models/{run.info.run_id}/model_{model}"
        mlflow.pyfunc.save_model(path=saved_model_path, python_model=create_model(model))
        docker_client = docker.from_env()
        docker_client.images.build(path="Dockerfile", tag=f"my_model_{model}")
        docker_client.containers.run(image=f"my_model_{model}", ports={"8080": f"808{i}"}, detach=True)

print("Models deployed successfully.")
