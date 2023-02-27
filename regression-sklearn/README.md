# Regression example
## sklearn

---

## How to use me

- `cd` to the `regression` directory
- Run `mlflow run .`
- Go to `localhost:5000` and validate that a model is register
- Serve the model
  - If Windows
    - `mlflow models serve -m path/to/model/ --no-conda -p 1234`
  - MacOS or Linux
    - `mlflow models serve -m path/to/model/ -p 1234`
- Test the REST API endpoint
  - `curl -X POST -H "Content-Type:application/json" --data '{"dataframe_split": {"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],"data":[[6.2, 0.66, 0.48, 1.2, 0.029, 29, 75, 0.98, 3.33, 0.39, 12.8]]}}' http://127.0.0.1:1234/invocations`
    - The expected result is:
      - `{"predictions": [5.767793342737516]}`
