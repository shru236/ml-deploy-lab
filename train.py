import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def main():
    # Optional: set tracking server URI if you run mlflow server separately
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

    df = pd.read_csv("data/data.csv")
    X = df[["x"]]
    y = df["y"]

    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Debug: Check where mean_squared_error comes from
    print("mean_squared_error function:", mean_squared_error)

    # Calculate RMSE properly
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", float(rmse))
        # Log model artifact (mlflow will save under mlruns/ by default)
        mlflow.sklearn.log_model(model, "model")

    # Save a local copy for FastAPI to load easily
    joblib.dump(model, "model.pkl")

    print("Trained model saved as model.pkl. RMSE:", rmse)
    print("MLflow run id:", run.info.run_id)

if __name__ == '__main__':
    main()
