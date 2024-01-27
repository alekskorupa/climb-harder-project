import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
from mlflow import sklearn
import logging

from src.data.load_data import load_reddit_performance_data, load_reddit_targets_data
from src.data.load_data import load_reddit_performance_data, load_reddit_targets_data, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)

TARGET_VAR = "median_grade"


def plot_predicted_vs_measured(y_true, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, c="crimson")
    p1 = max(max(y_pred), max(y_true))
    p2 = min(min(y_pred), min(y_true))
    plt.plot([p1, p2], [p1, p2], "b-")
    plt.xlabel("True Values", fontsize=15)
    plt.ylabel("Predictions", fontsize=15)
    plt.axis("equal")
    plt.show()


def main():
    with mlflow.start_run():
        X_train, X_test = load_reddit_performance_data()
        y_train, y_test = load_reddit_targets_data()

        y_train = y_train[TARGET_VAR]
        y_test = y_test[TARGET_VAR]

        # Define the pipeline
        pipeline = make_pipeline(
            KNNImputer(n_neighbors=5),
            StandardScaler(),
        )

        # Define the models and parameters to test
        models = [
            ("linearregression", LinearRegression(), {}),
            ("ridge", Ridge(), {"ridge__alpha": [0.1, 1.0, 10.0]}),
            ("lasso", Lasso(), {"lasso__alpha": [0.1, 1.0, 10.0]}),
        ]

        best_score = float("inf")
        best_model = None
        best_params = None

        for model_name, model, params in models:
            # Add the model to the pipeline
            pipeline.steps.append((model_name, model))

            # Define the grid search
            grid_search = GridSearchCV(pipeline, param_grid=params, cv=10, scoring="neg_mean_squared_error")

            # Perform the grid search
            grid_search.fit(X_train, y_train)

            # Check if this model is better than the previous best
            if -grid_search.best_score_ < best_score:
                best_score = -grid_search.best_score_
                best_model = clone(grid_search.best_estimator_)
                best_params = grid_search.best_params_

            # logging.info the best score and parameters
            logging.info(f"Best score for {model_name}: {-grid_search.best_score_:0.2f}")
            logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logging.info(" ")

            # Remove the model from the pipeline for the next iteration
            pipeline.steps.pop()

        # Train the best model on the training data
        best_model.fit(X_train, y_train)

        # Evaluate the best model on the testing data
        y_pred = best_model.predict(X_test)

        plot_predicted_vs_measured(y_test, y_pred)
        metrics = {
            "corr_test": np.corrcoef(y_test, best_model.predict(X_test))[0, 1],
            "corr_train": np.corrcoef(y_train, best_model.predict(X_train))[0, 1],
            "r2_score_test": r2_score(y_test, best_model.predict(X_test)),
            "r2_score_train": r2_score(y_train, best_model.predict(X_train)),
            "mae_score_test": mean_absolute_error(y_test, best_model.predict(X_test)),
            "mae_score_train": mean_absolute_error(y_train, best_model.predict(X_train)),
            "mse_score_test": mean_squared_error(y_test, best_model.predict(X_test)),
            "mse_score_train": mean_squared_error(y_train, best_model.predict(X_train)),
        }
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
            logging.info(f"{metric}: {value:.2f}")

        mlflow.log_param("model_params", best_params)
        mlflow.log_param("model_name", best_model[-1].__class__.__name__)

        # Train the best model on the entire data
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        best_model.fit(X, y)

        # Save the best model
        mlflow.sklearn.log_model(best_model, "model")
        joblib.dump(best_model, PROJECT_ROOT / "data/models/performance_model/model.pkl")


if __name__ == "__main__":
    main()
