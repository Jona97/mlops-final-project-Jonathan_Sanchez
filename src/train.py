# src/train.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn

from src.data_preparation import default_config, load_data, split_xy, get_feature_groups


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eval_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_preprocess(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def get_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "lasso": Lasso(alpha=0.001, random_state=random_state),
        "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state),

        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gbr": GradientBoostingRegressor(random_state=random_state),
    }


def main():
    cfg = default_config()

    # MLflow local
    mlruns_path = cfg.project_root / "mlruns"
    mlflow.set_tracking_uri("file:" + str(mlruns_path))
    mlflow.set_experiment("insurance_charges_prediction")

    print(f"[INFO] Project root: {cfg.project_root}")
    print(f"[INFO] Data path:     {cfg.data_path}")
    print(f"[INFO] MLruns path:   {mlruns_path}")

    df = load_data(cfg)
    X, y = split_xy(df, cfg.target)

    cat_cols, num_cols = get_feature_groups(X)
    preprocess = build_preprocess(cat_cols, num_cols)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # CV setup
    cv = KFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    models = get_models(cfg.random_state)
    results = []

    best_model_name = None
    best_cv_rmse = float("inf")
    best_pipe = None

    for model_name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)

            # log parámetros relevantes (si existen)
            for p in ["n_estimators", "max_depth", "alpha", "l1_ratio", "random_state"]:
                if hasattr(model, p):
                    mlflow.log_param(p, getattr(model, p))

            # ---- CV predict (sobre TRAIN) ----
            cv_preds = cross_val_predict(pipe, X_train, y_train, cv=cv)
            cv_m = eval_metrics(y_train, cv_preds)

            mlflow.log_metric("cv_mae", cv_m["mae"])
            mlflow.log_metric("cv_rmse", cv_m["rmse"])
            mlflow.log_metric("cv_r2", cv_m["r2"])

            # ---- Fit final en train y evaluar en test ----
            pipe.fit(X_train, y_train)
            test_preds = pipe.predict(X_test)
            test_m = eval_metrics(y_test, test_preds)

            mlflow.log_metric("test_mae", test_m["mae"])
            mlflow.log_metric("test_rmse", test_m["rmse"])
            mlflow.log_metric("test_r2", test_m["r2"])

            # log model dentro del run (para inspección)
            mlflow.sklearn.log_model(pipe, name="model")

            results.append({
                "model": model_name,
                "cv_rmse": cv_m["rmse"],
                "cv_mae": cv_m["mae"],
                "cv_r2": cv_m["r2"],
                "test_rmse": test_m["rmse"],
                "test_mae": test_m["mae"],
                "test_r2": test_m["r2"],
            })

            # Tracking best
            if cv_m["rmse"] < best_cv_rmse:
                best_cv_rmse = cv_m["rmse"]
                best_model_name = model_name
                best_pipe = pipe

    # Tabla resultados
    df_results = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    print("\n[RESULTS] Sorted by CV RMSE:")
    print(df_results)

    # Guardar best_model como MLflow model en models/best_model
    if best_pipe is None:
        raise RuntimeError("No se pudo entrenar ningún modelo.")

    out_dir = cfg.project_root / "models" / "best_model"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Guardado MLflow
    mlflow.sklearn.save_model(sk_model=best_pipe, path=str(out_dir))

    print(f"\n[BEST] model={best_model_name} | best_cv_rmse={best_cv_rmse:.4f}")
    print(f"[SAVED] MLflow model exported to: {out_dir}")


if __name__ == "__main__":
    main()