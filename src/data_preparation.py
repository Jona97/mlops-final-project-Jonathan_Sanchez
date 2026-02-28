# src/data_preparation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


@dataclass
class Config:
    project_root: Path
    data_path: Path
    target: str = "charges"
    test_size: float = 0.2
    random_state: int = 42


def default_config(project_root: str | Path | None = None) -> Config:
    """
    Define rutas y parámetros por defecto.
    Espera el dataset en: <project_root>/data/insurance/insurance.csv
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    data_path = root / "data" / "insurance" / "insurance.csv"
    return Config(project_root=root, data_path=data_path)


def load_data(cfg: Config) -> pd.DataFrame:
    if not cfg.data_path.exists():
        raise FileNotFoundError(
            f"No encuentro el dataset en: {cfg.data_path}\n"
            f"Verifica que exista: data/insurance/insurance.csv"
        )
    df = pd.read_csv(cfg.data_path)

    # Normalizar columnas (por si vienen con espacios)
    df.columns = [c.strip() for c in df.columns]

    # Validación mínima
    if cfg.target not in df.columns:
        raise ValueError(f"Target '{cfg.target}' no existe. Columnas: {list(df.columns)}")

    return df


def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def get_feature_groups(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Devuelve (cat_cols, num_cols) basado en dtype.
    """
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols