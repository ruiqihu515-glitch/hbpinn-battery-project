"""Reusable physics-inspired hybrid voltage models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


BASELINE_FEATURES = [
    "soc",
    "soc_squared",
    "soc_cubed",
    "discharge_current_a",
]

RESIDUAL_FEATURES = [
    "soc",
    "soc_squared",
    "soc_cubed",
    "discharge_current_a",
    "used_capacity_ah",
    "cumulative_energy_kwh",
    "discharge_index",
]

TEMPERATURE_RESIDUAL_FEATURES = RESIDUAL_FEATURES + ["temperature_c"]


@dataclass
class HybridVoltageModel:
    """Fitted baseline and RF residual models for hybrid voltage prediction."""

    baseline_model: LinearRegression
    residual_model: RandomForestRegressor
    include_temperature: bool
    baseline_feature_names: list[str]
    residual_feature_names: list[str]


def build_voltage_features(
    table: pd.DataFrame,
    include_temperature: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Example-16 baseline and residual feature matrices in row order.

    ``include_temperature=False`` reproduces the selected RF hybrid model.
    ``include_temperature=True`` builds the temperature-aware RF variant.
    """
    feature_table = table.copy()
    feature_table["soc"] = (
        1.0 - feature_table["used_capacity_ah"] / feature_table["capacity_ah"]
    )
    feature_table["soc"] = feature_table["soc"].clip(lower=0.0, upper=1.0)
    feature_table["soc_squared"] = feature_table["soc"] ** 2
    feature_table["soc_cubed"] = feature_table["soc"] ** 3
    feature_table["discharge_current_a"] = (-feature_table["current_a"]).clip(
        lower=0.0
    )

    residual_features = (
        TEMPERATURE_RESIDUAL_FEATURES if include_temperature else RESIDUAL_FEATURES
    )
    return feature_table[BASELINE_FEATURES], feature_table[residual_features]


def split_discharge_cycles(
    table: pd.DataFrame,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete discharge cycles chronologically, as in Example 16."""
    all_cycles = sorted(table["discharge_index"].unique())
    split_index = int(train_fraction * len(all_cycles))
    train_cycles = all_cycles[:split_index]
    test_cycles = all_cycles[split_index:]

    train_table = table[table["discharge_index"].isin(train_cycles)].copy()
    test_table = table[table["discharge_index"].isin(test_cycles)].copy()
    return train_table, test_table


def fit_voltage_baseline(X, y) -> LinearRegression:
    """Fit the physics-inspired linear voltage baseline from Example 16."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def fit_rf_residual_model(X, residual) -> RandomForestRegressor:
    """Fit the Example-16 Random Forest voltage-residual model."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, residual)
    return model


def fit_hybrid_voltage_model(
    train_table: pd.DataFrame,
    include_temperature: bool = False,
) -> HybridVoltageModel:
    """Fit the selected RF hybrid or its temperature-aware RF variant."""
    baseline_features, residual_features = build_voltage_features(
        train_table,
        include_temperature=include_temperature,
    )
    baseline_model = fit_voltage_baseline(
        baseline_features,
        train_table["voltage_v"],
    )
    baseline_prediction = baseline_model.predict(baseline_features)
    residual = train_table["voltage_v"] - baseline_prediction
    residual_model = fit_rf_residual_model(residual_features, residual)
    residual_feature_names = (
        TEMPERATURE_RESIDUAL_FEATURES if include_temperature else RESIDUAL_FEATURES
    )
    return HybridVoltageModel(
        baseline_model=baseline_model,
        residual_model=residual_model,
        include_temperature=include_temperature,
        baseline_feature_names=list(BASELINE_FEATURES),
        residual_feature_names=list(residual_feature_names),
    )


def predict_hybrid_voltage(
    model: HybridVoltageModel,
    table: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Predict baseline, RF residual, and their hybrid voltage sum."""
    baseline_features, residual_features = build_voltage_features(
        table,
        include_temperature=model.include_temperature,
    )
    baseline_prediction = model.baseline_model.predict(
        baseline_features[model.baseline_feature_names]
    )
    residual_prediction = model.residual_model.predict(
        residual_features[model.residual_feature_names]
    )
    hybrid_prediction = baseline_prediction + residual_prediction
    return {
        "baseline_prediction": np.asarray(baseline_prediction),
        "residual_prediction": np.asarray(residual_prediction),
        "hybrid_prediction": np.asarray(hybrid_prediction),
    }
