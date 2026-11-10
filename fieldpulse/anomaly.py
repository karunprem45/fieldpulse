"""Cross-signal consistency model.

Rule checks look at one signal at a time. This model learns how fuel rate and hydraulic
pressure normally follow the other signals, then flags stretches where a reading is
plausible on its own but does not fit what the rest of the machine is doing.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import SIGNALS

TARGETS = ["fuel_rate_lph", "hyd_pressure_bar"]


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() or 1e-9
    return (x - med) / (1.4826 * mad)


def fit_models(clean: pd.DataFrame, trim_z: float = 4.0) -> dict:
    """Train on rows the rule checks passed.

    Faults the rules cannot see (like a fuel leak) still sit in that training data. Two
    things keep them from teaching the model that they are normal: an absolute-error loss,
    which fits the median instead of the mean, and a second fit after dropping rows the
    first fit could not explain.
    """
    ok = clean[(clean["rule_flags"] == "") & ~clean["is_imputed"]].dropna(subset=SIGNALS)
    models = {}
    for target in TARGETS:
        features = [c for c in SIGNALS if c != target]
        train = ok
        for _ in range(2):
            model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, random_state=0)
            model.fit(train[features], train[target])
            resid = train[target] - model.predict(train[features])
            train = train[_robust_z(resid).abs() <= trim_z]
        models[target] = (model, features)
    return models


def score(clean: pd.DataFrame, models: dict, threshold: float = 4.0, smooth: int = 15) -> pd.DataFrame:
    out = clean.copy()
    hits = []
    for target, (model, features) in models.items():
        usable = out[features + [target]].notna().all(axis=1)
        resid = pd.Series(np.nan, index=out.index)
        resid[usable] = out.loc[usable, target] - model.predict(out.loc[usable, features])
        # A real fault lasts. A centered rolling median ignores one-off noisy seconds.
        z = _robust_z(resid).groupby(out["run_id"]).transform(
            lambda s: s.rolling(smooth, center=True, min_periods=1).median())
        hits.append(np.where(z.abs() > threshold, f"inconsistent:{target}", ""))
    out["model_flags"] = [";".join(h for h in row if h) for row in zip(*hits)]
    return out
