"""
Compare sklearn vs arboria RandomForest performance (classification + regression)

Usage
-----
python arboria_perf.py

Notes on API matching
---------------------
Arboria RandomForest:
- __init__(n_estimators, max_features ("sqrt"/"log"/int), max_depth, max_samples, min_sample_split, seed)
- fit(X, y, criterion)  -> internally computes mtry if max_features is str
- predict / predict_proba

sklearn RandomForestClassifier / RandomForestRegressor:
- n_estimators, max_depth, max_features, max_samples, min_samples_split, random_state
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
import statistics as stats
import csv
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from arboria import RandomForestClassifier as ArboriaRandomForestClassifier
from arboria import RandomForestRegressor as ArboriaRandomForestRegressor


# helpers

def now() -> float:
    return time.perf_counter()


def median_ms(times_s: List[float]) -> float:
    return 1000.0 * stats.median(times_s)


def iqr_ms(times_s: List[float]) -> Tuple[float, float]:
    q1, q3 = np.quantile(times_s, [0.25, 0.75])
    return 1000.0 * float(q1), 1000.0 * float(q3)


def ensure_contig_f32(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)
    return X


def ensure_i32(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.int32)


def ensure_f32(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float32)


def run_repeated(fn: Callable[[], None], *, warmup: int, repeats: int) -> List[float]:
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        t0 = now()
        fn()
        t1 = now()
        times.append(t1 - t0)
    return times


@dataclass
class BenchResult:
    fit_ms_med: float
    fit_ms_q1: float
    fit_ms_q3: float
    pred_ms_med: float
    pred_ms_q1: float
    pred_ms_q3: float
    proba_ms_med: Optional[float]
    proba_ms_q1: Optional[float]
    proba_ms_q3: Optional[float]
    test_metric_name: str
    test_metric: float


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(diff * diff))


# Benchmark drivers
def bench_sklearn_rf(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, *,
    n_estimators: int,
    max_features,
    max_depth: Optional[int],
    max_samples: Optional[float],
    min_samples_split: Optional[int],
    seed: Optional[int],
    warmup: int,
    repeats: int,
) -> BenchResult:
    def make_model():
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            max_samples=max_samples,
            min_samples_split=min_samples_split if min_samples_split is not None else 2,
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )

    holder = {"m": None}

    def fit_once():
        m = make_model()
        m.fit(X_train, y_train)
        holder["m"] = m

    fit_times = run_repeated(fit_once, warmup=warmup, repeats=repeats)

    def pred_once():
        _ = holder["m"].predict(X_test)

    pred_times = run_repeated(pred_once, warmup=0, repeats=repeats)

    def proba_once():
        _ = holder["m"].predict_proba(X_test)

    proba_times = run_repeated(proba_once, warmup=0, repeats=repeats)

    m = holder["m"]
    acc = accuracy(y_test, m.predict(X_test))

    fit_q1, fit_q3 = iqr_ms(fit_times)
    pred_q1, pred_q3 = iqr_ms(pred_times)
    proba_q1, proba_q3 = iqr_ms(proba_times)

    return BenchResult(
        fit_ms_med=median_ms(fit_times), fit_ms_q1=fit_q1, fit_ms_q3=fit_q3,
        pred_ms_med=median_ms(pred_times), pred_ms_q1=pred_q1, pred_ms_q3=pred_q3,
        proba_ms_med=median_ms(proba_times), proba_ms_q1=proba_q1, proba_ms_q3=proba_q3,
        test_metric_name="accuracy",
        test_metric=acc,
    )


def bench_arboria_rf(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, *,
    n_estimators: int,
    max_features: int | str,
    max_depth: Optional[int],
    max_samples: Optional[float],
    min_sample_split: Optional[int],
    seed: Optional[int],
    criterion: str,
    warmup: int,
    repeats: int,
) -> BenchResult:
    def make_model():
        return ArboriaRandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            max_samples=max_samples,
            min_sample_split=min_sample_split,
            n_jobs=-1,
            seed=seed,
        )

    holder = {"m": None}

    def fit_once():
        m = make_model()
        m.fit(X_train, y_train, criterion=criterion)
        holder["m"] = m

    fit_times = run_repeated(fit_once, warmup=warmup, repeats=repeats)

    def pred_once():
        _ = holder["m"].predict(X_test)

    pred_times = run_repeated(pred_once, warmup=0, repeats=repeats)

    def proba_once():
        _ = holder["m"].predict_proba(X_test)

    proba_times = run_repeated(proba_once, warmup=0, repeats=repeats)

    m = holder["m"]
    acc = accuracy(y_test, m.predict(X_test))

    fit_q1, fit_q3 = iqr_ms(fit_times)
    pred_q1, pred_q3 = iqr_ms(pred_times)
    proba_q1, proba_q3 = iqr_ms(proba_times)

    return BenchResult(
        fit_ms_med=median_ms(fit_times), fit_ms_q1=fit_q1, fit_ms_q3=fit_q3,
        pred_ms_med=median_ms(pred_times), pred_ms_q1=pred_q1, pred_ms_q3=pred_q3,
        proba_ms_med=median_ms(proba_times), proba_ms_q1=proba_q1, proba_ms_q3=proba_q3,
        test_metric_name="accuracy",
        test_metric=acc,
    )

def bench_sklearn_rf_reg(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, *,
    n_estimators: int,
    max_features,
    max_depth: Optional[int],
    max_samples: Optional[float],
    min_samples_split: Optional[int],
    seed: Optional[int],
    warmup: int,
    repeats: int,
) -> BenchResult:
    def make_model():
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            max_samples=max_samples,
            min_samples_split=min_samples_split if min_samples_split is not None else 2,
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )

    holder = {"m": None}

    def fit_once():
        m = make_model()
        m.fit(X_train, y_train)
        holder["m"] = m

    fit_times = run_repeated(fit_once, warmup=warmup, repeats=repeats)

    def pred_once():
        _ = holder["m"].predict(X_test)

    pred_times = run_repeated(pred_once, warmup=0, repeats=repeats)

    m = holder["m"]
    err = mse(y_test, m.predict(X_test))

    fit_q1, fit_q3 = iqr_ms(fit_times)
    pred_q1, pred_q3 = iqr_ms(pred_times)

    return BenchResult(
        fit_ms_med=median_ms(fit_times), fit_ms_q1=fit_q1, fit_ms_q3=fit_q3,
        pred_ms_med=median_ms(pred_times), pred_ms_q1=pred_q1, pred_ms_q3=pred_q3,
        proba_ms_med=None, proba_ms_q1=None, proba_ms_q3=None,
        test_metric_name="mse",
        test_metric=err,
    )


def bench_arboria_rf_reg(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, *,
    n_estimators: int,
    max_features: int | str,
    max_depth: Optional[int],
    max_samples: Optional[float],
    min_sample_split: Optional[int],
    seed: Optional[int],
    criterion: str,
    warmup: int,
    repeats: int,
) -> BenchResult:
    def make_model():
        return ArboriaRandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            max_samples=max_samples,
            min_sample_split=min_sample_split,
            n_jobs=-1,
            seed=seed,
        )

    holder = {"m": None}

    def fit_once():
        m = make_model()
        m.fit(X_train, y_train, criterion=criterion)
        holder["m"] = m

    fit_times = run_repeated(fit_once, warmup=warmup, repeats=repeats)

    def pred_once():
        _ = holder["m"].predict(X_test)

    pred_times = run_repeated(pred_once, warmup=0, repeats=repeats)

    m = holder["m"]
    err = mse(y_test, m.predict(X_test))

    fit_q1, fit_q3 = iqr_ms(fit_times)
    pred_q1, pred_q3 = iqr_ms(pred_times)

    return BenchResult(
        fit_ms_med=median_ms(fit_times), fit_ms_q1=fit_q1, fit_ms_q3=fit_q3,
        pred_ms_med=median_ms(pred_times), pred_ms_q1=pred_q1, pred_ms_q3=pred_q3,
        proba_ms_med=None, proba_ms_q1=None, proba_ms_q3=None,
        test_metric_name="mse",
        test_metric=err,
    )


def print_block(title: str, r_sk: BenchResult, r_ar: BenchResult) -> None:
    def fmt(med, q1, q3):
        return f"{med:8.2f} ms  (IQR {q1:6.2f}–{q3:6.2f})"

    def speedup(sk, ar):
        return f"{sk / ar:6.2f}x" if ar > 0 else "n/a"

    print("\n")
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)

    print("\nFIT")
    print(f"  sklearn : {fmt(r_sk.fit_ms_med, r_sk.fit_ms_q1, r_sk.fit_ms_q3)}")
    print(f"  arboria : {fmt(r_ar.fit_ms_med, r_ar.fit_ms_q1, r_ar.fit_ms_q3)}")
    print(f"  speedup (sk/arb) : {speedup(r_sk.fit_ms_med, r_ar.fit_ms_med)}")

    print("\nPREDICT")
    print(f"  sklearn : {fmt(r_sk.pred_ms_med, r_sk.pred_ms_q1, r_sk.pred_ms_q3)}")
    print(f"  arboria : {fmt(r_ar.pred_ms_med, r_ar.pred_ms_q1, r_ar.pred_ms_q3)}")
    print(f"  speedup (sk/arb) : {speedup(r_sk.pred_ms_med, r_ar.pred_ms_med)}")

    if r_sk.proba_ms_med is not None and r_ar.proba_ms_med is not None:
        print("\nPREDICT_PROBA")
        print(f"  sklearn : {fmt(r_sk.proba_ms_med, r_sk.proba_ms_q1, r_sk.proba_ms_q3)}")
        print(f"  arboria : {fmt(r_ar.proba_ms_med, r_ar.proba_ms_q1, r_ar.proba_ms_q3)}")
        print(f"  speedup (sk/arb) : {speedup(r_sk.proba_ms_med, r_ar.proba_ms_med)}")

    print(f"\n{r_sk.test_metric_name.upper()} (test)")
    print(f"  sklearn : {r_sk.test_metric:.4f}")
    print(f"  arboria : {r_ar.test_metric:.4f}")


def main():
    # Datasets (binary)
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    reg_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    sample_sizes = [500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 200_000, 250_000]

    for n in sample_sizes:
        Xn, yn = make_classification(
            n_samples=n,
            n_features=30,
            n_informative=10,
            n_redundant=10,
            n_clusters_per_class=2,
            flip_y=0.02,
            class_sep=1.0,
            random_state=0,
        )
        datasets[f"synthetic_{n}_30f"] = (ensure_contig_f32(Xn), ensure_i32(yn))

        Xr, yr = make_regression(
            n_samples=n,
            n_features=30,
            n_informative=10,
            noise=5.0,
            random_state=0,
        )
        reg_datasets[f"reg_synthetic_{n}_30f"] = (ensure_contig_f32(Xr), ensure_f32(yr))

    seed = 10
    warmup = 1
    repeats = 10

    n_estimators = 50
    max_depth = 8
    max_samples = 0.9
    min_samples_split = 10
    min_sample_split = 10

    max_features_str = "sqrt"

    rows = []

    for name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )

        r_sk = bench_sklearn_rf(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators,
            max_features="sqrt",
            max_depth=max_depth,
            max_samples=max_samples,
            min_samples_split=min_samples_split,
            seed=seed,
            warmup=warmup,
            repeats=repeats,
        )

        r_ar = bench_arboria_rf(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators,
            max_features=max_features_str,
            max_depth=max_depth,
            max_samples=max_samples,
            min_sample_split=min_sample_split,
            seed=seed,
            criterion="gini",
            warmup=warmup,
            repeats=repeats,
        )

        print_block(
            f"Dataset: {name} | n_train={X_train.shape[0]} n_test={X_test.shape[0]} n_features={X_train.shape[1]}",
            r_sk,
            r_ar,
        )

        rows.append({
            "dataset": name,
            "lib": "sklearn",
            "task": "classification",
            "run_datetime": datetime.now().isoformat(timespec="seconds"),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth) if max_depth is not None else "",
            "max_features": "sqrt",
            "max_samples": float(max_samples) if max_samples is not None else "",
            "min_samples_split": int(min_samples_split) if min_samples_split is not None else "",
            "seed": int(seed) if seed is not None else "",
            "warmup": int(warmup),
            "repeats": int(repeats),
            "fit_ms_med": r_sk.fit_ms_med,
            "fit_ms_q1": r_sk.fit_ms_q1,
            "fit_ms_q3": r_sk.fit_ms_q3,
            "pred_ms_med": r_sk.pred_ms_med,
            "pred_ms_q1": r_sk.pred_ms_q1,
            "pred_ms_q3": r_sk.pred_ms_q3,
            "proba_ms_med": r_sk.proba_ms_med,
            "proba_ms_q1": r_sk.proba_ms_q1,
            "proba_ms_q3": r_sk.proba_ms_q3,
            "test_metric_name": r_sk.test_metric_name,
            "test_metric": r_sk.test_metric,
        })

        rows.append({
            "dataset": name,
            "lib": "arboria",
            "task": "classification",
            "run_datetime": datetime.now().isoformat(timespec="seconds"),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth) if max_depth is not None else "",
            "max_features": str(max_features_str),
            "max_samples": float(max_samples) if max_samples is not None else "",
            "min_samples_split": int(min_sample_split) if min_sample_split is not None else "",
            "seed": int(seed) if seed is not None else "",
            "warmup": int(warmup),
            "repeats": int(repeats),
            "fit_ms_med": r_ar.fit_ms_med,
            "fit_ms_q1": r_ar.fit_ms_q1,
            "fit_ms_q3": r_ar.fit_ms_q3,
            "pred_ms_med": r_ar.pred_ms_med,
            "pred_ms_q1": r_ar.pred_ms_q1,
            "pred_ms_q3": r_ar.pred_ms_q3,
            "proba_ms_med": r_ar.proba_ms_med,
            "proba_ms_q1": r_ar.proba_ms_q1,
            "proba_ms_q3": r_ar.proba_ms_q3,
            "test_metric_name": r_ar.test_metric_name,
            "test_metric": r_ar.test_metric,
        })

    for name, (X, y) in reg_datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )

        r_sk = bench_sklearn_rf_reg(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators,
            max_features="sqrt",
            max_depth=max_depth,
            max_samples=max_samples,
            min_samples_split=min_samples_split,
            seed=seed,
            warmup=warmup,
            repeats=repeats,
        )

        r_ar = bench_arboria_rf_reg(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators,
            max_features=max_features_str,
            max_depth=max_depth,
            max_samples=max_samples,
            min_sample_split=min_sample_split,
            seed=seed,
            criterion="sse",
            warmup=warmup,
            repeats=repeats,
        )

        print_block(
            f"Dataset: {name} | n_train={X_train.shape[0]} n_test={X_test.shape[0]} n_features={X_train.shape[1]}",
            r_sk,
            r_ar,
        )

        rows.append({
            "dataset": name,
            "lib": "sklearn",
            "task": "regression",
            "run_datetime": datetime.now().isoformat(timespec="seconds"),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth) if max_depth is not None else "",
            "max_features": "sqrt",
            "max_samples": float(max_samples) if max_samples is not None else "",
            "min_samples_split": int(min_samples_split) if min_samples_split is not None else "",
            "seed": int(seed) if seed is not None else "",
            "warmup": int(warmup),
            "repeats": int(repeats),
            "fit_ms_med": r_sk.fit_ms_med,
            "fit_ms_q1": r_sk.fit_ms_q1,
            "fit_ms_q3": r_sk.fit_ms_q3,
            "pred_ms_med": r_sk.pred_ms_med,
            "pred_ms_q1": r_sk.pred_ms_q1,
            "pred_ms_q3": r_sk.pred_ms_q3,
            "proba_ms_med": r_sk.proba_ms_med if r_sk.proba_ms_med is not None else "",
            "proba_ms_q1": r_sk.proba_ms_q1 if r_sk.proba_ms_q1 is not None else "",
            "proba_ms_q3": r_sk.proba_ms_q3 if r_sk.proba_ms_q3 is not None else "",
            "test_metric_name": r_sk.test_metric_name,
            "test_metric": r_sk.test_metric,
        })

        rows.append({
            "dataset": name,
            "lib": "arboria",
            "task": "regression",
            "run_datetime": datetime.now().isoformat(timespec="seconds"),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth) if max_depth is not None else "",
            "max_features": str(max_features_str),
            "max_samples": float(max_samples) if max_samples is not None else "",
            "min_samples_split": int(min_sample_split) if min_sample_split is not None else "",
            "seed": int(seed) if seed is not None else "",
            "warmup": int(warmup),
            "repeats": int(repeats),
            "fit_ms_med": r_ar.fit_ms_med,
            "fit_ms_q1": r_ar.fit_ms_q1,
            "fit_ms_q3": r_ar.fit_ms_q3,
            "pred_ms_med": r_ar.pred_ms_med,
            "pred_ms_q1": r_ar.pred_ms_q1,
            "pred_ms_q3": r_ar.pred_ms_q3,
            "proba_ms_med": r_ar.proba_ms_med if r_ar.proba_ms_med is not None else "",
            "proba_ms_q1": r_ar.proba_ms_q1 if r_ar.proba_ms_q1 is not None else "",
            "proba_ms_q3": r_ar.proba_ms_q3 if r_ar.proba_ms_q3 is not None else "",
            "test_metric_name": r_ar.test_metric_name,
            "test_metric": r_ar.test_metric,
        })

    # write recap CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"bench_results_4_tests_{timestamp}.csv"
    out_path = Path(out_name)
    suffix = 1
    while out_path.exists():
        out_path = Path(f"bench_results_4_tests_{timestamp}_{suffix}.csv")
        suffix += 1

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n")
    print(f"Wrote recap CSV: {out_path}")


if __name__ == "__main__":
    main()
