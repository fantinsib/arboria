"""
Compare sklearn vs arboria RandomForest performance (fit / predict / predict_proba)

Usage
-----
python bench_sklearn_vs_arboria.py

Notes on API matching
---------------------
Arboria RandomForest:
- __init__(n_estimators, max_features ("sqrt"/"log"/int), max_depth, max_samples, min_sample_split, seed)
- fit(X, y, criterion)  -> internally computes mtry if max_features is str
- predict / predict_proba

sklearn RandomForestClassifier:
- n_estimators, max_depth, max_features, max_samples, min_samples_split, random_state
"""

from __future__ import annotations

import time
import statistics as stats
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from arboria import RandomForest as ArboriaRandomForest



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
    proba_ms_med: float
    proba_ms_q1: float
    proba_ms_q3: float
    test_acc: float


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())



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
        test_acc=acc,
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
        return ArboriaRandomForest(
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
        test_acc=acc,
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

    print("\nPREDICT_PROBA")
    print(f"  sklearn : {fmt(r_sk.proba_ms_med, r_sk.proba_ms_q1, r_sk.proba_ms_q3)}")
    print(f"  arboria : {fmt(r_ar.proba_ms_med, r_ar.proba_ms_q1, r_ar.proba_ms_q3)}")
    print(f"  speedup (sk/arb) : {speedup(r_sk.proba_ms_med, r_ar.proba_ms_med)}")

    print("\nACCURACY (test)")
    print(f"  sklearn : {r_sk.test_acc:.4f}")
    print(f"  arboria : {r_ar.test_acc:.4f}")


def main():
    # Datasets (binary)
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Breast cancer (small standard)
    X, y = load_breast_cancer(return_X_y=True)
    datasets["breast_cancer"] = (ensure_contig_f32(X), ensure_i32(y))

    # Synthetic 5K
    X2, y2 = make_classification(
        n_samples=5000,
        n_features=25,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=2,
        flip_y=0.02,
        class_sep=1.0,
        random_state=0,
    )
    datasets["synthetic_5k_25f"] = (ensure_contig_f32(X2), ensure_i32(y2))

    # Synthetic 50K
    X3, y3 = make_classification(
        n_samples=50000,
        n_features=30,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=2,
        flip_y=0.02,
        class_sep=1.0,
        random_state=0,
    )

    datasets["synthetic_50k_30f"] = (ensure_contig_f32(X3), ensure_i32(y3))

    # Synthetic 500K
    X4, y4 = make_classification(
        n_samples=500000,
        n_features=30,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=2,
        flip_y=0.02,
        class_sep=1.0,
        random_state=0,
    )

    datasets["synthetic_500k_30f"] = (ensure_contig_f32(X4), ensure_i32(y4))



    seed = 10
    warmup = 1
    repeats = 5

    n_estimators = 200
    max_depth = 10
    max_samples = 0.9
    min_samples_split = 10         
    min_sample_split = 10          

    max_features_str = "sqrt"

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


if __name__ == "__main__":
    main()
