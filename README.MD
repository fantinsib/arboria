# 🌳 Arboria

![CI](https://github.com/fantinsib/arboria/actions/workflows/ci.yml/badge.svg)

### A lightweight machine learning Python/C++ module for tree-based models.

This repository contains C++ implementations of tree-based models with Python bindings.

## Features

This version currently supports :

#### `DecisionTree` (binary classification)
- `.fit`: trains a decision tree using the CART algorithm
- `.predict`: predicts the labels for input samples
- Hyperparameters currently include :
    - `max_depth` : maximum depth the tree is allowed to reach
    - `min_sample_split` : minimum number of sample required in a node to allow a split


#### `RandomForest` (binary classification)
- `.fit`: fits a RandomForest with `n_estimators` trees. Uses RandomK feature selection (`max_features` features). Supports reproducible randomness via `seed`.
- `.predict` / `.predict_proba`: predicts class labels or returns the average class probabilities across all trees
- `.out_of_bag`: returns accuracy on training samples not bootstrapped during training
- Hyperparameters currently include :
    - `n_estimators` : number of trees in the forest
    - `max_depth` : maximum depth each tree is allowed to reach
    - `max_features` : number of features randomly selected at each split
    - `max_samples` : fraction of training set to bootstrap for each tree
    - `min_sample_split` : minimum number of sample required in a node to allow a split
    - `seed` : random seed 

## Installation

### Requirements
- Python >= 3.9
- A C++ compiler with C++20 support (clang / gcc)
- CMake (installed automatically by `pip` when possible)

````bash
git clone https://github.com/fantinsib/arboria.git
cd arboria
pip install .
````
Recommended tip : install in a venv 
````bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
````

## Usage

A demo notebook is available : **[arboria_demo.ipynb](https://github.com/fantinsib/arboria/blob/main/docs/arboria_demo.ipynb)**

````python
from arboria import DecisionTree, RandomForest, accuracy
````

### Initializing the models

````python
tree = DecisionTree(max_depth=5, min_sample_split=10)

rf = RandomForest(
    n_estimators= 80,
    max_depth= 10, 
    max_features=6,
    max_samples= 0.9,
    min_sample_split=10,
    seed=10
)
````

### Fit

````python
tree.fit(x_train, y_train, criterion="gini")
rf.fit(x_train, y_train, criterion = "entropy")
````

### Predict

````python
# Returns predicted classes as a np array of int
tree.predict(x_test) 
rf.predict(x_test) 

# Returns an array of the trees voting averages :
rf.predict_proba(x_test)
````

### Evaluate 
````python
# Returns the accuracy score for the predictions :
accuracy(y_pred, y_test)

# Returns accuracy score on the samples not bootstrapped during training :
rf.out_of_bag(x_train, y_train)
````

### Additionnal Documentation 

Additional [notes](docs/notes/) and [diagrams](docs/diagrams/) can be found in the [docs folder](docs/) of this repository.

## Performance Tests

The chosen benchmark is the `sklearn` algorithm for RandomForest. Test details ([history](test_python/perf/perf_log.md) and [the testing script](test_python/perf/arboria_perf.py) can be found in test_python/perf folder).

**Note** : as parallelism has not yet been implemented in arboria, comparison were made with `sklearn` `n_job` set to 1. 

### Current performance 

On larger datasets, scikit-learn remains significantly faster. On a mid-sized synthetic dataset (5k samples, 25 features), arboria reaches about **77% of scikit-learn training speed** and **~80% of inference speed**. The gap increases as dataset size grows: on the 50k / 30-feature dataset, **arboria reaches 63% of scikit-learn training speed** and is **~2.2× to ~2.9× slower** for inference.

On smaller datasets (Breast Cancer), arboria outperforms scikit-learn, with **~2× faster training** and **~5× faster inference**, largely due to scikit-learn’s higher overhead on small workloads.

---

**Breast Cancer Dataset (569 samples / 30 features)**

|               | Fit   | predict | predict_proba | accuracy  |
|---------------|-------|---------|---------------|-----------|
| arboria | 72.71 ms | 0.83 ms | 0.83 ms |  0.9357 |
| sklearn | 146.58 ms | 4.22 ms | 4.21 ms | 0.9298 |
| Sk/arb | 202% | 508% | 509% | -

---

**Synthetic 5K samples / 25 features**

|               | Fit   | predict | predict_proba | accuracy  |
|---------------|-------|---------|---------------|-----------|
| arboria | 1886.21 ms | 21.95 ms | 22.01 ms |  0.9167 |
| sklearn | 1447.22 ms | 17.93 ms | 17.78 ms | 0.9260 |
| Sk/arb  | 77% | 82% | 81% | -

---

**Synthetic 50K samples / 30 features**

|               | Fit   | predict | predict_proba | accuracy  |
|---------------|-------|---------|---------------|-----------|
| arboria | 30623.22 ms  | 342.24 ms |  448.23 ms |  0.9597 |
| sklearn | 19267.12 ms |  152.51 ms | 154.03 ms |  0.9607 |
| Sk/arb  | 63% | 45% | 34% | -




## Roadmap 
 
⚠️ Work still in progress

#### Planned improvements:
- Additional model parameters (min_samples_leaf, class_weights...)
- Performance optimizations for RandomForest
- Parallelism
- Extra-Trees, Honest Trees, Quantile Trees

