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
    - `n_jobs` : number of threads to launch for training; -1 uses all cores
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
    n_jobs = -1,
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

The chosen benchmark is the `sklearn` algorithm for RandomForest. Test details ([history](test_python/perf/perf_logs) and [the testing script](test_python/perf/arboria_perf.py) can be found in test_python/perf folder).

**Note** : the following test was ran with `n_jobs = -1` for both arboria and sklearn. This setting can lead to high performance variability from one run to another due to OS scheduling noise, especially on large datasets.Therefore, each test was executed four times and the reported results below correspond to the average across all.

### Current performance 

`sklearn` remains dominant on the large dataset (50k samples, 30 features), where arboria reaches **87% of sklearn training performance** on average, and significantly lower inference performance **(≈ 19–20%** of sklearn speed for predict and predict_proba). The considerably lower performance for inference is mainly due to the fact that multiprocessing for inference has not yet been implemented in arboria.

On the intermediate-sized dataset (5k samples, 25 features), arboria achieves comparable or slightly better performance than scikit-learn for both training and inference (around **10-15%** faster). 

On small datasets, arboria outperforms scikit-learn in all measured operations (**8x faster training speed, 10x faster inference**). This can be explained by the higher fixed overhead incurred by scikit-learn, which becomes dominant at small problem sizes.

---

**Breast Cancer Dataset (569 samples / 30 features)**

|               | Fit   | predict | predict_proba | accuracy  |
|---------------|-------|---------|---------------|-----------|
| sklearn | 138.605 ms | 13.7925 ms | 13.735 ms | 0.9298  |
| arboria | 16.7575 ms | 1.18 ms | 1.1825 ms | 0.9415 |
| Sk/arb | 827% | 1169% | 1162% | -

---

**Synthetic 5K samples / 25 features**

|               | Fit   | predict | predict_proba | accuracy  |
|---------------|-------|---------|---------------|-----------|
| sklearn | 420.2725 ms | 26.4475 ms | 26.135 ms | 0.9260 |
| arboria | 358.85 ms | 23.6475 ms | 22.9625 ms | 0.9213 |
| Sk/arb  | 117% | 112% | 114% | -

---

**Synthetic 50K samples / 30 features**

|               | Fit   | predict | predict_proba | accuracy  |
|---------------|-------|---------|---------------|-----------|
| sklearn | 4538.71 ms  | 54.5775 ms | 54.56 ms | 0.9607 |
| arboria | 5242.64 ms | 290.01 ms | 272.05 ms | 0.9602 |
| Sk/arb  | 87% | 19% | 20% | -



## Roadmap 
 
⚠️ Work still in progress

#### Planned improvements:
- Additional model parameters (min_samples_leaf, class_weights...)
- Performance optimizations for RandomForest
- Extra-Trees, Honest Trees, Quantile Trees

