# 🌳 Arboria

![CI](https://github.com/fantinsib/arboria/actions/workflows/ci.yml/badge.svg)

### A lightweight machine learning Python/C++ module for tree-based models.

This repository contains C++ implementations of tree-based models with Python bindings.

## Features

This version currently supports :

#### `DecisionTree` (binary classification)
- `.fit`: fits a tree using the CART algorithm
- `.predict`: predicts the class of passed samples


#### `RandomForest` (binary classification)
- `.fit`: fits a RandomForest with `n_estimators` trees. Uses RandomK feature selection (`max_features` features). Supports reproducible randomness via `seed`.
- `.predict` / `.predict_proba`: predicts classes or returns average votes across trees
- `.out_of_bag`: returns accuracy on training samples not bootstrapped during training

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

A demo notebook can be found [here](https://github.com/fantinsib/arboria/blob/main/docs/arboria_demo.ipynb).

````python
from arboria import DecisionTree, RandomForest
````

### Initializing the models

````python
tree = DecisionTree(max_depth = 5)
rf = RandomForest(n_estimators = 30, max_features = "sqrt", max_depth = 5, seed = 10)
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
# Returns accuracy score on the samples not bootstrapped during training :
rf.out_of_bag(x_train, y_train)
````

### Additionnal Documentation 

Additionnal [notes](docs/notes/) and [diagrams](docs/diagrams/) can be found in the [docs folder](docs/) of this repository.


## Roadmap 
 
⚠️ Still work in progress

#### Planned improvements:
- Additional model parameters (min_samples_split, min_samples_leaf, max_samples, class_weights...)
- Performance optimizations for RandomForest
- Parallelism
- Extra-Trees, Honest Trees, Quantile Trees

