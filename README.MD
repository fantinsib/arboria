# 🌳 Arboria

![CI](https://github.com/fantinsib/arboria/actions/workflows/ci.yml/badge.svg)

### A lightweight machine learning Python library for tree-based models.

⚠️ **Work in progress**

Arboria is a C++ project focused on implementing core decision tree algorithms from scratch, with an emphasis on clear design, explicit behavior, and solid test coverage.

The long-term goal is to make Arboria a **usable library with Python bindings**, while keeping the core logic implemented in C++.

---

## What’s implemented so far

- `DataSet` class
- Split criteria : Gini, Entropy
- Threshold generation : midpoint splits
- Sampling methods : bootstrapping and subsampling
- Best split search with criterion, threshold and sampling parameters
- DecisionTree training and prediction
- Tests


## Not implemented yet

- Random / quantile threshold strategies
- Python bindings
- Performance optimizations


## Current projet structure

```text
arboria/
├── src/       
│   ├── dataset/         # Dataset representation and validation
│   ├── split_strategy/  # Split logic (split search, threshold, sampling)
│   ├── split_criterion/ # Impurity measures (Gini, entropy)
│   ├── node/            # Tree node structure
│   └── tree/            # Decision tree construction and prediction
│
├── tests/      # Unit tests (Catch2), organized by component
├── docs/       # Design notes
├── CMakeLists.txt
└── README.md
```

## Notes

- APIs and internal structures may change
- The project is not yet production-ready
