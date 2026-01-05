# 🌳 Arboria

![CI](https://github.com/fantinsib/arboria/actions/workflows/ci.yml/badge.svg)

### A lightweight machine learning Python library for tree-based models.

⚠️ **Work in progress**

Arboria is a C++ project focused on implementing core decision tree algorithms from scratch, with an emphasis on clear design, explicit behavior, and solid test coverage.

The long-term goal is to make Arboria a **usable library with Python bindings**, while keeping the core logic implemented in C++.

---

## What’s implemented so far

- `DataSet` class
- Split criteria (Gini & Entropy)
- CART-style threshold generation
- Best split search with criterion, threshold and sampling parameters
- Tests
- DecisionTree training and prediction

---

## Not implemented yet

- Random / quantile threshold strategies
- Feature subsampling
- Python bindings
- Performance optimizations

---

## Notes

- APIs and internal structures may change
- The project is not yet production-ready
