
---
26/01/2025 (Multithreading for inference)
---

This test was ran with parameters :
```python
n_estimators = 100	
max_depth = 8
max_features = "sqrt"
max_samples = 0.9
min_samples_split = 10
```
for sklearn and arboria with 1 warmup and 10 repeats.

```bash
======================================================================================
Dataset: synthetic_500_30f | n_train=350 n_test=150 n_features=30
======================================================================================

FIT
  sklearn :    78.70 ms  (IQR  75.00– 87.71)
  arboria :    10.85 ms  (IQR  10.21– 10.91)
  speedup (sk/arb) :   7.25x

PREDICT
  sklearn :    13.74 ms  (IQR  12.33– 14.07)
  arboria :     0.26 ms  (IQR   0.26–  0.27)
  speedup (sk/arb) :  52.55x

PREDICT_PROBA
  sklearn :    13.72 ms  (IQR  12.36– 13.84)
  arboria :     0.25 ms  (IQR   0.24–  0.25)
  speedup (sk/arb) :  55.67x

ACCURACY (test)
  sklearn : 0.8600
  arboria : 0.8400



======================================================================================
Dataset: synthetic_1000_30f | n_train=700 n_test=300 n_features=30
======================================================================================

FIT
  sklearn :    98.48 ms  (IQR  91.10–104.05)
  arboria :    27.34 ms  (IQR  25.99– 29.98)
  speedup (sk/arb) :   3.60x

PREDICT
  sklearn :    13.49 ms  (IQR  12.92– 13.73)
  arboria :     0.41 ms  (IQR   0.40–  0.41)
  speedup (sk/arb) :  32.68x

PREDICT_PROBA
  sklearn :    13.82 ms  (IQR  13.78– 13.89)
  arboria :     0.41 ms  (IQR   0.39–  0.41)
  speedup (sk/arb) :  34.06x

ACCURACY (test)
  sklearn : 0.8700
  arboria : 0.8533



======================================================================================
Dataset: synthetic_2500_30f | n_train=1750 n_test=750 n_features=30
======================================================================================

FIT
  sklearn :   137.09 ms  (IQR 131.37–139.56)
  arboria :    76.67 ms  (IQR  76.22– 77.80)
  speedup (sk/arb) :   1.79x

PREDICT
  sklearn :    13.69 ms  (IQR  13.41– 13.90)
  arboria :     1.02 ms  (IQR   0.91–  1.03)
  speedup (sk/arb) :  13.45x

PREDICT_PROBA
  sklearn :    13.73 ms  (IQR  13.67– 13.81)
  arboria :     0.97 ms  (IQR   0.96–  0.98)
  speedup (sk/arb) :  14.21x

ACCURACY (test)
  sklearn : 0.9160
  arboria : 0.9013



======================================================================================
Dataset: synthetic_5000_30f | n_train=3500 n_test=1500 n_features=30
======================================================================================

FIT
  sklearn :   219.91 ms  (IQR 214.06–226.45)
  arboria :   192.86 ms  (IQR 187.09–201.64)
  speedup (sk/arb) :   1.14x

PREDICT
  sklearn :    13.98 ms  (IQR  13.28– 14.31)
  arboria :     2.21 ms  (IQR   2.03–  2.55)
  speedup (sk/arb) :   6.33x

PREDICT_PROBA
  sklearn :    14.09 ms  (IQR  13.87– 14.13)
  arboria :     2.16 ms  (IQR   2.06–  2.36)
  speedup (sk/arb) :   6.54x

ACCURACY (test)
  sklearn : 0.8927
  arboria : 0.8867



======================================================================================
Dataset: synthetic_10000_30f | n_train=7000 n_test=3000 n_features=30
======================================================================================

FIT
  sklearn :   395.82 ms  (IQR 383.16–406.30)
  arboria :   343.26 ms  (IQR 341.67–354.23)
  speedup (sk/arb) :   1.15x

PREDICT
  sklearn :    13.66 ms  (IQR  13.25– 25.69)
  arboria :     3.32 ms  (IQR   3.31–  3.40)
  speedup (sk/arb) :   4.11x

PREDICT_PROBA
  sklearn :    13.87 ms  (IQR  13.72– 14.40)
  arboria :     3.36 ms  (IQR   3.32–  3.40)
  speedup (sk/arb) :   4.13x

ACCURACY (test)
  sklearn : 0.9267
  arboria : 0.9277



======================================================================================
Dataset: synthetic_25000_30f | n_train=17500 n_test=7500 n_features=30
======================================================================================

FIT
  sklearn :   865.80 ms  (IQR 854.88–887.77)
  arboria :   978.34 ms  (IQR 964.57–983.19)
  speedup (sk/arb) :   0.88x

PREDICT
  sklearn :    26.50 ms  (IQR  26.31– 26.59)
  arboria :     6.70 ms  (IQR   6.61–  7.01)
  speedup (sk/arb) :   3.96x

PREDICT_PROBA
  sklearn :    26.50 ms  (IQR  26.40– 26.67)
  arboria :     6.68 ms  (IQR   6.63–  6.69)
  speedup (sk/arb) :   3.97x

ACCURACY (test)
  sklearn : 0.9105
  arboria : 0.9031



======================================================================================
Dataset: synthetic_50000_30f | n_train=35000 n_test=15000 n_features=30
======================================================================================

FIT
  sklearn :  1607.06 ms  (IQR 1582.95–1619.98)
  arboria :  2166.19 ms  (IQR 2015.70–2197.56)
  speedup (sk/arb) :   0.74x

PREDICT
  sklearn :    26.55 ms  (IQR  26.02– 26.90)
  arboria :    15.35 ms  (IQR  14.95– 16.29)
  speedup (sk/arb) :   1.73x

PREDICT_PROBA
  sklearn :    26.37 ms  (IQR  25.71– 27.38)
  arboria :    14.92 ms  (IQR  14.57– 15.48)
  speedup (sk/arb) :   1.77x

ACCURACY (test)
  sklearn : 0.9496
  arboria : 0.9446



======================================================================================
Dataset: synthetic_100000_30f | n_train=70000 n_test=30000 n_features=30
======================================================================================

FIT
  sklearn :  3778.43 ms  (IQR 3506.26–3851.12)
  arboria :  4101.98 ms  (IQR 4090.61–4125.61)
  speedup (sk/arb) :   0.92x

PREDICT
  sklearn :    37.99 ms  (IQR  37.35– 38.60)
  arboria :    26.39 ms  (IQR  26.33– 27.09)
  speedup (sk/arb) :   1.44x

PREDICT_PROBA
  sklearn :    38.95 ms  (IQR  37.76– 39.50)
  arboria :    26.60 ms  (IQR  26.50– 26.68)
  speedup (sk/arb) :   1.46x

ACCURACY (test)
  sklearn : 0.9336
  arboria : 0.9298



======================================================================================
Dataset: synthetic_150000_30f | n_train=105000 n_test=45000 n_features=30
======================================================================================

FIT
  sklearn :  5316.68 ms  (IQR 5272.62–5354.21)
  arboria :  6681.21 ms  (IQR 6639.64–6745.12)
  speedup (sk/arb) :   0.80x

PREDICT
  sklearn :    52.15 ms  (IQR  50.41– 53.13)
  arboria :    45.17 ms  (IQR  44.89– 45.53)
  speedup (sk/arb) :   1.15x

PREDICT_PROBA
  sklearn :    51.08 ms  (IQR  50.05– 52.36)
  arboria :    45.57 ms  (IQR  44.68– 46.11)
  speedup (sk/arb) :   1.12x

ACCURACY (test)
  sklearn : 0.9081
  arboria : 0.9054



======================================================================================
Dataset: synthetic_200000_30f | n_train=140000 n_test=60000 n_features=30
======================================================================================

FIT
  sklearn :  8908.29 ms  (IQR 8682.09–8968.83)
  arboria : 11072.99 ms  (IQR 10877.43–11190.48)
  speedup (sk/arb) :   0.80x

PREDICT
  sklearn :    71.43 ms  (IQR  67.66– 75.55)
  arboria :    60.58 ms  (IQR  60.32– 61.04)
  speedup (sk/arb) :   1.18x

PREDICT_PROBA
  sklearn :    66.48 ms  (IQR  65.28– 69.53)
  arboria :    60.74 ms  (IQR  60.17– 61.27)
  speedup (sk/arb) :   1.09x

ACCURACY (test)
  sklearn : 0.8819
  arboria : 0.8768



======================================================================================
Dataset: synthetic_250000_30f | n_train=175000 n_test=75000 n_features=30
======================================================================================

FIT
  sklearn : 12110.11 ms  (IQR 11794.51–12248.21)
  arboria : 14591.73 ms  (IQR 14400.03–15739.24)
  speedup (sk/arb) :   0.83x

PREDICT
  sklearn :    90.30 ms  (IQR  83.71– 96.48)
  arboria :    75.19 ms  (IQR  73.93– 76.19)
  speedup (sk/arb) :   1.20x

PREDICT_PROBA
  sklearn :    80.39 ms  (IQR  79.11– 81.19)
  arboria :    76.12 ms  (IQR  75.50– 76.32)
  speedup (sk/arb) :   1.06x

ACCURACY (test)
  sklearn : 0.8932
  arboria : 0.8864
```