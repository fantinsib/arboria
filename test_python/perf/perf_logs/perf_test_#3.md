
---
24/01/2025 (Multithreading update, n_jobs = -1)
---

This test was ran after adding multithreading, and uses n_jobs = -1 for both sklearn and arboria. 4 tests were ran, as OS scheduling noise create high variance in performance for large datasets. 

### RUN 1 
---
```bash

======================================================================================
Dataset: breast_cancer | n_train=398 n_test=171 n_features=30
======================================================================================

FIT
  sklearn :   115.63 ms  (IQR 115.32–121.31)
  arboria :    14.43 ms  (IQR  14.37– 14.43)
  speedup (sk/arb) :   8.01x

PREDICT
  sklearn :    13.64 ms  (IQR  13.00– 13.68)
  arboria :     1.11 ms  (IQR   1.10–  1.11)
  speedup (sk/arb) :  12.30x

PREDICT_PROBA
  sklearn :    13.61 ms  (IQR  13.58– 13.68)
  arboria :     1.11 ms  (IQR   1.11–  1.11)
  speedup (sk/arb) :  12.29x

ACCURACY (test)
  sklearn : 0.9298
  arboria : 0.9415



======================================================================================
Dataset: synthetic_5k_25f | n_train=3500 n_test=1500 n_features=25
======================================================================================

FIT
  sklearn :   360.55 ms  (IQR 357.71–361.02)
  arboria :   336.77 ms  (IQR 320.62–356.76)
  speedup (sk/arb) :   1.07x

PREDICT
  sklearn :    26.47 ms  (IQR  26.47– 26.64)
  arboria :    22.52 ms  (IQR  22.52– 22.55)
  speedup (sk/arb) :   1.18x

PREDICT_PROBA
  sklearn :    26.45 ms  (IQR  26.39– 26.51)
  arboria :    22.53 ms  (IQR  22.52– 22.53)
  speedup (sk/arb) :   1.17x

ACCURACY (test)
  sklearn : 0.9260
  arboria : 0.9213



======================================================================================
Dataset: synthetic_50k_30f | n_train=35000 n_test=15000 n_features=30
======================================================================================

FIT
  sklearn :  3828.73 ms  (IQR 3667.98–4352.48)
  arboria :  6287.02 ms  (IQR 6121.38–6435.13)
  speedup (sk/arb) :   0.61x

PREDICT
  sklearn :    52.11 ms  (IQR  50.46– 53.28)
  arboria :   400.25 ms  (IQR 396.44–422.29)
  speedup (sk/arb) :   0.13x

PREDICT_PROBA
  sklearn :    50.10 ms  (IQR  49.89– 51.75)
  arboria :   362.48 ms  (IQR 360.03–371.09)
  speedup (sk/arb) :   0.14x

ACCURACY (test)
  sklearn : 0.9607
  arboria : 0.9602
(base) fantinsibony@MacBook-Air-de-Fantin-4 perf % python3 arboria_perf.py
```
---
### RUN 2

```bash
======================================================================================
Dataset: breast_cancer | n_train=398 n_test=171 n_features=30
======================================================================================

FIT
  sklearn :   120.19 ms  (IQR 115.26–121.69)
  arboria :    14.40 ms  (IQR  14.36– 14.45)
  speedup (sk/arb) :   8.35x

PREDICT
  sklearn :    13.66 ms  (IQR  13.55– 13.74)
  arboria :     1.09 ms  (IQR   1.09–  1.09)
  speedup (sk/arb) :  12.54x

PREDICT_PROBA
  sklearn :    13.61 ms  (IQR  13.33– 13.63)
  arboria :     1.09 ms  (IQR   1.09–  1.09)
  speedup (sk/arb) :  12.48x

ACCURACY (test)
  sklearn : 0.9298
  arboria : 0.9415



======================================================================================
Dataset: synthetic_5k_25f | n_train=3500 n_test=1500 n_features=25
======================================================================================

FIT
  sklearn :   372.79 ms  (IQR 366.12–406.50)
  arboria :   331.49 ms  (IQR 329.15–341.58)
  speedup (sk/arb) :   1.12x

PREDICT
  sklearn :    26.42 ms  (IQR  14.18– 26.74)
  arboria :    22.67 ms  (IQR  22.67– 22.67)
  speedup (sk/arb) :   1.17x

PREDICT_PROBA
  sklearn :    26.69 ms  (IQR  26.68– 26.83)
  arboria :    22.67 ms  (IQR  22.67– 22.67)
  speedup (sk/arb) :   1.18x

ACCURACY (test)
  sklearn : 0.9260
  arboria : 0.9213



======================================================================================
Dataset: synthetic_50k_30f | n_train=35000 n_test=15000 n_features=30
======================================================================================

FIT
  sklearn :  4927.14 ms  (IQR 4557.60–5410.51)
  arboria :  5440.16 ms  (IQR 5102.62–5966.67)
  speedup (sk/arb) :   0.91x

PREDICT
  sklearn :    64.03 ms  (IQR  50.09– 74.17)
  arboria :   312.66 ms  (IQR 272.96–344.16)
  speedup (sk/arb) :   0.20x

PREDICT_PROBA
  sklearn :    66.01 ms  (IQR  65.77– 72.79)
  arboria :   273.65 ms  (IQR 231.50–289.35)
  speedup (sk/arb) :   0.24x

ACCURACY (test)
  sklearn : 0.9607
  arboria : 0.9602
(base) fantinsibony@MacBook-Air-de-Fantin-4 perf % python3 arboria_perf.py

```
---
### RUN 3

```bash
======================================================================================
Dataset: breast_cancer | n_train=398 n_test=171 n_features=30
======================================================================================

FIT
  sklearn :   201.66 ms  (IQR 127.33–399.43)
  arboria :    18.56 ms  (IQR  18.36– 24.17)
  speedup (sk/arb) :  10.87x

PREDICT
  sklearn :    14.27 ms  (IQR  14.01– 27.83)
  arboria :     1.36 ms  (IQR   1.34–  1.45)
  speedup (sk/arb) :  10.49x

PREDICT_PROBA
  sklearn :    14.38 ms  (IQR  13.94– 14.51)
  arboria :     1.36 ms  (IQR   1.35–  1.37)
  speedup (sk/arb) :  10.61x

ACCURACY (test)
  sklearn : 0.9298
  arboria : 0.9415



======================================================================================
Dataset: synthetic_5k_25f | n_train=3500 n_test=1500 n_features=25
======================================================================================

FIT
  sklearn :   517.23 ms  (IQR 505.08–530.61)
  arboria :   394.14 ms  (IQR 391.44–394.99)
  speedup (sk/arb) :   1.31x

PREDICT
  sklearn :    26.47 ms  (IQR  26.42– 26.61)
  arboria :    25.47 ms  (IQR  24.76– 26.19)
  speedup (sk/arb) :   1.04x

PREDICT_PROBA
  sklearn :    25.57 ms  (IQR  22.97– 25.82)
  arboria :    23.11 ms  (IQR  23.08– 24.06)
  speedup (sk/arb) :   1.11x

ACCURACY (test)
  sklearn : 0.9260
  arboria : 0.9213



======================================================================================
Dataset: synthetic_50k_30f | n_train=35000 n_test=15000 n_features=30
======================================================================================

FIT
  sklearn :  5144.16 ms  (IQR 4498.22–6166.87)
  arboria :  4763.15 ms  (IQR 4412.18–5021.13)
  speedup (sk/arb) :   1.08x

PREDICT
  sklearn :    50.13 ms  (IQR  49.96– 51.63)
  arboria :   221.27 ms  (IQR 221.26–221.33)
  speedup (sk/arb) :   0.23x

PREDICT_PROBA
  sklearn :    52.09 ms  (IQR  50.56– 52.13)
  arboria :   224.87 ms  (IQR 224.80–228.28)
  speedup (sk/arb) :   0.23x

ACCURACY (test)
  sklearn : 0.9607
  arboria : 0.9602
(base) fantinsibony@MacBook-Air-de-Fantin-4 perf % python3 arboria_perf.py

```
---
### RUN 4

```bash
======================================================================================
Dataset: breast_cancer | n_train=398 n_test=171 n_features=30
======================================================================================

FIT
  sklearn :   116.94 ms  (IQR 116.31–119.96)
  arboria :    19.64 ms  (IQR  18.97– 21.87)
  speedup (sk/arb) :   5.95x

PREDICT
  sklearn :    13.60 ms  (IQR  13.56– 13.71)
  arboria :     1.16 ms  (IQR   1.15–  1.17)
  speedup (sk/arb) :  11.73x

PREDICT_PROBA
  sklearn :    13.34 ms  (IQR  13.24– 13.43)
  arboria :     1.17 ms  (IQR   1.15–  1.18)
  speedup (sk/arb) :  11.45x

ACCURACY (test)
  sklearn : 0.9298
  arboria : 0.9415



======================================================================================
Dataset: synthetic_5k_25f | n_train=3500 n_test=1500 n_features=25
======================================================================================

FIT
  sklearn :   430.52 ms  (IQR 425.93–452.38)
  arboria :   373.00 ms  (IQR 366.09–374.43)
  speedup (sk/arb) :   1.15x

PREDICT
  sklearn :    26.43 ms  (IQR  25.46– 39.01)
  arboria :    23.93 ms  (IQR  23.59– 24.03)
  speedup (sk/arb) :   1.10x

PREDICT_PROBA
  sklearn :    25.83 ms  (IQR  25.25– 26.62)
  arboria :    23.54 ms  (IQR  23.52– 23.56)
  speedup (sk/arb) :   1.10x

ACCURACY (test)
  sklearn : 0.9260
  arboria : 0.9213



======================================================================================
Dataset: synthetic_50k_30f | n_train=35000 n_test=15000 n_features=30
======================================================================================

FIT
  sklearn :  4254.82 ms  (IQR 4111.36–4421.02)
  arboria :  4480.23 ms  (IQR 4426.11–4634.25)
  speedup (sk/arb) :   0.95x

PREDICT
  sklearn :    52.04 ms  (IQR  50.94– 52.55)
  arboria :   225.86 ms  (IQR 224.33–227.04)
  speedup (sk/arb) :   0.23x

PREDICT_PROBA
  sklearn :    50.04 ms  (IQR  49.47– 50.07)
  arboria :   227.20 ms  (IQR 224.48–248.55)
  speedup (sk/arb) :   0.22x

ACCURACY (test)
  sklearn : 0.9607
  arboria : 0.9602


```
