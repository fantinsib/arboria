
---
23/01/2025 (before optimisation update)
---

This test was ran before any optimization was made on arboria and on monothreading.
```bash
======================================================================================
Dataset: breast_cancer | n_train=398 n_test=171 n_features=30
======================================================================================

FIT
  sklearn :   147.88 ms  (IQR 147.63–148.22)
  arboria :   546.49 ms  (IQR 536.38–557.93)
  speedup (sk/arb) :   0.27x

PREDICT
  sklearn :     4.59 ms  (IQR   4.36–  5.04)
  arboria :     0.81 ms  (IQR   0.81–  0.84)
  speedup (sk/arb) :   5.65x

PREDICT_PROBA
  sklearn :     4.29 ms  (IQR   4.29–  4.30)
  arboria :     0.82 ms  (IQR   0.82–  0.83)
  speedup (sk/arb) :   5.21x

ACCURACY (test)
  sklearn : 0.9298
  arboria : 0.9357

======================================================================================
Dataset: synthetic_5k_25f | n_train=3500 n_test=1500 n_features=25
======================================================================================

FIT
  sklearn :  1443.20 ms  (IQR 1432.64–1532.15)
  arboria : 55460.77 ms  (IQR 54228.86–59204.05)
  speedup (sk/arb) :   0.03x

PREDICT
  sklearn :    17.51 ms  (IQR  17.50– 17.56)
  arboria :    21.71 ms  (IQR  21.71– 22.27)
  speedup (sk/arb) :   0.81x

PREDICT_PROBA
  sklearn :    17.45 ms  (IQR  17.44– 17.49)
  arboria :    22.13 ms  (IQR  21.95– 22.14)
  speedup (sk/arb) :   0.79x

ACCURACY (test)
  sklearn : 0.9260
  arboria : 0.9167
```