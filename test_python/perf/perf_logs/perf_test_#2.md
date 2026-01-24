
---
23/01/2025 (after optimisation update, in monothreading)
---

This test was ran after commit `0d39f33` (perf optimizations) was made on arboria and on monothreading.

```bash
======================================================================================
Dataset: breast_cancer | n_train=398 n_test=171 n_features=30
======================================================================================

FIT
  sklearn :   146.58 ms  (IQR 145.75–147.47)
  arboria :    72.71 ms  (IQR  71.88– 74.78)
  speedup (sk/arb) :   2.02x

PREDICT
  sklearn :     4.22 ms  (IQR   4.21–  4.24)
  arboria :     0.83 ms  (IQR   0.83–  0.83)
  speedup (sk/arb) :   5.08x

PREDICT_PROBA
  sklearn :     4.21 ms  (IQR   4.20–  4.21)
  arboria :     0.83 ms  (IQR   0.83–  0.83)
  speedup (sk/arb) :   5.09x

ACCURACY (test)
  sklearn : 0.9298
  arboria : 0.9357



======================================================================================
Dataset: synthetic_5k_25f | n_train=3500 n_test=1500 n_features=25
======================================================================================

FIT
  sklearn :  1447.22 ms  (IQR 1446.37–1469.98)
  arboria :  1886.21 ms  (IQR 1883.25–1891.88)
  speedup (sk/arb) :   0.77x

PREDICT
  sklearn :    17.93 ms  (IQR  17.91– 17.97)
  arboria :    21.95 ms  (IQR  21.94– 21.98)
  speedup (sk/arb) :   0.82x

PREDICT_PROBA
  sklearn :    17.78 ms  (IQR  17.78– 17.85)
  arboria :    22.01 ms  (IQR  22.00– 22.03)
  speedup (sk/arb) :   0.81x

ACCURACY (test)
  sklearn : 0.9260
  arboria : 0.9167



======================================================================================
Dataset: synthetic_50k_30f | n_train=35000 n_test=15000 n_features=30
======================================================================================

FIT
  sklearn : 19267.12 ms  (IQR 19023.09–19555.93)
  arboria : 30623.22 ms  (IQR 30469.85–33096.31)
  speedup (sk/arb) :   0.63x

PREDICT
  sklearn :   152.51 ms  (IQR 151.33–163.33)
  arboria :   342.24 ms  (IQR 328.61–367.17)
  speedup (sk/arb) :   0.45x

PREDICT_PROBA
  sklearn :   154.03 ms  (IQR 153.95–154.04)
  arboria :   448.23 ms  (IQR 375.19–479.12)
  speedup (sk/arb) :   0.34x

ACCURACY (test)
  sklearn : 0.9607
  arboria : 0.9597
```



