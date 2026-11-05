# Reproduced benchmark

Seed 42; stratified holdout 450 / 1797; train-only standardization.

## Quality

| Runtime | Accuracy |
|---|---:|
| sklearn reference | 0.9600 |
| NumPy float32 | 0.9600 |
| NumPy int8 | 0.9600 |

Float32 max probability difference vs sklearn: 0. Int8 mean probability absolute difference: 0.0003; prediction agreement: 100.0%.

## What the numbers mean

The custom float32 implementation reproduces the framework probabilities on this run, so the exported layer layout and forward-pass math agree for the evaluated model. Int8 preserved every test prediction here and changed probabilities only slightly, but this is not evidence that quantization is always harmless: per-batch activation calibration can change when batch composition changes, and this small test set has limited power to expose rare failures.

The error matrix has 18 mistakes among 450 test samples (96.0% accuracy). The largest single confusion is true digit 8 predicted as 1 (5 cases); 5→9 occurs 3 times and 4→7 occurs 2 times. These are a few observed errors, not a claim about general handwriting. Inspect `confusion_int8.csv` for the complete class-by-class counts.

Batching reduced overhead for the custom float32 path: median throughput rose from 61,091 samples/s at batch 1 to 1,578,947 samples/s at batch 450 in this run. The int8 path was slower at all tested batch sizes because this educational implementation pays quantization and conversion costs without optimized integer kernels. The sklearn reference was timed only on the full 450-row batch, so it is not a per-batch comparison. Results are local to the recorded two-core environment and seven repeats.

## Timing

| Runtime | Batch | Median ms | P95 ms | Samples/s |
|---|---:|---:|---:|---:|
| sklearn_reference | 450 | 0.364 | 0.437 | 1236263.7 |
| numpy_float32 | 1 | 7.366 | 7.797 | 61091.5 |
| numpy_float32 | 16 | 0.869 | 0.993 | 517836.6 |
| numpy_float32 | 64 | 0.632 | 0.673 | 712025.3 |
| numpy_float32 | 450 | 0.285 | 0.335 | 1578947.4 |
| numpy_int8 | 1 | 28.732 | 28.989 | 15662.0 |
| numpy_int8 | 16 | 6.922 | 7.037 | 65010.1 |
| numpy_int8 | 64 | 5.894 | 6.894 | 76348.8 |
| numpy_int8 | 450 | 5.672 | 6.077 | 79337.1 |

Timings are local observations, not portable performance guarantees. See `metrics.json` for environment, full quality, confusion matrix and limitations.
