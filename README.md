# FastML Runtime Lab

An inspectable mini inference runtime and benchmark for a real, technical ML systems question: what changes when a trained dense classifier is served with a custom NumPy float32 engine, per-batch int8 quantization, and different request batch sizes?

This is not another dataset-prediction demo. It trains a modest MLP, exports its learned dense layers, reproduces the reference model's outputs, implements int32-accumulating symmetric int8 matrix multiplies, and compares quality and latency/throughput on a held-out workload. It makes accuracy-performance tradeoffs and runtime mechanics visible.

## Why publish it?

ML deployment is more than training a model. This project makes the boundary between a framework model and an inference runtime concrete: tensor shapes, ReLU/softmax math, numerical equivalence, quantization error, micro-batching, and reproducible measurement. The goal is a small educational systems lab—not a claim to beat optimized serving libraries.

## Dataset and experiment

- Uses `sklearn.datasets.load_digits`, a public, bundled 8×8 handwritten-digit classification dataset; no personal data or network download.
- A stratified 75/25 train/test split is fixed by seed. `StandardScaler` is fitted only on training examples.
- Trains a two-hidden-layer `(128, 64)` ReLU MLP with Adam and early stopping. The test set is not used for training or selecting parameters.
- Exports framework layer weights and biases to a self-contained runtime. Float32 outputs are checked against scikit-learn. Int8 mode quantizes weights per output channel, dynamically quantizes activations per batch, accumulates integer products in int32, then dequantizes before bias and activation.
- Times reference inference and custom float32/int8 inference over batch sizes 1, 16, 64 and full holdout, after warmup; reports median and p95 latency plus throughput. Quality, confusion matrix, predictions, fitted parameters and environment metadata are saved.

## What the reference run showed

On the recorded 450-example holdout, scikit-learn and the custom float32 runtime both reached 96.0% accuracy; float32 probabilities matched exactly in this run. The int8 runtime also reached 96.0%, with 0.00031 mean absolute probability error and 100% prediction agreement with float32. That does not make this int8 implementation faster: its Python/NumPy quantization overhead dominated, and it was slower than the optimized reference in this environment. Batching improved the custom float32 path's throughput; read `reports/RESULTS.md` for the observed measurements and error breakdown. These are machine-specific measurements, not a general performance claim.

## Reproduce

Python 3.10+:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
fastml-lab --output reports --seed 42 --repeats 7
```

The sklearn digits data ships with scikit-learn. `reports/RESULTS.md`, `reports/metrics.json`, `reports/confusion_int8.csv`, and `reports/throughput.png` contain the measured result from this repository's reference run; run locally to regenerate. Exact performance varies by CPU, NumPy/BLAS build and system load.

## Scope and limitations

- This intentionally small dense CPU runtime is not a general model format, graph compiler, or production server; it does not claim superiority to ONNX Runtime, PyTorch or vendor kernels.
- Per-batch activation calibration means the int8 result depends on which samples share a batch. The experiment uses one fixed test batch per timing/quality report.
- Quantization can change confidence and predictions; use reported errors, not a speed number alone. This simple int8 path can be slower than vectorized optimized float32.
- Digits is a small, clean educational benchmark. It does not establish performance on larger or real serving workloads.
- Timings are measurements on the machine that generated the report and should not be generalized.

## Layout

- `src/fastml_runtime/runtime.py`: reusable float32 and int8 dense inference, batched API, model export.
- `src/fastml_runtime/experiment.py`: seeded training, evaluation, timing and artifact generation.
- `tests/`: runtime numerical equivalence, quantization sanity, batching, and validation errors.
- `reports/`: reference metrics, confusion matrix, throughput plot and reproducibility summary.
