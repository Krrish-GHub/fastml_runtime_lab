from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def export_mlp(model: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Copy sklearn MLPClassifier dense layers into (out,in) runtime layout."""
    if not hasattr(model, "coefs_") or not hasattr(model, "intercepts_"):
        raise TypeError("expected a fitted sklearn MLPClassifier")
    return [(np.asarray(w.T, dtype=np.float32).copy(), np.asarray(b, dtype=np.float32).copy())
            for w, b in zip(model.coefs_, model.intercepts_)]


@dataclass
class DenseRuntime:
    """Dense ReLU MLP inference in float32 or symmetric per-output int8 weights.

    For int8 mode, each input activation tensor is symmetrically quantized per
    batch; weight scales are per output channel. Integer dot products accumulate
    in int32 and are dequantized before bias and activation.
    """
    layers: list[tuple[np.ndarray, np.ndarray]]
    mode: str = "float32"

    def __post_init__(self):
        if self.mode not in {"float32", "int8"}:
            raise ValueError("mode must be 'float32' or 'int8'")
        if not self.layers:
            raise ValueError("at least one dense layer is required")
        self._prepared = []
        expected = None
        for weight, bias in self.layers:
            w, b = np.asarray(weight), np.asarray(bias)
            if w.ndim != 2 or b.ndim != 1 or w.shape[0] != b.shape[0]:
                raise ValueError("each layer needs weight [out,in] and bias [out]")
            if expected is not None and w.shape[1] != expected:
                raise ValueError("adjacent layer dimensions do not match")
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("weights and biases must be finite")
            expected = w.shape[0]
            w, b = w.astype(np.float32), b.astype(np.float32)
            if self.mode == "int8":
                scale = np.max(np.abs(w), axis=1) / 127.0
                scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
                qw = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
                self._prepared.append((qw, b, scale))
            else:
                self._prepared.append((w, b, None))
        self.input_features = self.layers[0][0].shape[1]
        self.output_features = self.layers[-1][0].shape[0]

    @staticmethod
    def _quantize(x: np.ndarray):
        max_abs = np.max(np.abs(x), axis=1, keepdims=True)
        scale = np.where(max_abs == 0, 1.0, max_abs / 127.0).astype(np.float32)
        q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
        return q, scale

    def logits(self, inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_features:
            raise ValueError(f"inputs must have shape (batch, {self.input_features})")
        if not np.isfinite(x).all():
            raise ValueError("inputs must be finite")
        for index, (weight, bias, weight_scale) in enumerate(self._prepared):
            if self.mode == "float32":
                x = x @ weight.T + bias
            else:
                qx, x_scale = self._quantize(x)
                accum = qx.astype(np.int32) @ weight.astype(np.int32).T
                x = accum.astype(np.float32) * x_scale * weight_scale[None, :] + bias
            if index < len(self._prepared) - 1:
                np.maximum(x, 0, out=x)
        return x

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(inputs))

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(inputs), axis=1)

    def predict_batched(self, inputs: np.ndarray, batch_size: int) -> np.ndarray:
        x = np.asarray(inputs)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if x.ndim != 2 or x.shape[1] != self.input_features:
            raise ValueError(f"inputs must have shape (batch, {self.input_features})")
        if len(x) == 0:
            return np.empty((0, self.output_features), dtype=np.float32)
        return np.concatenate([self.predict_proba(x[i:i + batch_size]) for i in range(0, len(x), batch_size)], axis=0)
