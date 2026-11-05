from __future__ import annotations
import argparse, json, platform, statistics, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .runtime import DenseRuntime, export_mlp


def _bench(fn, repeats: int = 7, warmups: int = 2) -> dict:
    for _ in range(warmups): fn()
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter(); fn(); samples.append(time.perf_counter()-start)
    return {"median_ms": round(statistics.median(samples)*1000, 3), "p95_ms": round(float(np.percentile(samples,95))*1000,3), "repeats":repeats}


def run(output: Path, seed: int = 42, repeats: int = 7) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    data=load_digits()
    X_train, X_test, y_train, y_test=train_test_split(data.data.astype(np.float32), data.target, test_size=.25, random_state=seed, stratify=data.target)
    scaler=StandardScaler().fit(X_train)
    train=scaler.transform(X_train).astype(np.float32); test=scaler.transform(X_test).astype(np.float32)
    model=MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", solver="adam", alpha=1e-4,
        batch_size=64, learning_rate_init=1e-3, max_iter=350, early_stopping=True,
        validation_fraction=.15, n_iter_no_change=20, random_state=seed, tol=1e-4)
    model.fit(train,y_train)
    ref_prob=model.predict_proba(test); ref_pred=ref_prob.argmax(axis=1)
    layers=export_mlp(model)
    fp=DenseRuntime(layers,"float32"); q8=DenseRuntime(layers,"int8")
    fp_prob=fp.predict_proba(test); q_prob=q8.predict_proba(test)
    # Runtime integrity and accuracy comparison.
    max_abs=float(np.max(np.abs(fp_prob-ref_prob)))
    benchmark=[]
    for runtime_name, runtime, probs in [("sklearn_reference",None,ref_prob),("numpy_float32",fp,fp_prob),("numpy_int8",q8,q_prob)]:
        for batch in ([len(test)] if runtime is None else [1,16,64,len(test)]):
            fn=(lambda runtime=runtime: model.predict_proba(test)) if runtime is None else (lambda runtime=runtime,batch=batch: runtime.predict_batched(test,batch))
            timings=_bench(fn,repeats=repeats)
            benchmark.append({"runtime":runtime_name,"batch_size":batch,"samples":len(test),**timings,
                "samples_per_second":round(len(test)/(timings["median_ms"]/1000),1)})
    metrics={"dataset":"scikit-learn digits (8x8 handwritten digit images)","n_samples":int(len(data.data)),"n_features":int(data.data.shape[1]),"classes":10,
        "split":{"train":len(y_train),"test":len(y_test),"stratified":True,"seed":seed,"scaler_fit_on":"train only"},
        "training":{"estimator":"sklearn.neural_network.MLPClassifier","hidden_layers":[128,64],"activation":"ReLU","optimizer":"Adam","iterations":int(model.n_iter_),"converged_before_max_iter":bool(model.n_iter_<model.max_iter)},
        "quality":{"sklearn_accuracy":round(float(accuracy_score(y_test,ref_pred)),4),"float32_accuracy":round(float(accuracy_score(y_test,fp.predict(test))),4),"int8_accuracy":round(float(accuracy_score(y_test,q8.predict(test))),4),"sklearn_log_loss":round(float(log_loss(y_test,ref_prob)),4),"float32_max_probability_abs_diff":round(max_abs,7),"int8_probability_mae_vs_float32":round(float(np.mean(np.abs(q_prob-fp_prob))),6),"int8_prediction_agreement_with_float32":round(float(np.mean(q8.predict(test)==fp.predict(test))),4)},
        "benchmark_environment":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform(),"cpu_count":__import__("os").cpu_count(),"warmups":2,"repeats":repeats,"timing_note":"single-process local CPU; indicative, not a cross-machine claim"},"benchmark":benchmark,
        "confusion_matrix_int8":confusion_matrix(y_test,q8.predict(test)).tolist(),
        "limitations":["Digits is a small, clean educational dataset, not a production workload.","Benchmark timing depends on hardware, BLAS build, load and batch size; compare only within the same environment.","This is a minimal dense-network runtime, not a general ONNX or production serving engine.","Per-batch activation scaling makes int8 output dependent on batch composition; this experiment measures one fixed held-out batch.","Quantized mode is an educational CPU implementation; it may be slower than optimized floating-point libraries."]}
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    np.savez_compressed(output/"model.npz", means=scaler.mean_.astype(np.float32), scales=scaler.scale_.astype(np.float32), **{f"w{i}":w for i,(w,b) in enumerate(layers)}, **{f"b{i}":b for i,(w,b) in enumerate(layers)})
    (output/"confusion_int8.csv").write_text("true_label,"+",".join(map(str,range(10)))+"\n"+"\n".join(str(i)+","+",".join(map(str,row)) for i,row in enumerate(metrics["confusion_matrix_int8"]))+"\n")
    rows=metrics["benchmark"]
    fig,ax=plt.subplots(figsize=(8,4.5))
    selected=[r for r in rows if r["runtime"]!="sklearn_reference"]
    for name in ("numpy_float32","numpy_int8"):
        group=[r for r in selected if r["runtime"]==name]
        ax.plot([r["batch_size"] for r in group],[r["samples_per_second"] for r in group],marker="o",label=name.replace("_"," "))
    ax.set_xscale("log",base=2); ax.set_xlabel("inference batch size (log2)"); ax.set_ylabel("test samples / second (median)"); ax.set_title("Batching changes throughput on this machine"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output/"throughput.png",dpi=160); plt.close(fig)
    np.savez_compressed(output/"test_predictions.npz", y_true=y_test, sklearn_probabilities=ref_prob, float32_probabilities=fp_prob, int8_probabilities=q_prob)
    summary=["# Reproduced benchmark", "",f"Seed {seed}; stratified holdout {len(y_test)} / {len(data.data)}; train-only standardization.","", "## Quality", "", "| Runtime | Accuracy |", "|---|---:|", f"| sklearn reference | {metrics['quality']['sklearn_accuracy']:.4f} |",f"| NumPy float32 | {metrics['quality']['float32_accuracy']:.4f} |",f"| NumPy int8 | {metrics['quality']['int8_accuracy']:.4f} |", "", f"Float32 max probability difference vs sklearn: {max_abs:.2g}. Int8 mean probability absolute difference: {metrics['quality']['int8_probability_mae_vs_float32']:.4f}; prediction agreement: {metrics['quality']['int8_prediction_agreement_with_float32']:.1%}.","", "## Timing", "", "| Runtime | Batch | Median ms | P95 ms | Samples/s |", "|---|---:|---:|---:|---:|"]
    for r in rows: summary.append(f"| {r['runtime']} | {r['batch_size']} | {r['median_ms']:.3f} | {r['p95_ms']:.3f} | {r['samples_per_second']:.1f} |")
    summary.extend(["", "Timings are local observations, not portable performance guarantees. See `metrics.json` for environment, full quality, confusion matrix and limitations."])
    (output/"RESULTS.md").write_text("\n".join(summary)+"\n")
    return metrics


def main():
    p=argparse.ArgumentParser(description="Reproduce tiny MLP serving and quantization experiment")
    p.add_argument("--output",type=Path,default=Path("reports")); p.add_argument("--seed",type=int,default=42); p.add_argument("--repeats",type=int,default=7)
    a=p.parse_args(); m=run(a.output,a.seed,a.repeats); print(json.dumps({"quality":m["quality"],"benchmark":m["benchmark"]},indent=2))
if __name__=="__main__": main()
