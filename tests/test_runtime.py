import warnings
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from fastml_runtime.runtime import DenseRuntime, export_mlp


def fitted():
    d=load_digits(); x=StandardScaler().fit_transform(d.data[:900]).astype(np.float32); y=d.target[:900]
    # This fixture tests inference plumbing, not convergence; keep its training bounded.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m=MLPClassifier(hidden_layer_sizes=(24,),max_iter=250,random_state=3,early_stopping=False).fit(x,y)
    return x,m

def test_float_runtime_matches_sklearn_probabilities():
    x,m=fitted(); rt=DenseRuntime(export_mlp(m),"float32")
    np.testing.assert_allclose(rt.predict_proba(x[:25]),m.predict_proba(x[:25]),rtol=2e-5,atol=2e-6)

def test_quantized_runtime_is_finite_and_reasonably_accurate():
    x,m=fitted(); rt=DenseRuntime(export_mlp(m),"int8")
    p=rt.predict_proba(x[:100]); assert p.shape==(100,10) and np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1),1,atol=1e-5)
    assert np.mean(rt.predict(x[:100])==m.predict(x[:100]))>.85

def test_batched_and_empty_shapes():
    x,m=fitted(); rt=DenseRuntime(export_mlp(m))
    np.testing.assert_allclose(rt.predict_batched(x[:33],7),rt.predict_proba(x[:33]),atol=1e-6)
    assert rt.predict_batched(x[:0],7).shape==(0,10)
    with pytest.raises(ValueError): rt.predict_batched(x,0)

def test_rejects_bad_input_and_layer_shapes():
    rt=DenseRuntime([(np.ones((2,3),dtype=np.float32),np.zeros(2,dtype=np.float32))])
    with pytest.raises(ValueError): rt.predict_proba(np.ones((2,4)))
    with pytest.raises(ValueError): rt.predict_proba(np.array([[np.nan, 0, 0]], dtype=np.float32).reshape(1, 3))
    with pytest.raises(ValueError): DenseRuntime([(np.ones((2,3)),np.zeros(3))])
