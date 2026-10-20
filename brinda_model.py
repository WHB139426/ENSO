"""Brinda Puri's single-frame PCA + pixelwise ridge setting.

Adapted from brinda/ECS289G_ENSO_project.ipynb, cells 29--52. The shared
comparison uses the Pacific domain, training-only climatology/mask/PCA, and
common target months. It does not reproduce the original global-domain run.
Unlike the lagged relative-SST setting, this method uses one ordinary SSTA
field, unweighted PCA scores without scaling, and predicts ocean pixels
directly. Hyperparameters maximize validation Niño3.4 correlation.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from evaluate import correlation

COMPONENTS = (5, 10, 15, 20, 30, 50)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def prepare_single_frame(prepared, seed=42):
    ordinary = prepared["relative"] + prepared["background"][:, None]
    kmax = min(max(COMPONENTS), int(prepared["train_months"].sum()) - 1, ordinary.shape[1])
    pca = PCA(n_components=kmax, svd_solver="randomized", random_state=seed)
    pca.fit(ordinary[prepared["train_months"]])
    return {"ordinary": ordinary, "pca": pca, "scores": pca.transform(ordinary)}


def fit_single_frame(context, origins, targets, split, nino_weights):
    """Select on index correlation, then fit the selected full-field regressor.

    With the same ridge penalty for every pixel, area-averaging ridge outputs
    commutes with fitting. A scalar index response therefore performs the
    exact same hyperparameter search without repeatedly fitting 11k outputs.
    The selected estimator is subsequently fitted to every target ocean cell;
    an explicit check verifies agreement with the scalar search predictions.
    """
    scores, ordinary = context["scores"], context["ordinary"]
    train, validation = split == "train", split == "validation"
    response_index = ordinary[targets] @ nino_weights
    best, rows = None, []
    for k in COMPONENTS:
        if k > scores.shape[1]:
            continue
        for alpha in ALPHAS:
            proxy = Ridge(alpha=alpha, solver="cholesky")
            proxy.fit(scores[origins[train], :k], response_index[train])
            predicted = proxy.predict(scores[origins[validation], :k])
            r = correlation(predicted, response_index[validation])
            if r is None:
                raise ValueError("Undefined single-frame validation correlation.")
            rows.append({"components": k, "alpha": alpha, "validation_nino34_correlation": r})
            if best is None or r > best[0]:
                best = (r, k, alpha, predicted)
    if best is None:
        raise ValueError("No feasible single-frame PCA candidate.")
    r, k, alpha, proxy_prediction = best
    regressor = Ridge(alpha=alpha, solver="cholesky")
    regressor.fit(scores[origins[train], :k], ordinary[targets[train]])
    selected = (split == "validation") | (split == "test")
    prediction = regressor.predict(scores[origins[selected], :k])
    np.testing.assert_allclose(prediction[split[selected] == "validation"] @ nino_weights,
                               proxy_prediction, atol=1e-8, rtol=1e-8)
    table = pd.DataFrame(rows)
    table["selected"] = (table.components == k) & (table.alpha == alpha)
    bundle = {"method": "single_frame_ridge", "pca": context["pca"],
              "regressor": regressor, "selected_components": k,
              "selected_alpha": alpha, "validation_correlation": r,
              "setting": "Single ordinary SSTA frame; unweighted PCA; direct pixel targets"}
    return bundle, prediction, table


def predict_single_frame(bundle, ordinary_ssta):
    """Predict anomaly vectors using only the saved transforms and estimator."""
    k = bundle["selected_components"]
    scores = bundle["pca"].transform(ordinary_ssta)
    return bundle["regressor"].predict(scores[:, :k])
