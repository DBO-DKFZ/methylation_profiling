"""Model registry keyed by ``(task_name, model_name)``.

Centralises both the classification and ordinal model registries. The CV loop picks the right block via ``MODELS[task.name]``.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ...config import RANDOM_STATE
from .classification import MLPClassifier, CNNClassifier
from .ordinal import CORALRegressor, CORNRegressor, MARKER_ORDINAL_MODELS, OrdinalMLPRegressor, ORDINAL_MODELS


def _scaled(est):
    """Wrap a scale-sensitive estimator in a ``StandardScaler`` pipeline.

    Important for marker/stacked features where Horvath EAA and CNV burden live on very different scales from
    EpiScore fractions and [0,1]-bounded OOF probabilities.

    :param est: Sklearn-compatible estimator to wrap.
    :return: ``Pipeline([('scaler', StandardScaler()), ('clf', est)])``.
    """
    return Pipeline([('scaler', StandardScaler()), ('clf', est)])


CLASSIFICATION_MODELS = {
    'xgboost': XGBClassifier(
        objective='multi:softprob', num_class=3,
        max_depth=5, n_estimators=1000, random_state=RANDOM_STATE,
    ),
    'random_forest': RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE),
    'logistic':   LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'elasticnet': LogisticRegression(solver='saga', l1_ratio=0.5, max_iter=1000, random_state=RANDOM_STATE),
    'svm':        SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    'mlp':        MLPClassifier(hidden_dims=(256, 128), dropout=0.3, epochs=30, batch_size=32),
    'cnn':        CNNClassifier(channels=(32, 64, 128), kernel_size=11, fc_dim=64,
                                dropout=0.3, epochs=30, batch_size=16),
}


# Compact marker-style classifiers (smaller MLP, no CNN): used when the feature space is already low-dimensional
# (markers, stacked).
MARKER_CLASSIFICATION_MODELS = {
    'logistic':      _scaled(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    'elasticnet':    _scaled(LogisticRegression(solver='saga', l1_ratio=0.5, max_iter=1000, random_state=RANDOM_STATE)),
    'svm':           _scaled(SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)),
    'random_forest': RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE),
    'xgboost': XGBClassifier(
        objective='multi:softprob', num_class=3,
        max_depth=5, n_estimators=500, random_state=RANDOM_STATE,
    ),
    'mlp':           _scaled(MLPClassifier(hidden_dims=(32, 16), dropout=0.3, epochs=30, batch_size=32)),
}


# Two-level registry: MODELS[task_name][feature_kind] -> dict[model_name, estimator] ``feature_kind`` is 'cpg'
# (raw CpGs, full model set) or 'lowdim' (markers/stacked, compact set including the _scaled variants).
MODELS: dict[str, dict[str, dict]] = {
    'classification': {
        'cpg':    CLASSIFICATION_MODELS,
        'lowdim': MARKER_CLASSIFICATION_MODELS,
    },
    'ordinal': {
        'cpg':    ORDINAL_MODELS,
        'lowdim': MARKER_ORDINAL_MODELS,
    },
}


__all__ = [
    'MODELS',
    'CLASSIFICATION_MODELS', 'MARKER_CLASSIFICATION_MODELS',
    'ORDINAL_MODELS', 'MARKER_ORDINAL_MODELS',
    'MLPClassifier', 'CNNClassifier',
    'CORALRegressor', 'CORNRegressor', 'OrdinalMLPRegressor',
    '_scaled',
]
