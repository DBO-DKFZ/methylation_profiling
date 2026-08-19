"""Ordinal regression models for therapeutic group prediction.

* :class:`CORALRegressor` — CORAL ordinal regressor (Cao, Mirjalili & Raschka 2020).
* :class:`OrdinalMLPRegressor` — cumulative-logits ordinal MLP with BCE loss.
* :class:`CORNRegressor` — CORN ordinal regressor (Shi et al. 2021).

The registries :data:`ORDINAL_MODELS` and :data:`MARKER_ORDINAL_MODELS` hold sklearn-compatible regressors for the
raw-CpG and low-dim (markers/stacked) feature cases respectively.
"""
import logging

import mord
import numpy as np
import torch
import torch.nn as nn
from coral_pytorch.dataset import corn_label_from_logits, levels_from_labelbatch
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
from ogboost import GradientBoostingOrdinal
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from ...config import RANDOM_STATE
from .._utils import resolve_device

logger = logging.getLogger(__name__)


def _build_backbone(n_features: int, hidden_dims: tuple, dropout: float) -> nn.Sequential:
    """Shared MLP backbone: [Linear → BatchNorm → ReLU → Dropout] × N."""
    layers = []
    in_dim = n_features
    for out_dim in hidden_dims:
        layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = out_dim
    return nn.Sequential(*layers)


def _cum_to_class_probs(cum: np.ndarray) -> np.ndarray:
    """Convert cumulative ordinal probabilities into per-class probabilities.

    :param cum: ``P(Y > k)`` for ``k = 0..K-2``, shape ``(n, K-1)`` and (for a well-behaved ordinal head)
        non-increasing along axis 1.
    :return: ``P(Y = k)`` for ``k = 0..K-1``, shape ``(n, K)``. Differences that go negative because an
        unconstrained head (``OrdinalMLPRegressor``) breaks monotonicity are clipped to 0, and each row is
        renormalised to sum to 1 so the result is a valid distribution.
    """
    probs = np.empty((cum.shape[0], cum.shape[1] + 1), dtype=float)
    probs[:, 0] = 1.0 - cum[:, 0]
    probs[:, 1:-1] = cum[:, :-1] - cum[:, 1:]
    probs[:, -1] = cum[:, -1]
    np.clip(probs, 0.0, None, out=probs)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


class CORALRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible CORAL ordinal regressor (Cao, Mirjalili & Raschka 2020).

    Uses a shared MLP backbone followed by a CoralLayer (single weight vector + K-1 bias terms). Prediction is the
    expected rank: ``sum(sigmoid(logit_k))`` for ``k=0..K-2``.

    :param hidden_dims: Tuple of hidden layer widths (default: ``(64, 32)``).
    :param dropout: Dropout probability applied after each hidden layer (default: 0.3).
    :param epochs: Training epochs (default: 100).
    :param batch_size: Mini-batch size (default: 32).
    :param lr: Adam learning rate (default: 1e-3).
    :param device: ``'cpu'``, ``'cuda'``, or ``'auto'`` — ``'auto'`` picks cuda if available.
    """

    def __init__(self, hidden_dims=(64, 32), dropout=0.3,
                 epochs=100, batch_size=32, lr=1e-3, device='auto'):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def fit(self, X, y):
        dev = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)

        self.num_classes_ = int(y_np.max() - y_np.min()) + 1
        self.y_min_ = int(y_np.min())
        self.classes_ = np.arange(self.y_min_, self.y_min_ + self.num_classes_)
        y_shifted = y_np - self.y_min_  # shift to 0-based

        backbone = _build_backbone(X_np.shape[1], self.hidden_dims, self.dropout)
        coral_head = CoralLayer(self.hidden_dims[-1], self.num_classes_)
        self.network_ = nn.Sequential(backbone, coral_head).to(dev)

        levels = levels_from_labelbatch(torch.from_numpy(y_shifted), self.num_classes_).to(dev)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_shifted), levels),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_STATE),
        )
        opt = torch.optim.Adam(self.network_.parameters(), lr=self.lr)

        self.network_.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for X_b, _, lvl_b in loader:
                X_b, lvl_b = X_b.to(dev), lvl_b.to(dev)
                opt.zero_grad()
                logits = self.network_(X_b)
                loss = coral_loss(logits, lvl_b)
                loss.backward()
                opt.step()
                total += loss.item() * len(X_b)
            if epoch % 25 == 0 or epoch == 1:
                logger.info('CORALRegressor: epoch %d/%d — loss=%.6f', epoch, self.epochs, total / len(X_np))

        self.network_.eval()
        return self

    def predict(self, X):
        dev = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        self.network_.to(dev).eval()
        with torch.no_grad():
            logits = self.network_(torch.from_numpy(X_np).to(dev))
            probas = torch.sigmoid(logits).cpu().numpy()
        return probas.sum(axis=1) + self.y_min_

    def predict_proba(self, X):
        """Per-class probabilities from the CORAL cumulative logits (``sigmoid`` gives ``P(Y > k)``)."""
        dev = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        self.network_.to(dev).eval()
        with torch.no_grad():
            logits = self.network_(torch.from_numpy(X_np).to(dev))
            cum = torch.sigmoid(logits).cpu().numpy()
        return _cum_to_class_probs(cum)


class OrdinalMLPRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible ordinal MLP using cumulative logits with binary cross-entropy.

    Each of K-1 output nodes models ``P(Y > k)``. Trained with BCE on cumulative binary labels. Prediction is the
    expected rank: ``sum(sigmoid(logit_k))``.

    :param hidden_dims: Tuple of hidden layer widths (default: ``(64, 32)``).
    :param dropout: Dropout probability applied after each hidden layer (default: 0.3).
    :param epochs: Training epochs (default: 100).
    :param batch_size: Mini-batch size (default: 32).
    :param lr: Adam learning rate (default: 1e-3).
    :param device: ``'cpu'``, ``'cuda'``, or ``'auto'``.
    """

    def __init__(self, hidden_dims=(64, 32), dropout=0.3,
                 epochs=100, batch_size=32, lr=1e-3, device='auto'):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def fit(self, X, y):
        dev = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)

        self.num_classes_ = int(y_np.max() - y_np.min()) + 1
        self.y_min_ = int(y_np.min())
        self.classes_ = np.arange(self.y_min_, self.y_min_ + self.num_classes_)
        y_shifted = y_np - self.y_min_

        backbone = _build_backbone(X_np.shape[1], self.hidden_dims, self.dropout)
        head = nn.Linear(self.hidden_dims[-1], self.num_classes_ - 1)
        self.network_ = nn.Sequential(backbone, head).to(dev)

        # Cumulative binary labels: level_k = 1 if y > k
        levels = levels_from_labelbatch(torch.from_numpy(y_shifted), self.num_classes_).float().to(dev)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_np), levels),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_STATE),
        )
        opt = torch.optim.Adam(self.network_.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self.network_.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for X_b, lvl_b in loader:
                X_b, lvl_b = X_b.to(dev), lvl_b.to(dev)
                opt.zero_grad()
                loss = criterion(self.network_(X_b), lvl_b)
                loss.backward()
                opt.step()
                total += loss.item() * len(X_b)
            if epoch % 25 == 0 or epoch == 1:
                logger.info('OrdinalMLPRegressor: epoch %d/%d — bce=%.6f', epoch, self.epochs, total / len(X_np))

        self.network_.eval()
        return self

    def predict(self, X):
        dev = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        self.network_.to(dev).eval()
        with torch.no_grad():
            logits = self.network_(torch.from_numpy(X_np).to(dev))
            probas = torch.sigmoid(logits).cpu().numpy()
        return probas.sum(axis=1) + self.y_min_

    def predict_proba(self, X):
        """Per-class probabilities from the cumulative logits (``sigmoid`` gives ``P(Y > k)``)."""
        dev = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        self.network_.to(dev).eval()
        with torch.no_grad():
            logits = self.network_(torch.from_numpy(X_np).to(dev))
            cum = torch.sigmoid(logits).cpu().numpy()
        return _cum_to_class_probs(cum)


class CORNRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible CORN ordinal regressor (Shi et al. 2021).

    Conditional Ordinal Regression: K-1 independent binary tasks where task k models ``P(Y > k | Y >= k)``.
    Prediction is the expected rank derived from conditional probabilities.

    :param hidden_dims: Tuple of hidden layer widths (default: ``(64, 32)``).
    :param dropout: Dropout probability applied after each hidden layer (default: 0.3).
    :param epochs: Training epochs (default: 100).
    :param batch_size: Mini-batch size (default: 32).
    :param lr: Adam learning rate (default: 1e-3).
    :param device: ``'cpu'``, ``'cuda'``, or ``'auto'``.
    """

    def __init__(self, hidden_dims=(64, 32), dropout=0.3,
                 epochs=100, batch_size=32, lr=1e-3, device='auto'):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def fit(self, X, y):
        dev = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)

        self.num_classes_ = int(y_np.max() - y_np.min()) + 1
        self.y_min_ = int(y_np.min())
        self.classes_ = np.arange(self.y_min_, self.y_min_ + self.num_classes_)
        y_shifted = y_np - self.y_min_

        backbone = _build_backbone(X_np.shape[1], self.hidden_dims, self.dropout)
        # CORN uses K-1 independent output nodes (no weight sharing)
        head = nn.Linear(self.hidden_dims[-1], self.num_classes_ - 1)
        self.network_ = nn.Sequential(backbone, head).to(dev)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_shifted)),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_STATE),
        )
        opt = torch.optim.Adam(self.network_.parameters(), lr=self.lr)

        self.network_.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(dev), y_b.to(dev)
                opt.zero_grad()
                logits = self.network_(X_b)
                loss = corn_loss(logits, y_b, self.num_classes_)
                loss.backward()
                opt.step()
                total += loss.item() * len(X_b)
            if epoch % 25 == 0 or epoch == 1:
                logger.info('CORNRegressor: epoch %d/%d — loss=%.6f', epoch, self.epochs, total / len(X_np))

        self.network_.eval()
        return self

    def predict(self, X):
        dev = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        self.network_.to(dev).eval()
        with torch.no_grad():
            logits = self.network_(torch.from_numpy(X_np).to(dev))
            labels = corn_label_from_logits(logits).cpu().numpy()
        return labels.astype(float) + self.y_min_

    def predict_proba(self, X):
        """Per-class probabilities from the CORN conditionals. The conditional ``sigmoid(logit_k) = P(Y > k | Y >= k)``
        combine into the cumulative ``P(Y > k) = prod_{j<=k} sigmoid(logit_j)``."""
        dev = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        self.network_.to(dev).eval()
        with torch.no_grad():
            logits = self.network_(torch.from_numpy(X_np).to(dev))
            cum = torch.cumprod(torch.sigmoid(logits), dim=1).cpu().numpy()
        return _cum_to_class_probs(cum)


# Full ordinal model set — used with raw CpG features (after filter/reducer).
ORDINAL_MODELS = {
    'ridge':            Ridge(alpha=1.0, random_state=RANDOM_STATE),
    'lasso':            Lasso(alpha=0.1, random_state=RANDOM_STATE),
    'elasticnet':       ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE),
    'svr':              SVR(kernel='rbf'),
    'random_forest':    RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE),
    'xgboost':          XGBRegressor(max_depth=5, n_estimators=1000, random_state=RANDOM_STATE),
    'coral':            CORALRegressor(hidden_dims=(64, 32), dropout=0.3, epochs=30, batch_size=32),
    'ordinal_mlp':      OrdinalMLPRegressor(hidden_dims=(64, 32), dropout=0.3, epochs=30, batch_size=32),
    'corn':             CORNRegressor(hidden_dims=(64, 32), dropout=0.3, epochs=30, batch_size=32),
    'mord_ridge':       mord.OrdinalRidge(alpha=1.0),
    'mord_logistic_at': mord.LogisticAT(alpha=1.0),
    'ogboost':          GradientBoostingOrdinal(n_estimators=50, random_state=RANDOM_STATE),
}


# Same model set is appropriate for the low-dim (markers/stacked) feature case — keep them aliased so the registry is
# symmetric with the classification side and easy to specialise later if hyperparameters need to diverge.
MARKER_ORDINAL_MODELS = dict(ORDINAL_MODELS)
