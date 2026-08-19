import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

from ...config import RANDOM_STATE
from .._utils import resolve_device

logger = logging.getLogger(__name__)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible MLP classifier backed by PyTorch.

    Architecture: Input → [Linear → BatchNorm1d → ReLU → Dropout] × N → Linear(n_classes)

    Input dimension is inferred at fit() time, so it adapts to whatever filter/reducer
    precedes it in the pipeline. Compatible with sklearn's clone() — all state is stored
    in fitted attributes (suffixed with _) and no mutable defaults are used in __init__.

    :param hidden_dims:  Tuple of hidden layer widths (default: (256, 128)).
    :param dropout:      Dropout probability applied after each hidden layer (default: 0.3).
    :param epochs:       Training epochs (default: 50).
    :param batch_size:   Mini-batch size (default: 32).
    :param lr:           Adam learning rate (default: 1e-3).
    :param device:       'cpu', 'cuda', or 'auto' — 'auto' picks cuda if available (default: 'auto').
    """

    def __init__(self, hidden_dims=(256, 128), dropout=0.3,
                 epochs=50, batch_size=32, lr=1e-3, device='auto'):
        self.hidden_dims = hidden_dims
        self.dropout     = dropout
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.device      = device

    def _build_network(self, n_features, n_classes):
        layers = []
        in_dim = n_features
        for out_dim in self.hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        dev    = resolve_device(self.device)
        X_np   = np.asarray(X, dtype=np.float32)
        y_np   = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y_np)
        n_features    = X_np.shape[1]
        n_classes     = len(self.classes_)
        # Re-encode labels to contiguous 0-based indices for CrossEntropyLoss
        y_enc = np.searchsorted(self.classes_, y_np).astype(np.int64)

        self.network_ = self._build_network(n_features, n_classes).to(dev)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_enc)),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_STATE),
        )
        opt       = torch.optim.Adam(self.network_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.network_.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(dev), y_batch.to(dev)
                opt.zero_grad()
                loss = criterion(self.network_(X_batch), y_batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(X_batch)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("MLPClassifier: epoch %d/%d — ce=%.6f",
                            epoch, self.epochs, total / len(X_np))

        self.network_.eval()
        logger.info("MLPClassifier: trained (%d, %d) → %d classes on %s",
                    X_np.shape[0], n_features, n_classes, dev)
        return self

    def predict_proba(self, X):
        dev  = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        self.network_.to(dev)
        self.network_.eval()
        with torch.no_grad():
            logits = self.network_(torch.from_numpy(X_np).to(dev))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class CNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible 1D CNN classifier backed by PyTorch.

    Probes are sorted by genomic position (chr, coordinate) derived from DataFrame column names
    in 'chrN-pos' format before being fed into the network as a 1D spatial signal. This lets
    local convolutional filters detect methylation patterns within genomic blocks (promoters,
    CpG islands, regulatory regions).

    Architecture:
        Input: (batch, 1, n_probes)  ← probes sorted by (chr, pos)
        → [Conv1d → BatchNorm1d → ReLU] × len(channels)
        → AdaptiveAvgPool1d(1)       ← handles any probe count from any filter/reducer
        → Flatten → Linear → ReLU → Dropout → Linear(n_classes)

    When X is not a DataFrame or columns are not in genomic format, falls back to as-is order.

    :param channels:     Tuple of conv output channel widths (default: (32, 64, 128)).
    :param kernel_size:  Conv1d kernel width (default: 11).
    :param fc_dim:       Fully-connected hidden dim after pooling (default: 64).
    :param dropout:      Dropout probability in the FC layer (default: 0.3).
    :param epochs:       Training epochs (default: 50).
    :param batch_size:   Mini-batch size (default: 16).
    :param lr:           Adam learning rate (default: 1e-3).
    :param device:       'cpu', 'cuda', or 'auto' — 'auto' picks cuda if available (default: 'auto').
    """

    def __init__(self, channels=(32, 64, 128), kernel_size=11, fc_dim=64,
                 dropout=0.3, epochs=50, batch_size=16, lr=1e-3, device='auto'):
        self.channels    = channels
        self.kernel_size = kernel_size
        self.fc_dim      = fc_dim
        self.dropout     = dropout
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.device      = device

    def _genomic_sort(self, X):
        """Sort DataFrame columns by (chr_number, position) and cache the order."""
        if not isinstance(X, pd.DataFrame):
            return np.ascontiguousarray(X, dtype=np.float32)

        def _key(col):
            chr_str, pos = col.rsplit('-', 1)
            n = chr_str[3:]  # strip 'chr'
            order = {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}.get(n, int(n) if n.isdigit() else 26)
            return (order, int(pos))

        try:
            sorted_cols = sorted(X.columns, key=_key)
            self.feature_names_in_ = sorted_cols
            return np.ascontiguousarray(X[sorted_cols], dtype=np.float32)
        except (ValueError, AttributeError):
            return np.ascontiguousarray(X, dtype=np.float32)

    def _build_network(self, n_classes):
        layers = []
        in_ch = 1
        for out_ch in self.channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=self.kernel_size,
                          padding=self.kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        layers += [
            nn.Linear(in_ch, self.fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc_dim, n_classes),
        ]
        return nn.Sequential(*layers)

    def fit(self, X, y):
        dev  = resolve_device(self.device)
        X_np = self._genomic_sort(X)
        y_np = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y_np)
        n_classes     = len(self.classes_)
        n_probes      = X_np.shape[1]
        y_enc = np.searchsorted(self.classes_, y_np).astype(np.int64)

        # (batch, 1, n_probes) — channel dim for Conv1d
        X_t = torch.from_numpy(X_np).unsqueeze(1)
        y_t = torch.from_numpy(y_enc)

        self.network_ = self._build_network(n_classes).to(dev)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_STATE),
        )
        opt       = torch.optim.Adam(self.network_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.network_.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(dev), y_batch.to(dev)
                opt.zero_grad()
                loss = criterion(self.network_(X_batch), y_batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(X_batch)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("CNNClassifier: epoch %d/%d — ce=%.6f",
                            epoch, self.epochs, total / len(X_np))

        self.network_.eval()
        logger.info("CNNClassifier: trained (%d, %d) → %d classes on %s",
                    X_np.shape[0], n_probes, n_classes, dev)
        return self

    def predict_proba(self, X):
        dev = resolve_device(self.device)
        # Re-sort using stored column order if available, else sort fresh
        if isinstance(X, pd.DataFrame) and hasattr(self, 'feature_names_in_'):
            X_np = np.ascontiguousarray(X[self.feature_names_in_], dtype=np.float32)
        else:
            X_np = self._genomic_sort(X)
        X_t = torch.from_numpy(X_np).unsqueeze(1)
        self.network_.to(dev)
        self.network_.eval()
        with torch.no_grad():
            logits = self.network_(X_t.to(dev))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
