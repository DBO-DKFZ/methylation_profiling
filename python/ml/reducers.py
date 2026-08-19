import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import RANDOM_STATE
from ._utils import load_chrom_hmm_lookup, resolve_device

logger = logging.getLogger(__name__)


class PCAReducer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper around PCA for use as a post-filter dimensionality reducer.

    Fits PCA on training data and projects both train and validation folds into the
    learned latent space. Compatible with sklearn's clone() so a fresh instance is
    created per cross-validation fold without data leakage.

    :param n_components: Number of principal components to retain (default: 50).
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components

    def fit(self, X, y=None):
        n_comp = min(self.n_components, X.shape[0], X.shape[1])
        if n_comp < self.n_components:
            logger.info(
                "PCAReducer: requested n_components=%d exceeds min(n_samples=%d, n_features=%d); using %d.",
                self.n_components, X.shape[0], X.shape[1], n_comp,
            )
        self.pca_ = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        self.pca_.fit(X)
        logger.info(
            "PCAReducer: fitted on (%d, %d), explained variance ratio sum = %.3f",
            X.shape[0], X.shape[1], self.pca_.explained_variance_ratio_.sum(),
        )
        return self

    def transform(self, X):
        return self.pca_.transform(X)


class PLSReducer(BaseEstimator, TransformerMixin):
    """
    Supervised dimensionality reducer based on PLS.

    For classification (``task='classification'``, default), fits PLSRegression on (X, one-hot(y)) — i.e. PLS-DA —
    and projects X into the n_components latent space that maximises covariance with the class indicators. For
    ordinal regression (``task='ordinal'``), passes the continuous y directly to PLSRegression as a
    single-column response. Compatible with sklearn's clone() so a fresh instance is created per CV fold.

    :param n_components: Number of PLS components to retain (default: 50).
    :param task: ``'classification'`` (default) or ``'ordinal'``.
    """

    def __init__(self, n_components: int = 50, task: str = 'classification'):
        self.n_components = n_components
        self.task = task

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("PLSReducer requires y at fit time (supervised).")
        if self.task == 'ordinal':
            Y = np.asarray(y, dtype=float).reshape(-1, 1)
            self.lb_ = None
        else:
            self.lb_  = LabelBinarizer().fit(y)
            Y        = self.lb_.transform(y)
            if Y.shape[1] == 1:  # binary case → expand to 2 columns
                Y = np.hstack([1 - Y, Y])
        n_comp   = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.pls_ = PLSRegression(n_components=n_comp, scale=False)
        self.pls_.fit(X, Y)
        logger.info(
            "PLSReducer: fitted on (%d, %d) with Y shape %s → %d components (task=%s)",
            X.shape[0], X.shape[1], Y.shape, n_comp, self.task,
        )
        return self

    def transform(self, X):
        return self.pls_.transform(X)


class ChromHMMReducer(BaseEstimator, TransformerMixin):
    """
    Aggregates CpG beta values by ChromHMM state.

    For each ChromHMM state observed among the input CpGs, computes the mean beta
    value across all member probes, reducing (n_samples, n_cpgs) → (n_samples, n_states).
    The state mapping is loaded from the annotation file configured at paths.chrom_hmm.
    CpGs not present in the annotation are silently excluded from all states.

    Compatible with sklearn's clone() so a fresh instance is created per CV fold.

    :param agg: Aggregation function name accepted by DataFrame (default: 'mean').
    """

    def __init__(self, agg: str = 'mean'):
        self.agg = agg

    def fit(self, X, y=None):
        # X: DataFrame (n_samples, n_cpgs), columns = "seqnames-start" CpG IDs
        lookup = load_chrom_hmm_lookup()

        # Only keep CpGs that are present in the annotation and have a valid state
        self.cpg_to_state_ = {
            cpg: lookup[cpg] for cpg in X.columns
            if cpg in lookup and pd.notna(lookup[cpg])
        }
        self.states_ = sorted(set(self.cpg_to_state_.values()))

        n_mapped = len(self.cpg_to_state_)
        n_total  = len(X.columns)
        logger.info(
            "ChromHMMReducer: %d / %d CpGs mapped to %d states",
            n_mapped, n_total, len(self.states_),
        )
        if n_mapped == 0:
            raise ValueError("ChromHMMReducer: no input CpGs found in the ChromHMM annotation.")
        return self

    def transform(self, X):
        # Group columns by state, aggregate, return numpy array (states in sorted order)
        state_to_cpgs = {}
        for cpg, state in self.cpg_to_state_.items():
            state_to_cpgs.setdefault(state, []).append(cpg)

        cols = {
            state: getattr(X[cpgs], self.agg)(axis=1).values
            for state, cpgs in state_to_cpgs.items()
        }
        return pd.DataFrame(cols, index=X.index)[self.states_].values


class _BaseAutoEncoderReducer(BaseEstimator, TransformerMixin):
    """Shared logic for all AE variants."""

    def __init__(self, latent_dim=32, hidden_dims=(256,),
                 epochs=50, batch_size=32, lr=1e-3, device='auto'):
        self.latent_dim  = latent_dim
        self.hidden_dims = hidden_dims
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.device      = device

    def _encoder_layers(self, n_features):
        dims = [n_features, *self.hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], self.latent_dim))
        return layers

    def _decoder_layers(self, n_features):
        dims = [self.latent_dim, *reversed(self.hidden_dims)]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], n_features), nn.Sigmoid()]
        return layers

    def transform(self, X):
        """Encodes X using the fitted encoder_ (works for AE, DAE, Supervised AE)."""
        dev  = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        self.encoder_.to(dev)
        self.encoder_.eval()
        with torch.no_grad():
            latent = self.encoder_(torch.from_numpy(X_np).to(dev)).cpu().numpy()
        return latent


class AutoEncoderReducer(_BaseAutoEncoderReducer):
    """
    Sklearn-compatible dimensionality reducer based on a PyTorch fully-connected autoencoder.

    Trains an encoder-decoder network on training data using BCE reconstruction loss
    (appropriate for bimodal beta values in [0, 1]).
    transform() returns the encoder's latent activations (shape: n_samples × latent_dim).

    :param latent_dim:   Size of the bottleneck layer (default: 32).
    :param hidden_dims:  Tuple of hidden layer widths for the encoder;
                         mirrored for the decoder (default: (256,)).
    :param epochs:       Training epochs (default: 50).
    :param batch_size:   Mini-batch size (default: 32).
    :param lr:           Adam learning rate (default: 1e-3).
    :param device:       'cpu', 'cuda', or 'auto' — 'auto' picks CUDA if available (default: 'auto').
    """

    def fit(self, X, y=None):
        dev    = resolve_device(self.device)
        X_np   = np.asarray(X, dtype=np.float32)
        n_feat = X_np.shape[1]

        self.encoder_ = nn.Sequential(*self._encoder_layers(n_feat))
        decoder       = nn.Sequential(*self._decoder_layers(n_feat))
        model         = nn.Sequential(self.encoder_, decoder).to(dev)

        loader    = DataLoader(TensorDataset(torch.from_numpy(X_np)),
                               batch_size=self.batch_size, shuffle=True,
                               generator=torch.Generator().manual_seed(RANDOM_STATE))
        opt       = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for (batch,) in loader:
                batch = batch.to(dev)
                opt.zero_grad()
                loss = criterion(model(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("AutoEncoderReducer: epoch %d/%d — bce=%.6f",
                            epoch, self.epochs, total / len(X_np))

        model.eval()
        logger.info("AutoEncoderReducer: trained (%d,%d) → latent=%d on %s",
                    X_np.shape[0], n_feat, self.latent_dim, dev)
        return self


class DenoisingAutoEncoderReducer(_BaseAutoEncoderReducer):
    """
    AE that reconstructs clean inputs from corrupted ones (Gaussian noise).

    Adds Gaussian noise to inputs during training; network and transform are otherwise
    identical to the vanilla AE.

    :param latent_dim:   Size of the bottleneck layer (default: 32).
    :param hidden_dims:  Tuple of hidden layer widths for the encoder (default: (256,)).
    :param epochs:       Training epochs (default: 50).
    :param batch_size:   Mini-batch size (default: 32).
    :param lr:           Adam learning rate (default: 1e-3).
    :param device:       'cpu', 'cuda', or 'auto' (default: 'auto').
    :param noise_std:    Standard deviation of the additive Gaussian noise (default: 0.05).
    """

    def __init__(self, latent_dim=32, hidden_dims=(256,),
                 epochs=50, batch_size=32, lr=1e-3, device='auto', noise_std=0.05):
        super().__init__(latent_dim, hidden_dims, epochs, batch_size, lr, device)
        self.noise_std = noise_std

    def fit(self, X, y=None):
        dev    = resolve_device(self.device)
        X_np   = np.asarray(X, dtype=np.float32)
        n_feat = X_np.shape[1]

        self.encoder_ = nn.Sequential(*self._encoder_layers(n_feat))
        decoder       = nn.Sequential(*self._decoder_layers(n_feat))
        model         = nn.Sequential(self.encoder_, decoder).to(dev)

        loader    = DataLoader(TensorDataset(torch.from_numpy(X_np)),
                               batch_size=self.batch_size, shuffle=True,
                               generator=torch.Generator().manual_seed(RANDOM_STATE))
        opt       = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for (batch,) in loader:
                batch  = batch.to(dev)
                noisy  = torch.clamp(batch + self.noise_std * torch.randn_like(batch), 0.0, 1.0)
                opt.zero_grad()
                loss   = criterion(model(noisy), batch)   # target = clean batch
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("DenoisingAutoEncoderReducer: epoch %d/%d — bce=%.6f",
                            epoch, self.epochs, total / len(X_np))

        model.eval()
        logger.info("DenoisingAutoEncoderReducer: trained (%d,%d) → latent=%d on %s",
                    X_np.shape[0], n_feat, self.latent_dim, dev)
        return self


class _VAEEncoder(nn.Module):
    """Encoder that emits (mu, log_var) for a VAE."""

    def __init__(self, body: nn.Sequential, last_hidden: int, latent_dim: int):
        super().__init__()
        self.body        = body
        self.mu_head     = nn.Linear(last_hidden, latent_dim)
        self.logvar_head = nn.Linear(last_hidden, latent_dim)

    def forward(self, x):
        h = self.body(x)
        return self.mu_head(h), self.logvar_head(h)


class VariationalAutoEncoderReducer(_BaseAutoEncoderReducer):
    """
    VAE: ELBO = BCE(recon, x) + beta * KL(q(z|x) || N(0,I)).

    Encoder outputs (mu, log_var); reparameterisation trick is used during training.
    transform() returns mu (deterministic inference — no sampling).

    :param latent_dim:   Size of the bottleneck layer (default: 32).
    :param hidden_dims:  Tuple of hidden layer widths for the encoder (default: (256,)).
    :param epochs:       Training epochs (default: 50).
    :param batch_size:   Mini-batch size (default: 32).
    :param lr:           Adam learning rate (default: 1e-3).
    :param device:       'cpu', 'cuda', or 'auto' (default: 'auto').
    :param beta:         Weight on the KL term (default: 1.0).
    """

    def __init__(self, latent_dim=32, hidden_dims=(256,),
                 epochs=50, batch_size=32, lr=1e-3, device='auto', beta=1.0):
        super().__init__(latent_dim, hidden_dims, epochs, batch_size, lr, device)
        self.beta = beta

    @staticmethod
    def _reparameterise(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def fit(self, X, y=None):
        dev    = resolve_device(self.device)
        X_np   = np.asarray(X, dtype=np.float32)
        n_feat = X_np.shape[1]

        # Build encoder body (all layers up to last hidden dim, excluding latent projection)
        dims = [n_feat, *self.hidden_dims]
        body_layers = []
        for i in range(len(dims) - 1):
            body_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        body = nn.Sequential(*body_layers)

        self.encoder_ = _VAEEncoder(body, dims[-1], self.latent_dim)
        decoder       = nn.Sequential(*self._decoder_layers(n_feat)[:-1])   # drop trailing Sigmoid
        self.encoder_.to(dev)
        decoder.to(dev)

        params  = list(self.encoder_.parameters()) + list(decoder.parameters())
        opt     = torch.optim.Adam(params, lr=self.lr)
        bce     = nn.BCEWithLogitsLoss(reduction='sum')
        loader  = DataLoader(TensorDataset(torch.from_numpy(X_np)),
                             batch_size=self.batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(RANDOM_STATE))

        self.encoder_.train()
        decoder.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for (batch,) in loader:
                batch      = batch.to(dev)
                mu, lv     = self.encoder_(batch)
                lv         = torch.clamp(lv, -10, 10)   # prevent exp(lv) explosion in KL term
                z          = self._reparameterise(mu, lv)
                recon      = decoder(z)
                loss_recon = bce(recon, batch)
                loss_kl    = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
                loss       = (loss_recon + self.beta * loss_kl) / len(batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("VariationalAutoEncoderReducer: epoch %d/%d — elbo=%.6f",
                            epoch, self.epochs, total / len(X_np))

        self.encoder_.eval()
        logger.info("VariationalAutoEncoderReducer: trained (%d,%d) → latent=%d on %s",
                    X_np.shape[0], n_feat, self.latent_dim, dev)
        return self

    def transform(self, X):
        """Returns mu (deterministic); no sampling at inference time."""
        dev  = resolve_device(self.device)
        X_np = np.asarray(X, dtype=np.float32)
        self.encoder_.to(dev)
        self.encoder_.eval()
        with torch.no_grad():
            mu, _ = self.encoder_(torch.from_numpy(X_np).to(dev))
            return mu.cpu().numpy()


class SupervisedAutoEncoderReducer(_BaseAutoEncoderReducer):
    """
    AE with a joint classification head on the latent space.

    Loss = BCE(recon, x) + lambda_cls * CrossEntropy(head(z), y)

    Requires y to be passed to fit(). transform() still returns the latent z
    for use as features by the downstream model.

    :param latent_dim:   Size of the bottleneck layer (default: 32).
    :param hidden_dims:  Tuple of hidden layer widths for the encoder (default: (256,)).
    :param epochs:       Training epochs (default: 50).
    :param batch_size:   Mini-batch size (default: 32).
    :param lr:           Adam learning rate (default: 1e-3).
    :param device:       'cpu', 'cuda', or 'auto' (default: 'auto').
    :param n_classes:    Number of target classes (default: 3).
    :param lambda_cls:   Weight on the classification loss term (default: 1.0).
    """

    def __init__(self, latent_dim=32, hidden_dims=(256,),
                 epochs=50, batch_size=32, lr=1e-3, device='auto',
                 n_classes=3, lambda_cls=1.0):
        super().__init__(latent_dim, hidden_dims, epochs, batch_size, lr, device)
        self.n_classes  = n_classes
        self.lambda_cls = lambda_cls

    def fit(self, X, y=None):
        dev    = resolve_device(self.device)
        X_np   = np.asarray(X, dtype=np.float32)
        y_np   = np.asarray(y, dtype=np.int64)
        n_feat = X_np.shape[1]

        self.encoder_  = nn.Sequential(*self._encoder_layers(n_feat))
        decoder        = nn.Sequential(*self._decoder_layers(n_feat))
        self.cls_head_ = nn.Linear(self.latent_dim, self.n_classes)

        self.encoder_.to(dev)
        decoder.to(dev)
        self.cls_head_.to(dev)

        params = (list(self.encoder_.parameters()) +
                  list(decoder.parameters()) +
                  list(self.cls_head_.parameters()))
        opt = torch.optim.Adam(params, lr=self.lr)
        bce = nn.BCELoss()
        ce  = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np))
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(RANDOM_STATE))

        self.encoder_.train()
        decoder.train()
        self.cls_head_.train()
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(dev)
                batch_y = batch_y.to(dev)
                opt.zero_grad()
                z      = self.encoder_(batch_x)
                recon  = decoder(z)
                logits = self.cls_head_(z)
                loss   = bce(recon, batch_x) + self.lambda_cls * ce(logits, batch_y)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch_x)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("SupervisedAutoEncoderReducer: epoch %d/%d — loss=%.6f",
                            epoch, self.epochs, total / len(X_np))

        self.encoder_.eval()
        logger.info("SupervisedAutoEncoderReducer: trained (%d,%d) → latent=%d on %s",
                    X_np.shape[0], n_feat, self.latent_dim, dev)
        return self
