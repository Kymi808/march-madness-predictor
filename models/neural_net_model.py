"""PyTorch neural network model with configurable architecture and early stopping."""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED, CV_FOLDS
from models.base_model import BaseMarchMadnessModel


# ------------------------------------------------------------------
# PyTorch module
# ------------------------------------------------------------------

class MarchMadnessNet(nn.Module):
    """Feedforward network with configurable hidden layers, dropout, and
    optional batch normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ------------------------------------------------------------------
# Sklearn-compatible wrapper
# ------------------------------------------------------------------

class NeuralNetModel(BaseMarchMadnessModel):
    """Wraps :class:`MarchMadnessNet` to conform to the
    :class:`BaseMarchMadnessModel` interface.  Handles training loops,
    early stopping, and automatic GPU usage when available."""

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 200,
        patience: int = 15,
        random_seed: int = RANDOM_SEED,
        cv_folds: int = CV_FOLDS,
    ):
        super().__init__(name="NeuralNetModel", random_seed=random_seed)
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.cv_folds = cv_folds

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self._net: MarchMadnessNet | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetModel":
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        X_scaled = self.scaler.fit_transform(X)

        # Hold out a validation split for early stopping
        n_val = max(1, int(len(X_scaled) * 0.15))
        indices = np.random.permutation(len(X_scaled))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.long)
        X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._net = MarchMadnessNet(
            input_dim=X_scaled.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            self._net.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            val_loss = self._compute_loss(X_val, y_val, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        # Restore best weights
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self.model = self._net  # expose for base-class helpers
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self._net.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self._net(X_t)
            probas = torch.softmax(logits, dim=1).cpu().numpy()
        return probas

    # ------------------------------------------------------------------
    # Evaluation (override to avoid sklearn cross_val_score)
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        self._check_fitted()

        preds = self.predict(X)
        probas = self.predict_proba(X)

        metrics: dict = {
            "accuracy": accuracy_score(y, preds),
            "log_loss": log_loss(y, probas[:, 1]),
            "roc_auc": roc_auc_score(y, probas[:, 1]),
            "classification_report": classification_report(y, preds),
        }

        # Manual cross-validation (retrain per fold)
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
        )
        fold_accuracies = []
        for train_idx, val_idx in cv.split(X, y):
            fold_model = NeuralNetModel(
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                use_batch_norm=self.use_batch_norm,
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                epochs=self.epochs,
                patience=self.patience,
                random_seed=self.random_seed,
            )
            fold_model.train(X[train_idx], y[train_idx])
            fold_preds = fold_model.predict(X[val_idx])
            fold_accuracies.append(accuracy_score(y[val_idx], fold_preds))

        cv_scores = np.array(fold_accuracies)
        metrics["cv_mean_accuracy"] = cv_scores.mean()
        metrics["cv_std_accuracy"] = cv_scores.std()
        metrics["cv_scores"] = cv_scores
        return metrics

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> np.ndarray | None:
        """Approximate feature importance using the absolute weights of the
        first linear layer."""
        self._check_fitted()
        first_linear = None
        for module in self._net.network:
            if isinstance(module, nn.Linear):
                first_linear = module
                break
        if first_linear is None:
            return None
        return torch.abs(first_linear.weight).sum(dim=0).detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> float:
        self._net.eval()
        X = X.to(self.device)
        y = y.to(self.device)
        logits = self._net(X)
        return criterion(logits, y).item()
