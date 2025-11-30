"""
Training utilities for ad prediction models.

Provides:
- Trainer class for model training
- Learning rate schedulers
- Early stopping
- Metrics tracking
- Model checkpointing
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader

from liteads.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"  # adam, adamw, sgd

    # Scheduler
    scheduler: str = "cosine"  # cosine, plateau, none
    warmup_steps: int = 100
    min_lr: float = 1e-6

    # Training
    num_epochs: int = 10
    gradient_clip: float = 1.0
    accumulation_steps: int = 1

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0001

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    save_every_n_epochs: int = 1

    # Logging
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 500

    # Device
    device: str = "auto"  # auto, cpu, cuda, mps


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_auc: list[float] = field(default_factory=list)
    val_auc: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)

    best_val_loss: float = float("inf")
    best_val_auc: float = 0.0
    best_epoch: int = 0


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for metrics like AUC
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


class Trainer:
    """
    Model trainer with training loop, evaluation, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        loss_fn: nn.Module | None = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            config: Training configuration
            loss_fn: Loss function (default: BCELoss)
        """
        self.config = config or TrainingConfig()
        self.model = model
        self.loss_fn = loss_fn or nn.BCELoss()

        # Setup device
        self.device = self._get_device()
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler (created in fit())
        self.scheduler: _LRScheduler | None = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode="min",
        )

        # Metrics
        self.metrics = TrainingMetrics()

        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.current_epoch = 0
        self.global_step = 0

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer."""
        params = self.model.parameters()

        if self.config.optimizer == "adam":
            return Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            return AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self, num_training_steps: int) -> _LRScheduler | None:
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=self.config.min_lr,
            )
        return None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        callbacks: list[Callable] | None = None,
    ) -> TrainingMetrics:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            callbacks: Optional callbacks called after each epoch

        Returns:
            Training metrics
        """
        num_training_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = self._create_scheduler(num_training_steps)

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"Validation samples: {len(val_loader.dataset)}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training
            train_loss = self._train_epoch(train_loader)
            self.metrics.train_loss.append(train_loss)

            # Validation
            val_loss = None
            if val_loader:
                val_loss, val_auc = self._validate(val_loader)
                self.metrics.val_loss.append(val_loss)
                self.metrics.val_auc.append(val_auc)

                # Track best
                if val_loss < self.metrics.best_val_loss:
                    self.metrics.best_val_loss = val_loss
                    self.metrics.best_val_auc = val_auc
                    self.metrics.best_epoch = epoch

                    if self.config.save_best_only:
                        self._save_checkpoint("best.pt")

            epoch_time = time.time() - epoch_start_time
            self.metrics.epoch_times.append(epoch_time)

            # Logging
            log_msg = f"Epoch {epoch + 1}/{self.config.num_epochs} - "
            log_msg += f"train_loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" - val_loss: {val_loss:.4f}"
                log_msg += f" - val_auc: {val_auc:.4f}"
            log_msg += f" - time: {epoch_time:.1f}s"
            logger.info(log_msg)

            # Checkpointing
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch)

            # Early stopping
            if val_loss is not None and self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Final summary
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.metrics.best_val_loss:.4f} at epoch {self.metrics.best_epoch + 1}")
        logger.info(f"Best validation AUC: {self.metrics.best_val_auc:.4f}")

        return self.metrics

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            sparse_features = batch["sparse_features"].to(self.device)
            dense_features = batch["dense_features"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                sparse_features=sparse_features,
                dense_features=dense_features,
            )

            # Get prediction (first task or ctr)
            if isinstance(outputs, dict):
                pred = outputs.get("ctr", outputs.get("task_0", list(outputs.values())[0]))
            else:
                pred = outputs

            # Calculate loss
            loss = self.loss_fn(pred.squeeze(), labels.squeeze())

            # Add regularization
            if hasattr(self.model, "get_regularization_loss"):
                reg_loss = self.model.get_regularization_loss()
                loss = loss + reg_loss

            # Gradient accumulation
            loss = loss / self.config.accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler and not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()

            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                logger.debug(
                    f"Step {self.global_step} - loss: {avg_loss:.4f} - lr: {lr:.2e}"
                )

        return total_loss / num_batches

    def _validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                sparse_features = batch["sparse_features"].to(self.device)
                dense_features = batch["dense_features"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    sparse_features=sparse_features,
                    dense_features=dense_features,
                )

                if isinstance(outputs, dict):
                    pred = outputs.get("ctr", outputs.get("task_0", list(outputs.values())[0]))
                else:
                    pred = outputs

                loss = self.loss_fn(pred.squeeze(), labels.squeeze())
                total_loss += loss.item()
                num_batches += 1

                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / num_batches

        # Calculate AUC
        auc = self._calculate_auc(all_labels, all_preds)

        # Update scheduler if plateau
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        return avg_loss, auc

    def _calculate_auc(self, labels: list, predictions: list) -> float:
        """Calculate AUC-ROC score."""
        from sklearn.metrics import roc_auc_score

        try:
            return roc_auc_score(labels, predictions)
        except ValueError:
            # All labels are the same class
            return 0.5

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics,
            "config": self.config,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.metrics = checkpoint["metrics"]

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {path}")

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Generate predictions.

        Args:
            data_loader: Data loader for prediction

        Returns:
            Numpy array of predictions
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in data_loader:
                sparse_features = batch["sparse_features"].to(self.device)
                dense_features = batch["dense_features"].to(self.device)

                outputs = self.model(
                    sparse_features=sparse_features,
                    dense_features=dense_features,
                )

                if isinstance(outputs, dict):
                    pred = outputs.get("ctr", outputs.get("task_0", list(outputs.values())[0]))
                else:
                    pred = outputs

                all_preds.extend(pred.cpu().numpy().flatten())

        return np.array(all_preds)

    def export_onnx(self, path: str | Path, example_inputs: dict[str, torch.Tensor]) -> None:
        """
        Export model to ONNX format.

        Args:
            path: Output path for ONNX model
            example_inputs: Example inputs for tracing
        """
        self.model.eval()

        torch.onnx.export(
            self.model,
            (
                example_inputs["sparse_features"],
                example_inputs["dense_features"],
            ),
            path,
            input_names=["sparse_features", "dense_features"],
            output_names=["prediction"],
            dynamic_axes={
                "sparse_features": {0: "batch_size"},
                "dense_features": {0: "batch_size"},
                "prediction": {0: "batch_size"},
            },
            opset_version=14,
        )
        logger.info(f"Exported model to ONNX: {path}")
