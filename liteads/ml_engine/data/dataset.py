"""
Dataset classes for ad prediction models.

Provides PyTorch datasets and data loaders for training and inference.
"""

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from liteads.common.logger import get_logger
from liteads.ml_engine.features.builder import FeatureBuilder, ModelInputs

logger = get_logger(__name__)


class AdDataset(Dataset):
    """
    PyTorch Dataset for ad data.

    Handles pre-transformed data stored as ModelInputs.
    """

    def __init__(self, inputs: ModelInputs):
        """
        Initialize dataset from ModelInputs.

        Args:
            inputs: Pre-transformed model inputs
        """
        self.sparse_features = inputs.sparse_features
        self.dense_features = inputs.dense_features
        self.sequence_features = inputs.sequence_features
        self.labels = inputs.labels

        self._length = self.sparse_features.size(0)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample."""
        sample = {
            "sparse_features": self.sparse_features[idx],
            "dense_features": self.dense_features[idx],
        }

        # Handle sequence features (need special handling due to variable length)
        # For now, return placeholder - actual batching handled in collate_fn
        sample["sequence_idx"] = idx

        if self.labels is not None:
            sample["labels"] = self.labels[idx]

        return sample


class StreamingAdDataset(IterableDataset):
    """
    Streaming dataset for large data files.

    Reads data in chunks to handle datasets that don't fit in memory.
    """

    def __init__(
        self,
        data_path: str | Path,
        feature_builder: FeatureBuilder,
        chunk_size: int = 10000,
        label_cols: list[str] | None = None,
        shuffle_buffer: int = 0,
    ):
        """
        Initialize streaming dataset.

        Args:
            data_path: Path to data file (CSV, Parquet, etc.)
            feature_builder: Fitted feature builder
            chunk_size: Number of rows to read at a time
            label_cols: Column names for labels
            shuffle_buffer: Size of shuffle buffer (0 = no shuffle)
        """
        self.data_path = Path(data_path)
        self.feature_builder = feature_builder
        self.chunk_size = chunk_size
        self.label_cols = label_cols or ["click"]
        self.shuffle_buffer = shuffle_buffer

        # Determine file format
        self.file_format = self.data_path.suffix.lower()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over data in chunks."""
        if self.file_format == ".csv":
            reader = pd.read_csv(self.data_path, chunksize=self.chunk_size)
        elif self.file_format == ".parquet":
            # Read parquet in chunks
            df = pd.read_parquet(self.data_path)
            reader = [df[i:i + self.chunk_size] for i in range(0, len(df), self.chunk_size)]
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

        buffer = []

        for chunk in reader:
            # Extract labels
            labels = chunk[self.label_cols].values if self.label_cols else None

            # Drop label columns from features
            feature_cols = [c for c in chunk.columns if c not in self.label_cols]
            data = chunk[feature_cols].to_dict("records")

            # Transform features
            inputs = self.feature_builder.transform(data, labels)

            # Yield samples
            for i in range(len(data)):
                sample = {
                    "sparse_features": inputs.sparse_features[i],
                    "dense_features": inputs.dense_features[i],
                }
                if inputs.labels is not None:
                    sample["labels"] = inputs.labels[i]

                if self.shuffle_buffer > 0:
                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer:
                        np.random.shuffle(buffer)
                        while len(buffer) > self.shuffle_buffer // 2:
                            yield buffer.pop()
                else:
                    yield sample

        # Yield remaining buffer
        if buffer:
            np.random.shuffle(buffer)
            for sample in buffer:
                yield sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Handles batching of samples including variable-length sequences.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors
    """
    result = {
        "sparse_features": torch.stack([s["sparse_features"] for s in batch]),
        "dense_features": torch.stack([s["dense_features"] for s in batch]),
    }

    if "labels" in batch[0]:
        result["labels"] = torch.stack([s["labels"] for s in batch])

    return result


class AdDataModule:
    """
    Data module for managing datasets and dataloaders.

    Handles:
    - Data loading from various sources
    - Feature building and transformation
    - Train/validation split
    - DataLoader creation
    """

    def __init__(
        self,
        feature_builder: FeatureBuilder | None = None,
        batch_size: int = 256,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
    ):
        """
        Initialize data module.

        Args:
            feature_builder: Feature builder (created if None)
            batch_size: Batch size for dataloaders
            num_workers: Number of data loading workers
            shuffle: Whether to shuffle training data
            pin_memory: Whether to pin memory for GPU transfer
        """
        self.feature_builder = feature_builder or FeatureBuilder()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self._train_data: ModelInputs | None = None
        self._val_data: ModelInputs | None = None
        self._test_data: ModelInputs | None = None

    def setup_from_dataframe(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
        label_cols: list[str] | None = None,
        val_split: float = 0.1,
    ) -> None:
        """
        Setup data from pandas DataFrames.

        Args:
            train_df: Training data
            val_df: Validation data (split from train if None)
            test_df: Test data
            label_cols: Column names for labels
            val_split: Validation split ratio if val_df is None
        """
        label_cols = label_cols or ["click"]

        # Split validation if needed
        if val_df is None and val_split > 0:
            n_val = int(len(train_df) * val_split)
            indices = np.random.permutation(len(train_df))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            val_df = train_df.iloc[val_indices].reset_index(drop=True)
            train_df = train_df.iloc[train_indices].reset_index(drop=True)

        # Extract labels
        def extract_labels(df: pd.DataFrame) -> tuple[list[dict], np.ndarray]:
            labels = df[label_cols].values
            feature_cols = [c for c in df.columns if c not in label_cols]
            data = df[feature_cols].to_dict("records")
            return data, labels

        train_data, train_labels = extract_labels(train_df)

        # Fit feature builder on training data
        logger.info("Fitting feature builder...")
        self.feature_builder.fit(train_data)

        # Transform training data
        logger.info("Transforming training data...")
        self._train_data = self.feature_builder.transform(train_data, train_labels)

        # Transform validation data
        if val_df is not None:
            logger.info("Transforming validation data...")
            val_data, val_labels = extract_labels(val_df)
            self._val_data = self.feature_builder.transform(val_data, val_labels)

        # Transform test data
        if test_df is not None:
            logger.info("Transforming test data...")
            test_data, test_labels = extract_labels(test_df)
            self._test_data = self.feature_builder.transform(test_data, test_labels)

    def setup_from_file(
        self,
        train_path: str | Path,
        val_path: str | Path | None = None,
        test_path: str | Path | None = None,
        label_cols: list[str] | None = None,
        val_split: float = 0.1,
    ) -> None:
        """
        Setup data from files.

        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            label_cols: Column names for labels
            val_split: Validation split ratio
        """
        train_path = Path(train_path)

        # Load training data
        if train_path.suffix == ".csv":
            train_df = pd.read_csv(train_path)
        elif train_path.suffix == ".parquet":
            train_df = pd.read_parquet(train_path)
        else:
            raise ValueError(f"Unsupported file format: {train_path.suffix}")

        # Load validation data
        val_df = None
        if val_path:
            val_path = Path(val_path)
            if val_path.suffix == ".csv":
                val_df = pd.read_csv(val_path)
            elif val_path.suffix == ".parquet":
                val_df = pd.read_parquet(val_path)

        # Load test data
        test_df = None
        if test_path:
            test_path = Path(test_path)
            if test_path.suffix == ".csv":
                test_df = pd.read_csv(test_path)
            elif test_path.suffix == ".parquet":
                test_df = pd.read_parquet(test_path)

        self.setup_from_dataframe(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_cols=label_cols,
            val_split=val_split,
        )

    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        if self._train_data is None:
            raise RuntimeError("Data not setup. Call setup_from_* first.")

        dataset = AdDataset(self._train_data)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Get validation DataLoader."""
        if self._val_data is None:
            return None

        dataset = AdDataset(self._val_data)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader | None:
        """Get test DataLoader."""
        if self._test_data is None:
            return None

        dataset = AdDataset(self._test_data)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration from feature builder."""
        return self.feature_builder.get_model_config()
