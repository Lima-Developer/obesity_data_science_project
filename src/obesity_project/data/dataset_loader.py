"""Reusable loader for the UCI obesity levels dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from ucimlrepo import fetch_ucirepo


@dataclass
class ObesityDatasetLoader:
    """Load and expose the UCI obesity levels dataset.

    This class intentionally keeps preprocessing out of the loading layer so
    experiments can compare different modeling pipelines fairly.
    """

    dataset_id: int = 544
    target_column: str = "NObeyesdad"
    random_state: int = 42
    _dataset: Any | None = field(default=None, init=False, repr=False)
    _features: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _target: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def load(self) -> "ObesityDatasetLoader":
        """Fetch the dataset from the UCI repository and cache its parts."""
        self._dataset = fetch_ucirepo(id=self.dataset_id)
        self._features = self._dataset.data.features
        self._target = self._dataset.data.targets
        return self

    def get_features(self) -> pd.DataFrame:
        """Return the feature matrix."""
        self._ensure_loaded()
        return self._features.copy()

    def get_target(self) -> pd.DataFrame:
        """Return the target column as provided by UCI."""
        self._ensure_loaded()
        return self._target.copy()

    def get_dataframe(self) -> pd.DataFrame:
        """Return a single dataframe with features and target side by side."""
        self._ensure_loaded()
        return pd.concat([self._features, self._target], axis=1)

    def get_metadata(self) -> Any:
        """Return dataset metadata from UCI."""
        self._ensure_loaded()
        return self._dataset.metadata

    def get_variables(self) -> pd.DataFrame:
        """Return variable information from UCI."""
        self._ensure_loaded()
        return self._dataset.variables.copy()

    def train_test_split(
        self,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split features and target into train and test sets."""
        self._ensure_loaded()
        stratify_values = self._target[self.target_column] if stratify else None

        return sklearn_train_test_split(
            self._features,
            self._target,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_values,
        )

    def _ensure_loaded(self) -> None:
        if self._dataset is None or self._features is None or self._target is None:
            raise RuntimeError("Dataset not loaded. Call load() before accessing data.")

