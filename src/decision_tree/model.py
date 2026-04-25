"""DecisionTreeClassifier with preprocessing pipeline for the obesity dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModel:
    """Multiclass obesity level classifier using Decision Tree.

    Wraps a scikit-learn Pipeline that applies OrdinalEncoder to categorical
    features and passes numeric features through unchanged, then trains a
    DecisionTreeClassifier. Designed for the UCI Obesity Levels dataset (id=544):
    16 input features, 7-class target NObesity.
    """

    TARGET_COLUMN = "NObeyesdad"

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self._pipeline: Pipeline | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> "DecisionTreeModel":
        """Build preprocessing pipeline and train the classifier."""
        cat_cols = X_train.select_dtypes(include="object").columns.tolist()
        num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    cat_cols,
                ),
                ("num", "passthrough", num_cols),
            ],
            remainder="drop",
        )

        self._pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        criterion=self.criterion,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        self._pipeline.fit(X_train, y_train[self.TARGET_COLUMN].values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels for the input features."""
        self._ensure_fitted()
        return self._pipeline.predict(X)

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> dict[str, object]:
        """Return accuracy, classification report, and confusion matrix."""
        self._ensure_fitted()
        y_true = y_test[self.TARGET_COLUMN].values
        y_pred = self.predict(X_test)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "labels": sorted(set(y_true)),
        }

    def feature_importances(self) -> pd.Series:
        """Return feature importances sorted descending.

        Feature names strip the transformer prefix added by ColumnTransformer
        (e.g. 'cat__Gender' -> 'Gender', 'num__Age' -> 'Age').
        """
        self._ensure_fitted()
        preprocessor = self._pipeline.named_steps["preprocessor"]
        clf = self._pipeline.named_steps["classifier"]
        raw_names = preprocessor.get_feature_names_out()
        names = [n.split("__", 1)[-1] for n in raw_names]
        return pd.Series(clf.feature_importances_, index=names).sort_values(
            ascending=False
        )

    def get_tree_depth(self) -> int:
        """Return the actual depth of the trained tree."""
        self._ensure_fitted()
        return self._pipeline.named_steps["classifier"].get_depth()

    def get_n_leaves(self) -> int:
        """Return the number of leaves in the trained tree."""
        self._ensure_fitted()
        return self._pipeline.named_steps["classifier"].get_n_leaves()

    def _ensure_fitted(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(
                "Model not trained. Call fit() before predict() or evaluate()."
            )
