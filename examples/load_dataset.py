"""Example usage of the UCI obesity dataset loader."""

from src.obesity_project.data.dataset_loader import ObesityDatasetLoader


def main() -> None:
    loader = ObesityDatasetLoader()
    loader.load()

    X = loader.get_features()
    y = loader.get_target()
    df = loader.get_dataframe()

    X_train, X_test, y_train, y_test = loader.train_test_split()

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Full dataframe shape:", df.shape)
    print("Train/test shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if __name__ == "__main__":
    main()
