"""Train and evaluate a Random Forest classifier on the UCI obesity dataset."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from obesity_project.data.dataset_loader import ObesityDatasetLoader
from random_forest.model import RandomForestModel

RESULTADO_DIR = Path(__file__).resolve().parent / "resultado"


def _save_relatorio(
    results: dict,
    importances: pd.Series,
    train_size: int,
    test_size: int,
    elapsed: float,
) -> None:
    lines = [
        "=" * 62,
        "  RESULTADOS — Random Forest | UCI Obesity Levels (id=544)",
        "=" * 62,
        f"Gerado em : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tempo de execução : {elapsed:.1f}s",
        f"Amostras treino   : {train_size}",
        f"Amostras teste    : {test_size}",
        "",
        f"Acurácia: {results['accuracy']:.4f}  ({results['accuracy'] * 100:.2f}%)",
        "",
        "--- Relatório de Classificação ---",
        results["classification_report"],
        "--- Importância das Features (completo) ---",
        importances.to_string(),
        "",
        "(Matriz de confusão: ver matriz_confusao.csv e matriz_confusao.png)",
    ]
    path = RESULTADO_DIR / "relatorio.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path.relative_to(Path.cwd())}")


def _save_matriz_confusao_csv(results: dict) -> None:
    labels = results["labels"]
    df = pd.DataFrame(results["confusion_matrix"], index=labels, columns=labels)
    df.index.name = "real \\ predito"
    path = RESULTADO_DIR / "matriz_confusao.csv"
    df.to_csv(path, encoding="utf-8")
    print(f"  -> {path.relative_to(Path.cwd())}")


def _save_matriz_confusao_png(results: dict) -> None:
    labels = results["labels"]
    cm = pd.DataFrame(results["confusion_matrix"], index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Matriz de Confusão — Random Forest", fontsize=13, pad=12)
    ax.set_ylabel("Classe Real", fontsize=11)
    ax.set_xlabel("Classe Predita", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = RESULTADO_DIR / "matriz_confusao.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path.relative_to(Path.cwd())}")


def _save_importancia_features(importances: pd.Series) -> None:
    df = importances.rename("importancia").to_frame()
    df.index.name = "feature"
    path = RESULTADO_DIR / "importancia_features.csv"
    df.to_csv(path, encoding="utf-8")
    print(f"  -> {path.relative_to(Path.cwd())}")


def main() -> None:
    t0 = time.perf_counter()

    print("Carregando dataset do UCI ML Repository...")
    loader = ObesityDatasetLoader()
    loader.load()

    X_train, X_test, y_train, y_test = loader.train_test_split(
        test_size=0.2, stratify=True
    )
    print(
        f"Amostras de treino: {X_train.shape[0]} | "
        f"Amostras de teste:  {X_test.shape[0]}"
    )
    print(f"Features: {X_train.shape[1]} | Classes: {y_train['NObeyesdad'].nunique()}")

    print("\nTreinando Random Forest...")
    model = RandomForestModel(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    print("Avaliando modelo no conjunto de teste...\n")
    results = model.evaluate(X_test, y_test)
    importances = model.feature_importances()

    elapsed = time.perf_counter() - t0

    separator = "=" * 62
    print(separator)
    print("  RESULTADOS — Random Forest | UCI Obesity Levels (id=544)")
    print(separator)

    accuracy = results["accuracy"]
    print(f"\nAcurácia: {accuracy:.4f}  ({accuracy * 100:.2f}%)\n")

    print("Relatório de Classificação:")
    print(results["classification_report"])

    print("Matriz de Confusão:")
    labels = results["labels"]
    print(f"Classes (ordem das linhas/colunas): {labels}")
    print(results["confusion_matrix"])

    print("\nImportância das Features (top 10):")
    print(importances.head(10).to_string())

    print(f"\nTempo de execução: {elapsed:.1f}s")

    print("\nSalvando artefatos em resultado/...")
    RESULTADO_DIR.mkdir(exist_ok=True)
    _save_relatorio(results, importances, X_train.shape[0], X_test.shape[0], elapsed)
    _save_matriz_confusao_csv(results)
    _save_matriz_confusao_png(results)
    _save_importancia_features(importances)
    print("Concluído.")


if __name__ == "__main__":
    main()
