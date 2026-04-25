# Decision Tree — Documentação da Implementação

Documentação completa do módulo `src/decision_tree/` do projeto de classificação de obesidade.

---

## Navegação

| Arquivo | Conteúdo |
|---------|----------|
| [`01_dados.md`](01_dados.md) | Dataset UCI, carregamento, split estratificado |
| [`02_algoritmo.md`](02_algoritmo.md) | Como a Árvore de Decisão funciona internamente (Gini, divisões, crescimento) |
| [`03_pipeline.md`](03_pipeline.md) | Preprocessing, ColumnTransformer, Pipeline sklearn, classe `DecisionTreeModel` |
| [`04_execucao.md`](04_execucao.md) | Script `run.py`, fluxo de execução, artefatos gerados |

---

## Fluxo geral da aplicação

```mermaid
flowchart TD
    A([python src/decision_tree/run.py]) --> B

    subgraph DADOS ["📦 Camada de Dados — dataset_loader.py"]
        B[ObesityDatasetLoader.load] --> C[fetch_ucirepo id=544]
        C --> D[2111 amostras · 16 features · 7 classes]
        D --> E[train_test_split estratificado 80/20]
        E --> F[X_train 1688 · X_test 423]
    end

    subgraph MODELO ["🌳 Camada de Modelo — model.py"]
        F --> G[DecisionTreeModel.fit]
        G --> H[ColumnTransformer\nOrdinalEncoder + passthrough]
        H --> I[DecisionTreeClassifier\ncriterion=gini · random_state=42]
        I --> J[Árvore treinada\nprofundidade=11 · folhas=105]
    end

    subgraph AVALIACAO ["📊 Avaliação"]
        J --> K[model.evaluate]
        K --> L[accuracy · classification_report\nconfusion_matrix]
        J --> M[model.feature_importances]
        J --> N[get_tree_depth · get_n_leaves]
    end

    subgraph ARTEFATOS ["💾 Artefatos — resultado/"]
        L --> O[relatorio.txt]
        L --> P[matriz_confusao.csv]
        L --> Q[matriz_confusao.png]
        M --> R[importancia_features.csv]
    end
```

---

## Estrutura do módulo

```
src/decision_tree/
├── __init__.py               ← expõe DecisionTreeModel publicamente
├── model.py                  ← classe principal: fit, predict, evaluate, importances
├── run.py                    ← script executável de ponta a ponta
├── README.md                 ← guia rápido de uso
└── resultado/                ← gerado na primeira execução
    ├── relatorio.txt
    ├── matriz_confusao.csv
    ├── matriz_confusao.png
    └── importancia_features.csv
```

---

## Resultado obtido

| Métrica | Valor |
|---------|-------|
| Acurácia | **91.25%** |
| Profundidade da árvore | 11 |
| Número de folhas | 105 |
| Tempo de execução | 2.6s |
| Amostras treino / teste | 1688 / 423 |
