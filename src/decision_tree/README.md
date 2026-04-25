# Decision Tree — Classificação de Obesidade

Módulo de classificação multiclasse usando **DecisionTreeClassifier** (scikit-learn) sobre o dataset UCI Obesity Levels (id=544, 2.111 amostras, 16 features, 7 classes alvo em `NObeyesdad`).

Este módulo faz parte de uma série de experimentos comparativos de algoritmos de Machine Learning. Os resultados aqui gerados são comparados diretamente com o módulo `random_forest/`.

---

## O que é uma Árvore de Decisão?

Uma **Árvore de Decisão** é um modelo de aprendizado supervisionado que aprende a classificar dados fazendo perguntas do tipo "sim/não" sobre os atributos. Ela constrói uma estrutura hierárquica onde:

- **Nó raiz:** a primeira pergunta (o atributo mais discriminativo)
- **Nós intermediários:** perguntas subsequentes sobre os dados
- **Folhas:** a decisão final (a classe prevista)

### Exemplo visual simplificado

```
                    Weight > 80?
                   /            \
                SIM              NÃO
                 |                 |
           Height > 1.75?      Age > 30?
           /        \           /      \
         SIM         NÃO      SIM      NÃO
          |           |        |         |
    Obesity_I   Normal_W   Overweight  Normal_W
```

O algoritmo escolhe, em cada nó, o atributo e o limiar que melhor separa as classes. O critério padrão usado aqui é o **Gini**, que mede o grau de "impureza" de um nó — quanto menor o Gini, mais pura (homogênea) é a divisão.

### Por que usar Árvore de Decisão neste projeto?

| Vantagem | Relevância para o projeto |
|----------|--------------------------|
| Interpretabilidade total | Possível visualizar as regras que classificam a obesidade |
| Sem necessidade de normalização | Dados mistos (numérico + categórico) são tratados nativamente |
| Rápida para treinar | Ideal para experimentação iterativa |
| Servir de baseline | Ponto de comparação honesto com o Random Forest |

---

## Estrutura do módulo

```
src/decision_tree/
├── __init__.py        # interface pública: expõe DecisionTreeModel
├── model.py           # classe DecisionTreeModel
├── run.py             # script executável de treinamento e avaliação
├── README.md          # este arquivo
└── resultado/         # criado automaticamente na primeira execução
    ├── relatorio.txt
    ├── matriz_confusao.csv
    ├── matriz_confusao.png
    └── importancia_features.csv
```

---

## Dependências

Instale a partir da raiz do projeto:

```bash
pip install -r requirements.txt
```

---

## Execução

Execute sempre a partir da **raiz do projeto**:

```bash
python src/decision_tree/run.py
```

A primeira execução baixa o dataset da UCI (requer internet). As seguintes usam o cache do `ucimlrepo`.

---

## Saída esperada

```
Carregando dataset do UCI ML Repository...
Amostras de treino: 1688 | Amostras de teste:  423
Features: 16 | Classes: 7

Treinando Decision Tree...
Avaliando modelo no conjunto de teste...

==============================================================
  RESULTADOS — Decision Tree | UCI Obesity Levels (id=544)
==============================================================

Acurácia: 0.9125  (91.25%)
Profundidade da árvore: 11
Número de folhas      : 105

Relatório de Classificação:
                     precision    recall  f1-score   support

Insufficient_Weight       0.98      0.87      0.92        54
      Normal_Weight       0.77      0.83      0.80        58
     Obesity_Type_I       0.94      0.94      0.94        70
    Obesity_Type_II       0.97      0.97      0.97        60
   Obesity_Type_III       1.00      0.98      0.99        65
 Overweight_Level_I       0.82      0.84      0.83        58
Overweight_Level_II       0.92      0.93      0.92        58

           accuracy                           0.91       423
          macro avg       0.91      0.91      0.91       423
       weighted avg       0.92      0.91      0.91       423

Importância das Features (top 10):
Weight    0.474104
Height    0.216299
Gender    0.157929
Age       0.042825
...

Tempo de execução: 2.6s
```

---

## Parâmetros configuráveis

A classe `DecisionTreeModel` aceita os seguintes parâmetros no construtor:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `max_depth` | `None` | Profundidade máxima da árvore. `None` = cresce até folhas puras. |
| `min_samples_split` | `2` | Mínimo de amostras para dividir um nó interno. |
| `min_samples_leaf` | `1` | Mínimo de amostras que uma folha deve ter. |
| `criterion` | `"gini"` | Critério de divisão: `"gini"` ou `"entropy"`. |
| `random_state` | `42` | Semente aleatória para reproducibilidade. |

### Exemplo: árvore controlada por profundidade

```python
model = DecisionTreeModel(max_depth=5, criterion="entropy", random_state=42)
model.fit(X_train, y_train)
```

Limitar `max_depth` reduz overfitting — a árvore aprende menos regras específicas do treino e generaliza melhor para dados novos.

---

## Uso programático

```python
from obesity_project.data.dataset_loader import ObesityDatasetLoader
from decision_tree.model import DecisionTreeModel

# 1. Carregar e dividir os dados
loader = ObesityDatasetLoader()
loader.load()
X_train, X_test, y_train, y_test = loader.train_test_split()

# 2. Treinar o modelo
model = DecisionTreeModel(random_state=42)
model.fit(X_train, y_train)

# 3. Avaliar
results = model.evaluate(X_test, y_test)
print(f"Acurácia: {results['accuracy']:.4f}")
print(results["classification_report"])

# 4. Importância das features
importances = model.feature_importances()
print(importances.head(5))

# 5. Informações sobre a estrutura da árvore
print(f"Profundidade: {model.get_tree_depth()}")
print(f"Folhas:       {model.get_n_leaves()}")
```

---

## Resultados obtidos

> Executado em 25/04/2026 | Python 3.x | scikit-learn | random_state=42 | split 80/20 estratificado

### Métricas gerais

| Métrica | Valor |
|---------|-------|
| **Acurácia** | **91.25%** |
| Tempo de execução | 2.6s |
| Amostras de treino | 1.688 |
| Amostras de teste | 423 |
| Profundidade da árvore | 11 |
| Número de folhas | 105 |

### Desempenho por classe

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Insufficient_Weight | 0.98 | 0.87 | 0.92 | 54 |
| Normal_Weight | 0.77 | 0.83 | 0.80 | 58 |
| Obesity_Type_I | 0.94 | 0.94 | 0.94 | 70 |
| Obesity_Type_II | 0.97 | 0.97 | 0.97 | 60 |
| Obesity_Type_III | 1.00 | 0.98 | 0.99 | 65 |
| Overweight_Level_I | 0.82 | 0.84 | 0.83 | 58 |
| Overweight_Level_II | 0.92 | 0.93 | 0.92 | 58 |

### Importância das features (top 10)

| Posição | Feature | Importância |
|---------|---------|-------------|
| 1 | Weight | 47.41% |
| 2 | Height | 21.63% |
| 3 | Gender | 15.79% |
| 4 | Age | 4.28% |
| 5 | CALC | 2.29% |
| 6 | FAVC | 2.18% |
| 7 | FCVC | 1.49% |
| 8 | TUE | 1.34% |
| 9 | CAEC | 1.25% |
| 10 | NCP | 0.92% |

**Interpretação:** A árvore concentra quase toda a sua capacidade preditiva em `Weight` (47.41%) e `Height` (21.63%). Isso indica que, para uma árvore simples, esses dois atributos já respondem por ~69% da classificação — os hábitos de vida têm papel secundário nessa estrutura.

---

## Comparação com Random Forest

Esta é a razão de existir deste módulo: comparar diretamente com o `random_forest/`.

| Métrica | Decision Tree | Random Forest | Diferença |
|---------|:-------------:|:-------------:|:---------:|
| **Acurácia** | **91.25%** | **95.98%** | -4.73 pp |
| Tempo de execução | 2.6s | 3.9s | -1.3s |
| Profundidade | 11 | N/A (floresta) | — |
| Número de estimadores | 1 árvore | 200 árvores | — |
| Weight (importância) | 47.41% | 34.14% | +13.27 pp |
| Height (importância) | 21.63% | 10.24% | +11.39 pp |
| Melhor classe | Obesity_Type_III (F1=0.99) | Obesity_Type_III (F1=0.99) | 0 |
| Pior classe | Normal_Weight (F1=0.80) | Normal_Weight (F1=0.91) | -0.11 |

### O que explica a diferença de performance?

**Random Forest é um ensemble:** em vez de construir uma única árvore, ele constrói 200 árvores, cada uma treinada em uma amostra aleatória dos dados e com um subconjunto aleatório das features. A previsão final é feita por votação majoritária. Isso reduz dois problemas clássicos de Árvores de Decisão isoladas:

1. **Overfitting:** Uma única árvore cresce memorizando o conjunto de treino. A floresta, ao fazer votação, suaviza erros individuais.
2. **Alta variância:** Uma pequena mudança nos dados pode gerar uma árvore completamente diferente. Na floresta, essa variabilidade é absorvida pelas outras 199 árvores.

**Diferença na importância das features:** A Decision Tree concentrou 69% do peso em `Weight` + `Height`. Já o Random Forest distribui mais o peso — `Weight` (34%) + `Height` (10%) + `Age` (10%) + `FCVC` (9%) + outros. Isso acontece porque cada árvore da floresta usa subconjuntos de features diferentes, forçando o modelo a aprender relações mais diversas e capturando o papel dos hábitos de vida com mais fidelidade.

### Quando usar cada um?

| Cenário | Recomendação |
|---------|-------------|
| Máxima performance preditiva | Random Forest |
| Explicar a lógica de classificação | Decision Tree |
| Produção com restrições de memória/tempo | Decision Tree |
| Análise de quais hábitos realmente importam | Random Forest (distribuição mais equilibrada) |
| Protótipo rápido | Decision Tree |

---

## Artefatos gerados em `resultado/`

| Arquivo | Conteúdo |
|---------|----------|
| `relatorio.txt` | Relatório completo em texto com todas as métricas |
| `matriz_confusao.csv` | Matriz de confusão em formato tabular |
| `matriz_confusao.png` | Heatmap visual da matriz de confusão |
| `importancia_features.csv` | Importância de todas as 16 features ordenadas |

---

## Como o pipeline funciona internamente

O modelo usa um `Pipeline` do scikit-learn com duas etapas:

```
X_train (raw)
    │
    ▼
┌─────────────────────────────┐
│  ColumnTransformer           │
│  ┌────────────────────────┐ │
│  │  OrdinalEncoder        │ │  ← features categóricas (Gender, CAEC, etc.)
│  │  (handle_unknown=-1)   │ │
│  └────────────────────────┘ │
│  ┌────────────────────────┐ │
│  │  passthrough           │ │  ← features numéricas (Age, Height, Weight, etc.)
│  └────────────────────────┘ │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  DecisionTreeClassifier      │
│  criterion="gini"            │
│  random_state=42             │
└─────────────────────────────┘
    │
    ▼
  y_pred (7 classes)
```

O `OrdinalEncoder` transforma strings categóricas em inteiros (`Female`→0, `Male`→1) sem criar colunas extras (diferente do One-Hot Encoding). Isso é adequado para Árvores de Decisão, que não assumem ordenação entre os valores.
