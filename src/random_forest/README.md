# Random Forest — Classificação de Obesidade

Módulo de classificação multiclasse usando **RandomForestClassifier** (scikit-learn) sobre o dataset UCI Obesity Levels (id=544, 2.111 amostras, 16 features, 7 classes alvo em `NObesity`).

## Dependências

Instale a partir da raiz do projeto:

```bash
pip install -r requirements.txt
```

## Execução

Execute sempre a partir da **raiz do projeto**:

```bash
python src/random_forest/run.py
```

A primeira execução baixa o dataset da UCI (requer conexão com a internet). As execuções seguintes utilizam o cache interno do `ucimlrepo`.

## Saída esperada

```
Carregando dataset do UCI ML Repository...
Amostras de treino: 1688 | Amostras de teste: 423
Features: 16 | Classes: 7

Treinando Random Forest...
Avaliando modelo no conjunto de teste...

==============================================================
  RESULTADOS — Random Forest | UCI Obesity Levels (id=544)
==============================================================

Acurácia: 0.XXXX  (XX.XX%)

Relatório de Classificação:
              precision    recall  f1-score   support
...

Matriz de Confusão:
...

Importância das Features (top 10):
Weight      0.XXXX
Height      0.XXXX
...
```

## Estrutura do módulo

```
src/random_forest/
├── __init__.py   # interface pública: expõe RandomForestModel
├── model.py      # classe RandomForestModel (fit, predict, evaluate, feature_importances)
├── run.py        # script executável
└── README.md     # este arquivo
```

## Uso programático

```python
from src.obesity_project.data.dataset_loader import ObesityDatasetLoader
from src.random_forest.model import RandomForestModel

loader = ObesityDatasetLoader()
loader.load()

X_train, X_test, y_train, y_test = loader.train_test_split()

model = RandomForestModel(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

results = model.evaluate(X_test, y_test)
print(results["accuracy"])
print(results["classification_report"])
print(results["confusion_matrix"])

importances = model.feature_importances()
```
