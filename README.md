# Obesity Data Science Project

Projeto de ciencia de dados para estudar e modelar niveis de obesidade com base em habitos alimentares, condicao fisica e caracteristicas demograficas.

O trabalho utiliza o dataset **Estimation of Obesity Levels Based on Eating Habits and Physical Condition**, disponibilizado pelo [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).

## Objetivo

O objetivo principal e tratar o problema como uma tarefa de **classificacao multiclasse**, prevendo uma das sete categorias de obesidade:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

Nesta primeira etapa, o projeto cria uma base reutilizavel para importar o dataset e preparar os dados para experimentos com diferentes algoritmos de machine learning.

## Dataset

A base contem:

- 2111 registros
- 16 atributos de entrada
- 1 variavel-alvo: `NObesity`
- 7 classes de classificacao
- Sem valores ausentes, segundo a documentacao do dataset

Os atributos incluem informacoes como idade, genero, altura, peso, historico familiar, frequencia de consumo de alimentos caloricos, atividade fisica, consumo de agua e meio de transporte.

Mais detalhes estao documentados em [`docs/DataSet/documentation.md`](docs/DataSet/documentation.md).

## Estrutura do Projeto

```text
.
├── docs/
│   ├── DataSet/
│   │   └── documentation.md
│   └── PlanImplementation/
│       └── 02.md
├── examples/
│   └── load_dataset.py
├── src/
│   └── obesity_project/
│       └── data/
│           └── dataset_loader.py
├── requirements.txt
└── README.md
```

## Instalacao

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale as dependencias:

```bash
python -m pip install -r requirements.txt
```

## Como Carregar o Dataset

O projeto possui uma classe reutilizavel para importar os dados da UCI:

```python
from obesity_project.data.dataset_loader import ObesityDatasetLoader

loader = ObesityDatasetLoader()
loader.load()

X = loader.get_features()
y = loader.get_target()
df = loader.get_dataframe()

X_train, X_test, y_train, y_test = loader.train_test_split()
```

Para executar o exemplo:

```bash
PYTHONPATH=src python examples/load_dataset.py
```

## Classe Principal

A classe `ObesityDatasetLoader` esta em [`src/obesity_project/data/dataset_loader.py`](src/obesity_project/data/dataset_loader.py).

Ela oferece:

- `load()`: importa o dataset pelo `ucimlrepo`
- `get_features()`: retorna os atributos de entrada
- `get_target()`: retorna a variavel-alvo
- `get_dataframe()`: retorna atributos e alvo no mesmo dataframe
- `get_metadata()`: retorna metadados da UCI
- `get_variables()`: retorna informacoes das variaveis
- `train_test_split()`: cria divisao treino/teste com estratificacao por padrao

## Direcao de Modelagem

Algoritmo inicial recomendado:

**Random Forest**

Motivos:

- bom desempenho em dados tabulares;
- suporte natural a classificacao multiclasse;
- lida bem com relacoes nao lineares;
- permite analisar importancia das variaveis;
- serve como baseline forte para comparacao.

Outros algoritmos interessantes para comparar:

- Decision Tree
- KNN
- Logistic Regression multinomial
- SVM
- Gradient Boosting, caso sejam adicionadas novas dependencias

## Analises Interessantes

Uma pergunta importante para o projeto:

> O modelo consegue prever obesidade a partir de habitos e estilo de vida, ou depende quase totalmente de `Weight` e `Height`?

Experimentos sugeridos:

- treinar modelos com todos os atributos;
- treinar modelos removendo `Weight` e `Height`;
- comparar a queda de desempenho;
- avaliar quais atributos mais influenciam a predicao;
- analisar erros entre classes proximas, como `Overweight Level I` e `Overweight Level II`.

Essa comparacao ajuda a separar uma predicao quase direta de IMC de uma analise mais rica sobre fatores comportamentais e de estilo de vida.

## Proximos Passos

- Criar pipelines de preprocessing para variaveis categoricas e numericas.
- Implementar um baseline com Random Forest.
- Comparar metricas por classe usando matriz de confusao.
- Avaliar importancia das variaveis.
- Documentar os resultados para a apresentacao final.

## Referencia

Palechor, F. M., & Manotas, A. D. (2019). *Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico*. Data in Brief, 25, 104344.

Dataset UCI: https://doi.org/10.24432/C5H31Z
