# 01 — Dataset e Carregamento de Dados

## O dataset

O projeto usa o **UCI Obesity Levels Dataset** (id=544), disponível publicamente no UCI Machine Learning Repository.

| Propriedade | Valor |
|-------------|-------|
| Total de amostras | 2.111 |
| Features de entrada | 16 |
| Variável alvo | `NObeyesdad` (7 classes) |
| Valores ausentes | 0 |
| Origem | México, Peru, Colômbia |
| Composição | 77% sintético (SMOTE) + 23% coletado via web |

### As 7 classes alvo

```mermaid
graph LR
    A[Insufficient Weight] --> B[Normal Weight]
    B --> C[Overweight Level I]
    C --> D[Overweight Level II]
    D --> E[Obesity Type I]
    E --> F[Obesity Type II]
    F --> G[Obesity Type III]

    style A fill:#a8d8ea
    style B fill:#a8e6cf
    style C fill:#ffd3b6
    style D fill:#ffaaa5
    style E fill:#ff8b94
    style F fill:#e05c5c
    style G fill:#b22222,color:#fff
```

As classes seguem uma escala contínua de IMC — classes vizinhas são naturalmente mais difíceis de separar, o que explica as principais confusões que o modelo comete.

### As 16 features de entrada

| Tipo | Features |
|------|----------|
| Demográfico | `Gender`, `Age`, `Height`, `Weight` |
| Hábitos alimentares | `FAVC`, `FCVC`, `NCP`, `CAEC`, `CH2O`, `CALC`, `SCC` |
| Histórico | `family_history_with_overweight` |
| Estilo de vida | `SMOKE`, `FAF`, `TUE`, `MTRANS` |

---

## A classe `ObesityDatasetLoader`

Definida em `src/obesity_project/data/dataset_loader.py`, é uma **dataclass** que centraliza todo o acesso ao dataset. O módulo Decision Tree não acessa o dataset diretamente — ele sempre passa por esse loader.

### Por que uma classe separada?

Separar o carregamento do modelo permite que múltiplos experimentos (Decision Tree, Random Forest, futuros) compartilhem o mesmo dataset carregado com a mesma lógica, sem duplicar código.

### Fluxo de carregamento

```mermaid
sequenceDiagram
    participant run as run.py
    participant loader as ObesityDatasetLoader
    participant uci as ucimlrepo (UCI)
    participant cache as Cache interno

    run->>loader: ObesityDatasetLoader()
    Note over loader: dataset_id=544<br/>target_column="NObeyesdad"<br/>random_state=42

    run->>loader: loader.load()
    loader->>uci: fetch_ucirepo(id=544)

    alt Primeira execução
        uci-->>loader: dataset completo (download)
    else Execuções seguintes
        cache-->>loader: dataset em cache local
    end

    loader->>loader: _features = dataset.data.features
    loader->>loader: _target   = dataset.data.targets
    loader-->>run: self (encadeável)
```

### Código da classe

```python
@dataclass
class ObesityDatasetLoader:
    dataset_id: int = 544
    target_column: str = "NObeyesdad"
    random_state: int = 42
    _dataset: Any | None = field(default=None, init=False, repr=False)
    _features: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _target: pd.DataFrame | None = field(default=None, init=False, repr=False)
```

Os atributos prefixados com `_` não aparecem no `repr` e não são passados no construtor (`init=False`). Isso garante que o estado interno (dataset carregado) seja gerenciado apenas pelos métodos da classe.

### Guard `_ensure_loaded`

Todos os métodos de acesso chamam internamente `_ensure_loaded()` antes de retornar dados:

```python
def _ensure_loaded(self) -> None:
    if self._dataset is None or self._features is None or self._target is None:
        raise RuntimeError("Dataset not loaded. Call load() before accessing data.")
```

Isso impede bugs silenciosos: se alguém chamar `get_features()` sem ter chamado `load()` antes, recebe um erro claro.

---

## Split treino/teste estratificado

O loader expõe o método `train_test_split`, que usa internamente o `train_test_split` do sklearn com `stratify`:

```python
def train_test_split(self, test_size=0.2, stratify=True):
    stratify_values = self._target[self.target_column] if stratify else None
    return sklearn_train_test_split(
        self._features,
        self._target,
        test_size=test_size,
        random_state=self.random_state,
        stratify=stratify_values,
    )
```

### Por que estratificar?

```mermaid
flowchart LR
    subgraph SEM ["Sem estratificação (risco)"]
        A1[Dataset original\n7 classes desequilibradas] --> B1[Divisão aleatória]
        B1 --> C1[Treino: pode ter\npoucas amostras de\nObesity_Type_II]
        B1 --> D1[Teste: pode ter\nmuitas amostras de\nObesity_Type_II]
    end

    subgraph COM ["Com estratificação (correto)"]
        A2[Dataset original\n7 classes] --> B2[Divisão proporcional]
        B2 --> C2[Treino: 80% de\ncada classe]
        B2 --> D2[Teste: 20% de\ncada classe]
    end
```

**Com `stratify=True`**, o sklearn garante que cada classe aparece no treino e no teste na mesma proporção do dataset original. Isso é fundamental para que as métricas de avaliação sejam confiáveis — especialmente em datasets onde as classes não são perfeitamente balanceadas.

### Resultado do split

| Conjunto | Amostras | Proporção |
|----------|----------|-----------|
| Treino | 1.688 | 80% |
| Teste | 423 | 20% |
| Total | 2.111 | 100% |

O `random_state=42` garante que qualquer execução do código produza exatamente o mesmo split — essencial para reprodutibilidade científica.
