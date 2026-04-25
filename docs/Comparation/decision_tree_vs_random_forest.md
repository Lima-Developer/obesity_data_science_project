# Comparação: Decision Tree vs Random Forest
### UCI Obesity Levels Dataset (id=544) — 2.111 amostras | 16 features | 7 classes

> **Condições idênticas:** mesmo dataset, mesmo split (80/20 estratificado), mesmo `random_state=42`, mesmo preprocessing (OrdinalEncoder + passthrough). As diferenças observadas são exclusivamente do algoritmo.

---

## 1. Visão Geral dos Algoritmos

### Decision Tree

Uma Decision Tree constrói uma única estrutura hierárquica de perguntas binárias. A cada nó, o algoritmo escolhe o atributo e o limiar que **minimiza a impureza** (critério Gini) nas duas partições resultantes. O processo repete-se recursivamente até que os nós contenham amostras de uma única classe (folhas puras) ou que restrições de parada sejam atingidas.

```
Nó raiz → Pergunta sobre o atributo mais discriminativo
    ├── SIM → Sub-árvore esquerda
    └── NÃO → Sub-árvore direita
              ├── SIM → ...
              └── NÃO → Folha (classe prevista)
```

**Resultado:** 1 árvore única, 11 níveis de profundidade, 105 folhas.

---

### Random Forest

Um Random Forest constrói **200 Decision Trees independentes**, cada uma com duas fontes de aleatoriedade introduzidas propositalmente:

1. **Bootstrap sampling:** cada árvore treina em uma amostra aleatória *com reposição* do conjunto de treino (~63% das amostras originais por árvore)
2. **Feature subsampling:** em cada nó, apenas um subconjunto aleatório de features é candidato à divisão (padrão: √16 ≈ 4 features por nó)

A previsão final é feita por **votação majoritária** das 200 árvores.

```
Amostra de entrada
        │
        ├── Árvore 1 → Obesity_Type_I   ┐
        ├── Árvore 2 → Obesity_Type_I   │
        ├── Árvore 3 → Normal_Weight    │ votação
        ├── ...                         │ majoritária
        └── Árvore 200 → Obesity_Type_I ┘
                │
        Previsão final: Obesity_Type_I
```

---

## 2. Comparação de Performance Geral

| Métrica | Decision Tree | Random Forest | Diferença |
|---------|:-------------:|:-------------:|:---------:|
| **Acurácia** | **91.25%** | **95.98%** | RF +4.73 pp |
| Macro Precision | 0.91 | 0.96 | RF +0.05 |
| Macro Recall | 0.91 | 0.96 | RF +0.05 |
| Macro F1-Score | 0.91 | 0.96 | RF +0.05 |
| Weighted Precision | 0.92 | 0.96 | RF +0.04 |
| Weighted Recall | 0.91 | 0.96 | RF +0.05 |
| Weighted F1-Score | 0.91 | 0.96 | RF +0.05 |
| Tempo de execução | **2.6s** | 3.9s | DT -1.3s |
| Amostras treino | 1.688 | 1.688 | — |
| Amostras teste | 423 | 423 | — |

**O Random Forest supera a Decision Tree em todas as métricas de classificação**, pagando um custo de tempo marginalmente maior (+1.3s) que é irrelevante na prática.

---

## 3. Comparação por Classe

### Precision (por classe)

| Classe | Decision Tree | Random Forest | Vencedor |
|--------|:---:|:---:|:---:|
| Insufficient_Weight | 0.98 | 1.00 | RF |
| Normal_Weight | 0.77 | 0.84 | RF |
| Obesity_Type_I | 0.94 | 0.97 | RF |
| Obesity_Type_II | 0.97 | 0.98 | RF |
| Obesity_Type_III | 1.00 | 1.00 | Empate |
| Overweight_Level_I | 0.82 | 0.98 | RF |
| Overweight_Level_II | 0.92 | 0.96 | RF |

### Recall (por classe)

| Classe | Decision Tree | Random Forest | Vencedor |
|--------|:---:|:---:|:---:|
| Insufficient_Weight | 0.87 | 0.93 | RF |
| Normal_Weight | 0.83 | **1.00** | RF |
| Obesity_Type_I | 0.94 | 0.97 | RF |
| Obesity_Type_II | 0.97 | 0.98 | RF |
| Obesity_Type_III | 0.98 | 0.98 | Empate |
| Overweight_Level_I | 0.84 | 0.90 | RF |
| Overweight_Level_II | 0.93 | 0.95 | RF |

### F1-Score (por classe)

| Classe | Decision Tree | Random Forest | Diferença |
|--------|:---:|:---:|:---:|
| Insufficient_Weight | 0.92 | 0.96 | RF +0.04 |
| **Normal_Weight** | **0.80** | **0.91** | **RF +0.11** |
| Obesity_Type_I | 0.94 | 0.97 | RF +0.03 |
| Obesity_Type_II | 0.97 | 0.98 | RF +0.01 |
| Obesity_Type_III | 0.99 | 0.99 | Empate |
| **Overweight_Level_I** | **0.83** | **0.94** | **RF +0.11** |
| Overweight_Level_II | 0.92 | 0.96 | RF +0.04 |

**Classes mais prejudicadas na Decision Tree:** `Normal_Weight` e `Overweight_Level_I` — exatamente as duas classes fronteiriças mais próximas entre si na escala de IMC. A árvore única não captura bem a fronteira entre elas.

---

## 4. Análise das Matrizes de Confusão

### Decision Tree — Matriz de Confusão

|  | Insuf. | Normal | Ob. I | Ob. II | Ob. III | Ow. I | Ow. II |
|--|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Insuf.** | **47** | 7 | 0 | 0 | 0 | 0 | 0 |
| **Normal** | 1 | **48** | 0 | 0 | 0 | 9 | 0 |
| **Ob. I** | 0 | 1 | **66** | 1 | 0 | 0 | 2 |
| **Ob. II** | 0 | 0 | 2 | **58** | 0 | 0 | 0 |
| **Ob. III** | 0 | 0 | 0 | 1 | **64** | 0 | 0 |
| **Ow. I** | 0 | 6 | 0 | 0 | 0 | **49** | 3 |
| **Ow. II** | 0 | 0 | 2 | 0 | 0 | 2 | **54** |

**Total de erros: 37 / 423 amostras**

Principais confusões:
- `Normal_Weight` → `Overweight_Level_I`: **9 casos** (maior fonte de erro)
- `Insufficient_Weight` → `Normal_Weight`: **7 casos**
- `Overweight_Level_I` → `Normal_Weight`: **6 casos**

---

### Random Forest — Matriz de Confusão

|  | Insuf. | Normal | Ob. I | Ob. II | Ob. III | Ow. I | Ow. II |
|--|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Insuf.** | **50** | 4 | 0 | 0 | 0 | 0 | 0 |
| **Normal** | 0 | **58** | 0 | 0 | 0 | 0 | 0 |
| **Ob. I** | 0 | 0 | **68** | 0 | 0 | 0 | 2 |
| **Ob. II** | 0 | 0 | 1 | **59** | 0 | 0 | 0 |
| **Ob. III** | 0 | 0 | 0 | 1 | **64** | 0 | 0 |
| **Ow. I** | 0 | 6 | 0 | 0 | 0 | **52** | 0 |
| **Ow. II** | 0 | 1 | 1 | 0 | 0 | 1 | **55** |

**Total de erros: 17 / 423 amostras**

Principais confusões:
- `Overweight_Level_I` → `Normal_Weight`: **6 casos** (mesma fronteira difícil)
- `Insufficient_Weight` → `Normal_Weight`: **4 casos**
- Demais confusões: 1-2 casos cada

---

### Comparação direta de erros

| Fonte de erro | Decision Tree | Random Forest | Redução |
|---------------|:---:|:---:|:---:|
| Normal → Overweight_I | 9 | 0 | **-9 casos** |
| Insuf. → Normal | 7 | 4 | -3 casos |
| Overweight_I → Normal | 6 | 6 | 0 |
| **Total de erros** | **37** | **17** | **-54%** |

O Random Forest eliminou completamente a confusão Normal→Overweight_I (9 casos) e reduziu os demais erros pela metade. A única fonte de erro que persiste com igual magnitude em ambos é `Overweight_Level_I` → `Normal_Weight` (6 casos), evidenciando que essa fronteira é genuinamente difícil e relacionada à similaridade de IMC entre as duas classes.

---

## 5. Importância das Features — Comparação Completa

| Feature | Decision Tree | Random Forest | Δ (DT − RF) | Interpretação |
|---------|:---:|:---:|:---:|---|
| Weight | **47.41%** | 34.14% | +13.27 pp | DT concentra mais em peso |
| Height | **21.63%** | 10.24% | +11.39 pp | DT concentra mais em altura |
| Gender | **15.79%** | 6.08% | +9.71 pp | DT usa gênero como atalho |
| Age | 4.28% | **9.65%** | -5.37 pp | RF capta mais o papel da idade |
| FCVC | 1.49% | **8.57%** | -7.08 pp | RF valoriza vegetais |
| NCP | 0.92% | **5.17%** | -4.25 pp | RF valoriza refeições |
| FAF | 0.23% | **4.33%** | -4.10 pp | RF valoriza atividade física |
| TUE | 1.34% | **4.24%** | -2.90 pp | RF valoriza tempo de tela |
| CH2O | 0.35% | **4.11%** | -3.76 pp | RF valoriza hidratação |
| CAEC | 1.25% | **3.37%** | -2.12 pp | RF valoriza alimentação entre refeições |
| CALC | 2.29% | 2.95% | -0.66 pp | Similar |
| family_history | 0.28% | 2.91% | -2.63 pp | RF capta mais hereditariedade |
| MTRANS | 0.39% | 1.84% | -1.45 pp | RF capta mais transporte |
| FAVC | 2.18% | 1.62% | +0.56 pp | Similar |
| SMOKE | 0.19% | 0.23% | -0.04 pp | Ambos descartam tabaco |
| SCC | **0.00%** | 0.54% | -0.54 pp | DT ignora completamente |

### Análise crítica da distribuição de importância

**Decision Tree (concentrada):**
- Top 3 features: `Weight` + `Height` + `Gender` = **84.83%** do peso total
- Restantes 13 features dividem apenas 15.17%
- A árvore encontrou um "atalho" — as 3 features mais superficiais já geram alta precisão, e ela para de explorar o restante

**Random Forest (distribuída):**
- Top 3 features: `Weight` + `Height` + `Age` = **54.03%** do peso total
- Restantes 13 features dividem 45.97%
- Com subsampling de features por nó, cada árvore é forçada a aprender com diferentes combinações, revelando a influência real de hábitos alimentares e físicos

**Conclusão sobre hábitos de vida:** O Random Forest revela que `FCVC` (frequência de vegetais, 8.57%), `FAF` (atividade física, 4.33%) e `CH2O` (água, 4.11%) têm contribuição real para a classificação. A Decision Tree subestima drasticamente esses atributos por já conseguir boa performance com as 3 features dominantes.

---

## 6. Complexidade Estrutural

| Característica | Decision Tree | Random Forest |
|----------------|:---:|:---:|
| Número de modelos | 1 | 200 |
| Profundidade (por árvore) | 11 | ~11–20 (variável) |
| Número de folhas | 105 | ~105 × 200 ≈ 21.000 |
| Parâmetros livres | `max_depth`, `criterion`, `min_samples_split`, `min_samples_leaf` | idem + `n_estimators`, `max_features` |
| Interpretabilidade | Alta (visualizável) | Baixa (ensemble) |
| Reprodutibilidade | Determinística | Determinística (com `random_state`) |

---

## 7. Custo Computacional

| Fase | Decision Tree | Random Forest | Fator |
|------|:---:|:---:|:---:|
| Treinamento | 2.6s | 3.9s | RF 1.5× mais lento |
| Inferência (por amostra) | ~µs | ~200× µs | RF mais lento |
| Memória | Baixa (1 árvore) | Alta (200 árvores) | RF ~200× mais memória |
| Paralelização | Não aplicável | Sim (`n_jobs=-1`) | RF escala com CPUs |

> Na escala deste dataset (2.111 amostras), a diferença de 1.3s é irrelevante. Em datasets com milhões de amostras, o custo do Random Forest pode se tornar um fator decisivo.

---

## 8. Overfitting e Generalização

A Decision Tree, sem restrições de profundidade (`max_depth=None`), cresce até criar folhas quase puras — o que significa **memorização** do conjunto de treino. O Random Forest combate isso de duas formas:

1. **Bootstrap:** cada árvore vê apenas ~63% dos dados → nenhuma árvore memoriza o dataset inteiro
2. **Feature subsampling:** cada divisão considera apenas √16 ≈ 4 features → evita que todas as árvores se tornem cópias da mesma árvore dominante

A perda de 4.73 pp de acurácia da Decision Tree para o Random Forest é, em grande parte, efeito de **overfitting**: a árvore aprendeu padrões específicos do treino que não generalizam tão bem para o teste.

| Indicador | Decision Tree | Random Forest |
|-----------|:---:|:---:|
| Risco de overfitting | Alto (sem `max_depth`) | Baixo (ensemble) |
| Estabilidade (sensível a outliers) | Alta sensibilidade | Robusta |
| Variância do modelo | Alta | Baixa |
| Viés do modelo | Baixo | Baixo |

---

## 9. Interpretabilidade

Este é o único eixo onde a Decision Tree supera o Random Forest.

**Decision Tree:** cada caminho da raiz até uma folha é uma **regra de decisão legível**:

```
SE Weight > 97.5
  E Height ≤ 1.73
  E Gender = Female
ENTÃO → Obesity_Type_III
```

Essas regras podem ser extraídas, auditadas e explicadas para qualquer pessoa — inclusive médicos ou gestores de saúde pública.

**Random Forest:** a previsão final é resultado de votação de 200 árvores. Não existe uma única regra explicável. É possível obter importância global de features, mas não uma justificativa por amostra individual — a menos que se use ferramentas externas como SHAP ou LIME.

| Necessidade | Recomendação |
|-------------|-------------|
| Explicar *por que* um paciente foi classificado como obeso | Decision Tree |
| Maximizar acurácia da classificação | Random Forest |
| Auditoria e compliance médico | Decision Tree |
| Análise de quais hábitos de vida importam (visão global) | Random Forest |

---

## 10. Resumo Comparativo Final

| Dimensão | Decision Tree | Random Forest | Vencedor |
|----------|:---:|:---:|:---:|
| Acurácia geral | 91.25% | 95.98% | **RF** |
| Macro F1-Score | 0.91 | 0.96 | **RF** |
| Erros totais (teste) | 37 | 17 | **RF** |
| Normal_Weight (F1) | 0.80 | 0.91 | **RF** |
| Overweight_Level_I (F1) | 0.83 | 0.94 | **RF** |
| Obesity_Type_III (F1) | 0.99 | 0.99 | Empate |
| Tempo de treino | **2.6s** | 3.9s | **DT** |
| Memória | **Baixa** | Alta | **DT** |
| Interpretabilidade | **Alta** | Baixa | **DT** |
| Robustez a overfitting | Baixa | **Alta** | **RF** |
| Distribuição de importância | Concentrada | **Equilibrada** | **RF** |
| Revela hábitos de vida | Não | **Sim** | **RF** |

### Veredicto

**Para o objetivo deste projeto — classificar níveis de obesidade com máxima precisão — o Random Forest é o algoritmo recomendado**, superando a Decision Tree em todas as métricas de performance.

A Decision Tree, porém, cumpre papel fundamental como **baseline interpretável**: ela mostra que `Weight`, `Height` e `Gender` sozinhos já classificam 91.25% dos casos corretamente. O ganho do Random Forest (+4.73 pp) vem exatamente de capturar os padrões sutis nos hábitos alimentares e de atividade física que a árvore única ignora.

Juntos, os dois modelos respondem à pergunta central do projeto:

> *"O modelo consegue prever obesidade apenas a partir de dados antropométricos, ou os hábitos de vida contribuem significativamente?"*

**Resposta:** Dados antropométricos dominam. Mas os hábitos de vida contribuem de forma mensurável — e o Random Forest os captura melhor.

---

*Gerado em 25/04/2026 | Condições: Python 3.x, scikit-learn, random_state=42, split 80/20 estratificado*
