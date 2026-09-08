# CatBoost per la predizione dei match di tennis

## Obiettivo

Il modello risolve un problema di **classificazione binaria**:

- `player_a_win = 1`: vince il giocatore A;
- `player_a_win = 0`: vince il giocatore B.

Il file principale è:

```text
scripts/models/train_catboost.py
```

Il codice è stato mantenuto volutamente semplice, in modo che ogni passaggio possa essere spiegato durante una presentazione orale.

## Flusso del programma

Il training segue sei passaggi:

```text
Dataset
   ↓
Feature X e target y
   ↓
Split cronologico Train/Test
   ↓
CatBoostClassifier
   ↓
model.fit()
   ↓
Prediction + Accuracy + ROC AUC
   ↓
Salvataggio modello
```

## 1. Caricamento dei dati

Il dataset finale viene letto da:

```text
data/interim/tennis_matches_features.data
```

Il file di metadata contiene l'elenco delle feature numeriche e categoriche:

```text
data/interim/tennis_matches_features_metadata.json
```

La matrice delle feature viene indicata con `X`, mentre il risultato del match viene indicato con `y`:

```python
X = data[feature_names]
y = data["player_a_win"]
```

## 2. Feature categoriche

CatBoost può utilizzare direttamente feature categoriche. Nel progetto, esempi di feature categoriche sono superficie, campo o torneo.

Le colonne categoriche vengono convertite in stringhe e gli eventuali valori mancanti vengono sostituiti con:

```text
__MISSING__
```

Non è quindi necessario applicare manualmente One-Hot Encoding prima del modello.

## 3. Split cronologico

Per un problema di predizione sportiva non è opportuno mischiare casualmente passato e futuro.

Il progetto usa quindi:

- l'85% delle date più vecchie per il training;
- il 15% delle date più recenti per il test.

Schema:

```text
passato                                      futuro
|---------------- TRAIN ----------------|------ TEST ------|
```

Il test simula quindi il caso reale in cui il modello viene addestrato sui match già disputati e utilizzato su match successivi.

## 4. Modello CatBoost

Il modello è definito con pochi iperparametri fissi:

```python
model = CatBoostClassifier(
    iterations=300,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=3,
    loss_function="Logloss",
    cat_features=categorical_features,
    random_seed=42,
    verbose=False,
)
```

### `iterations = 300`

Indica il numero di alberi costruiti dal boosting.

CatBoost non utilizza un solo Decision Tree. Costruisce più alberi in sequenza e ogni nuovo albero cerca di correggere gli errori dell'insieme precedente.

In forma semplificata:

\[
F_M(x) = F_0(x) + \eta h_1(x) + \eta h_2(x) + \dots + \eta h_M(x)
\]

### `depth = 8`

Indica la profondità massima degli alberi. Una profondità maggiore permette di rappresentare relazioni più complesse tra le feature.

### `learning_rate = 0.03`

Controlla quanto pesa ogni nuovo albero nella correzione del modello corrente.

Un valore piccolo rende l'apprendimento più graduale.

### `l2_leaf_reg = 3`

Introduce regolarizzazione sui valori delle foglie e aiuta a limitare l'overfitting.

### `loss_function = "Logloss"`

Per la classificazione binaria il modello minimizza la Log Loss.

Se `p` è la probabilità prevista che il giocatore A vinca, la loss penalizza soprattutto le previsioni molto sicure ma sbagliate.

## 5. Training

L'addestramento è eseguito con:

```python
model.fit(X_train, y_train)
```

Durante il boosting vengono costruiti sequenzialmente gli alberi che compongono il classificatore finale.

## 6. Predizione

La classe prevista viene ottenuta con:

```python
predictions = model.predict(X_test)
```

La probabilità di vittoria del giocatore A viene ottenuta con:

```python
probabilities = model.predict_proba(X_test)[:, 1]
```

Esempio:

```text
P(Player A wins) = 0.72
```

significa che il modello assegna al giocatore A una probabilità di vittoria del 72%.

## Metriche

### Accuracy

\[
Accuracy = \frac{predizioni\ corrette}{numero\ totale\ di\ predizioni}
\]

Misura la percentuale di match classificati correttamente.

### ROC AUC

La ROC AUC valuta quanto bene il modello ordina esempi positivi e negativi usando le probabilità predette.

Interpretazione semplice:

- `0.5`: comportamento simile a un classificatore casuale;
- `1.0`: separazione perfetta delle due classi.

## Output

Il training salva:

```text
data/interim/catboost_model.cbm
data/interim/catboost_metrics.json
```

Il primo file contiene il modello addestrato.

Il secondo contiene soltanto le informazioni necessarie per interpretare e riutilizzare il modello:

- algoritmo;
- feature numeriche;
- feature categoriche;
- Accuracy sul test;
- ROC AUC sul test.

## Risultati eseguiti — 2026-09-08

La tabella locale contiene 30.952 righe (15.476 partite), con 72 feature
numeriche e cinque categoriche. Il training usa 26.472 righe dal 2020-01-06 al
2025-08-07; il test usa 4.480 righe dal 2025-08-08 al 2026-07-12.

| Valutazione | Accuracy | ROC AUC | Log loss | Brier score |
| --- | ---: | ---: | ---: | ---: |
| CatBoost | 0.6946 | 0.7559 | 0.5834 | 0.2003 |
| Baseline di mercato normalizzata | 0.6915 | 0.7578 | 0.5818 | 0.1996 |
| Baseline maggioritaria | 0.5000 | — | — | — |

L'accuracy di training è 0.7062. Questa esecuzione usa parametri fissi, senza
ricerca o cross-validation. Riutilizza un holdout esistente, non un nuovo test
indipendente. Il piccolo miglioramento di accuracy rispetto al mercato non
dimostra un vantaggio affidabile; il mercato è migliore nelle altre tre metriche.

Entrambi i notebook eseguiti caricano il modello salvato e verificano che
accuracy e ROC AUC coincidano con il JSON di training. Le metriche aggiuntive
sono calcolate nei notebook. Le metriche principali usano le predizioni originali
per entrambi gli orientamenti; la diagnostica separata per partita media i due
orientamenti e ottiene accuracy 0.6915.

```bash
uv run python scripts/models/create_catboost_notebook.py --execute
uv run python scripts/models/create_catboost_notebook.py --language it --execute
```

Modello e dati restano locali in `data/interim/`; gli output eseguiti sono nel
[notebook inglese](catboost_training_analysis.ipynb) e nel
[notebook italiano](catboost_training_analysis_it.ipynb).

## Esecuzione

Dopo aver costruito il dataset delle feature:

```bash
uv run python scripts/models/train_catboost.py
```

Per eseguire i test:

```bash
uv run pytest
```

## Spiegazione breve per la presentazione

Una possibile spiegazione orale è:

> CatBoost è un algoritmo di Gradient Boosting basato su Decision Tree. Il modello costruisce più alberi in sequenza e ogni nuovo albero cerca di correggere gli errori prodotti dall'insieme precedente. Nel progetto lo utilizzo per un problema di classificazione binaria, dove la classe indica se il giocatore A vince o perde. Ho scelto CatBoost anche perché può gestire direttamente feature categoriche. Per simulare una predizione reale divido il dataset cronologicamente: addestro sui match più vecchi e valuto il modello sui match più recenti. Le metriche principali sono Accuracy e ROC AUC.

## Codice essenziale da ricordare

Il cuore del modello è concettualmente questo:

```python
X = data[feature_names]
y = data["player_a_win"]

model = CatBoostClassifier(
    iterations=300,
    depth=8,
    learning_rate=0.03,
    cat_features=categorical_features,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

Il resto del file serve soltanto a caricare i dati, mantenere lo split temporale e salvare il modello.
