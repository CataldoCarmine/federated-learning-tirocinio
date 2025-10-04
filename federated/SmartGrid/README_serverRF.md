# Server Random Forest Federato - SmartGrid

## Descrizione

`serverRF.py` implementa il server federato per Random Forest basato sul paper "Random Forest Based on Federated Learning for Intrusion Detection". Il server aggrega gli alberi di decisione dai client Random Forest e costruisce un Global Random Forest utilizzando diverse strategie di selezione.

## Architettura

### Workflow Federato

1. **Round di Training**:
   - Ogni client addestra un Random Forest locale sui propri dati
   - Gli alberi vengono serializzati con metadati (accuracy, weighted_accuracy)
   - Il client invia gli alberi al server

2. **Aggregazione sul Server**:
   - Il server deserializza gli alberi ricevuti da tutti i client
   - Applica una strategia di selezione basata sul paper
   - Costruisce un Global Random Forest con gli alberi selezionati
   - Serializza e invia il Global RF ai client

3. **Valutazione Globale**:
   - Il server valuta il Global RF su un dataset globale (client 14-15)
   - Calcola metriche complete (accuracy, F1, AUC, precision, recall, etc.)
   - Genera report delle metriche per ogni round

## Metodi di Aggregazione (dal Paper)

### 1. Selezione degli Alberi

#### Per Forest (`per_forest`)
- **S_DTs_A**: Seleziona i migliori alberi da ogni Random Forest client basato su **accuracy**
- **S_DTs_WA**: Seleziona i migliori alberi da ogni Random Forest client basato su **weighted accuracy**

#### Globale (`global`)
- **S_DTs_A_All**: Seleziona i migliori alberi globalmente tra tutti i client basato su **accuracy**
- **S_DTs_WA_All**: Seleziona i migliori alberi globalmente tra tutti i client basato su **weighted accuracy**

### 2. Metodi di Ensemble

- **Simple Voting**: Ogni albero ha peso uguale nelle predizioni
- **Weighted Voting**: Gli alberi sono pesati in base alla loro accuracy/weighted accuracy

## Configurazione

### Parametri Principali

```python
# Metodo di selezione degli alberi
TREE_SELECTION_METHOD = 'weighted_accuracy'  # 'accuracy' o 'weighted_accuracy'

# Strategia di aggregazione
TREE_AGGREGATION_STRATEGY = 'per_forest'     # 'per_forest' o 'global'

# Numero massimo di alberi nel modello globale
MAX_TREES_GLOBAL = 100

# Metodo di ensemble
ENSEMBLE_METHOD = 'weighted_voting'          # 'simple_voting' o 'weighted_voting'

# Numero di round di training federato
NUM_ROUNDS = 50
```

### Preprocessing

Il server utilizza la **stessa pipeline di preprocessing** del `serverDNN.py`:

```python
ENABLE_CLEAN_INF_NAN = True           # Pulizia inf/NaN
ENABLE_CLIPPING_OUTLIERS = True       # Clipping outlier (IQR)
ENABLE_IMPUTATION = True              # Imputazione mediana
ENABLE_SCALING = False                # StandardScaler
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Rimozione feature quasi-costanti
ENABLE_PCA = False                    # PCA fissa (74 componenti)
```

## Utilizzo

### 1. Avvio del Server

```bash
cd federated/SmartGrid
python serverRF.py
```

Il server si avvia su `localhost:8080` e attende almeno 2 client.

### 2. Output del Server

```
=== SERVER FEDERATO SMARTGRID RANDOM FOREST ===
Configurazione:
  - Rounds: 50
  - Client minimi: 2
  - Strategia: Random Forest Aggregation
  - Valutazione: Dataset globale senza PCA (client 14-15)
  - Pipeline: Pulizia → Imputazione → Normalizzazione → nessuna riduzione
  - Tree Selection Method: weighted_accuracy
  - Tree Aggregation Strategy: per_forest
  - Max Trees Global: 100
  - Ensemble Method: weighted_voting

Server Random Forest in attesa di client su localhost:8080...
```

### 3. Durante l'Aggregazione

Per ogni round, il server mostra:

```
=== AGGREGAZIONE TRAINING RF - ROUND 1 ===
Client partecipanti: 2
Client falliti: 0

=== METRICHE CLIENT ===
  Client 1: Accuracy=0.9234, F1=0.9156, Trees=100, Samples=1523
  Client 2: Accuracy=0.9187, F1=0.9098, Trees=100, Samples=1498

Deserializzazione alberi da Client 1...
  ✅ 100 alberi ricevuti, avg accuracy: 0.8945, avg weighted_accuracy: 0.8876

Deserializzazione alberi da Client 2...
  ✅ 100 alberi ricevuti, avg accuracy: 0.8923, avg weighted_accuracy: 0.8854

=== SELEZIONE ALBERI PER FOREST (metodo: weighted_accuracy) ===
  Client 1: selezionati 50/100 alberi, avg weighted_accuracy: 0.9123
  Client 2: selezionati 50/100 alberi, avg weighted_accuracy: 0.9087
✅ Totale alberi selezionati: 100

=== COSTRUZIONE GLOBAL RANDOM FOREST ===
Numero alberi: 100
Metodo ensemble: weighted_voting
Pesi alberi salvati (media: 0.0100)
✅ Global Random Forest creato con 100 alberi

✅ Aggregazione completata per round 1
✅ Global RF con 100 alberi creato e serializzato
```

### 4. Valutazione Globale

```
=== VALUTAZIONE GLOBALE RF - ROUND 1 ===
Deserializzazione Global RF da 100 parametri...
✅ Global RF ricostruito con 100 alberi

RISULTATI VALUTAZIONE:
  Loss: 0.0543
  Accuracy: 0.9457 (94.57%)
  F1-Score: 0.9589 (95.89%)
  Balanced Accuracy: 0.9312 (93.12%)
  Precision: 0.9501 (95.01%)
  Recall: 0.9678 (96.78%)
  AUC: 0.9876 (98.76%)
  Campioni test: 10391

Classification report (per classe):
              precision    recall  f1-score   support

     natural       0.93      0.91      0.92      3214
      attack       0.95      0.97      0.96      7177

    accuracy                           0.95     10391
   macro avg       0.94      0.94      0.94     10391
weighted avg       0.95      0.95      0.95     10391

Confusion matrix:
[[2925  289]
 [ 231 6946]]
```

## Report delle Metriche

Al termine del training, il server genera un report completo:

```
results/metrics_RF_complete_report_YYYYMMDD_HHMMSS.txt
```

Il report include:
- Metriche per round (accuracy, F1, precision, recall, AUC, balanced accuracy)
- Metriche per classe (natural vs attack)
- Confusion matrix finale

## Struttura del Codice

### Funzioni Principali

#### Preprocessing
- `apply_preprocessing_pipeline()`: Pipeline completa identica a serverDNN
- `clip_outliers_iqr()`: Clipping outlier con IQR
- `remove_near_constant_features()`: Rimozione feature costanti
- `apply_pca()`: PCA fissa (se abilitata)

#### Deserializzazione
- `deserialize_trees_from_client()`: Deserializza alberi da formato Flower
- `deserialize_tree_from_bytes()`: Deserializza singolo albero da pickle

#### Selezione e Aggregazione
- `select_trees_per_forest()`: Selezione per forest (S_DTs_A/S_DTs_WA)
- `select_trees_global()`: Selezione globale (S_DTs_A_All/S_DTs_WA_All)
- `create_global_random_forest()`: Costruzione Global RF
- `serialize_trees_for_clients()`: Serializzazione per invio ai client

#### Valutazione
- `get_smartgrid_evaluate_fn()`: Crea funzione di valutazione globale
- `load_global_test_data()`: Carica dataset test (client 14-15)
- `save_federated_metrics_report()`: Salva report metriche

### Classe Strategia

```python
class SmartGridRandomForestFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        """Aggrega alberi dai client e crea Global RF"""
        
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggrega risultati valutazione"""
        
    def _get_test_data(self):
        """Helper per caricare dataset test"""
```

## Testing

Un test suite completo è disponibile:

```bash
cd federated/SmartGrid
python test_serverRF.py
```

Il test verifica:
- ✓ Preprocessing functions
- ✓ Tree serialization/deserialization
- ✓ Tree selection methods (per_forest e global)
- ✓ Global RF construction
- ✓ Client-server compatibility
- ✓ Strategy class initialization

## Differenze con serverDNN.py

| Aspetto | serverDNN.py | serverRF.py |
|---------|-------------|-------------|
| **Modello** | Deep Neural Network | Random Forest |
| **Aggregazione** | Media pesata dei pesi | Selezione alberi migliori |
| **Parametri** | Weights & biases | Alberi serializzati |
| **Strategia** | FedAvg standard | Selezione basata su accuracy |
| **Ensemble** | N/A | Simple/Weighted Voting |
| **Preprocessing** | Identico | Identico |
| **Valutazione** | Identico | Identico |

## Compatibilità

### Con clientRF.py
✅ **Completamente compatibile**
- Formato serializzazione: `[tree_index, tree_size, accuracy, weighted_accuracy, tree_data...]`
- Il client serializza con `serialize_random_forest_trees()`
- Il server deserializza con `deserialize_trees_from_client()`

### Con serverDNN.py
✅ **Pipeline preprocessing identica**
- Stessi flag di configurazione
- Stesso dataset di valutazione (client 14-15)
- Stesso formato report metriche

## Vantaggi Random Forest Federato

1. **Privacy**: I dati rimangono locali sui client
2. **Diversità**: Alberi da diversi client catturano pattern diversi
3. **Robustezza**: Global RF combina migliori alberi da tutti i client
4. **Interpretabilità**: Random Forest più interpretabile di DNN
5. **Efficienza**: Nessun bisogno di retropropagazione o ottimizzazione gradiente

## Riferimenti

- Paper: "Random Forest Based on Federated Learning for Intrusion Detection"
- Client RF: `clientRF.py` e `README_clientRF.md`
- Server DNN: `serverDNN.py`
- Test Suite: `test_serverRF.py`

## Troubleshooting

### Errore: "Nessun albero valido ricevuto"
- Verificare che i client stiano inviando dati corretti
- Controllare che la serializzazione client sia corretta

### Errore: "Feature mismatch"
- Verificare che preprocessing sia identico su client e server
- Controllare che `ENABLE_PCA` sia configurato uguale

### Performance basse
- Aumentare `MAX_TREES_GLOBAL`
- Provare strategia `global` invece di `per_forest`
- Usare `weighted_accuracy` invece di `accuracy`

## Autore

Implementazione basata sul paper "Random Forest Based on Federated Learning for Intrusion Detection" per il progetto SmartGrid Intrusion Detection.
