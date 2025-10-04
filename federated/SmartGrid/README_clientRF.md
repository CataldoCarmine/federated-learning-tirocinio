# Federated Random Forest Client - SmartGrid

## Descrizione

Implementazione di un client Flower per Random Forest federato basato sul paper "Random Forest Based on Federated Learning for Intrusion Detection". Questo client sostituisce il modello DNN con Random Forest mantenendo la stessa pipeline di preprocessing e compatibilità con il framework esistente.

## Caratteristiche Principali

### 1. Architettura Federata
- Ogni client addestra un Random Forest locale sui propri dati
- Gli alberi di decisione vengono serializzati e inviati al server centrale
- Il server aggrega gli alberi in un Global Random Forest
- Supporto per metodi di aggregazione basati su accuracy degli alberi

### 2. Preprocessing
Mantiene la stessa pipeline del client DNN con flag configurabili:
- `ENABLE_CLEAN_INF_NAN`: Pulizia valori infiniti/NaN (default: True)
- `ENABLE_CLIPPING_OUTLIERS`: Clipping outlier con IQR (default: True)
- `ENABLE_IMPUTATION`: Imputazione mediana (default: True)
- `ENABLE_SCALING`: StandardScaler (default: False)
- `ENABLE_REMOVE_NEAR_CONSTANT_FEATURES`: Rimozione feature quasi-costanti (default: False)
- `ENABLE_PCA`: Riduzione dimensionalità con PCA (default: False)

### 3. Configurazione Random Forest
```python
RF_N_ESTIMATORS = 100          # Numero di alberi (come nel paper)
RF_MAX_DEPTH = None            # Profondità massima (illimitata)
RF_MIN_SAMPLES_SPLIT = 2       # Campioni minimi per split
RF_MIN_SAMPLES_LEAF = 1        # Campioni minimi per foglia
RF_MAX_FEATURES = 'sqrt'       # Feature per ogni split
RF_BOOTSTRAP = True            # Bootstrap sampling
RF_CLASS_WEIGHT = 'balanced'   # Gestione sbilanciamento
RF_CRITERION = 'gini'          # Splitting rule ('gini' o 'entropy')
```

### 4. Serializzazione Alberi
- Utilizza `pickle` per serializzazione alberi scikit-learn
- Conversione in numpy arrays per compatibilità Flower
- Metadati inclusi: `tree_index`, `tree_size`, `accuracy`, `weighted_accuracy`
- Supporto per ricostruzione completa del Random Forest

### 5. Metriche
Il client traccia e invia le seguenti metriche:
- **Training**: accuracy, precision, recall, AUC, F1-score, balanced accuracy
- **Validation**: stesso set di metriche + confusion matrix
- **Per classe**: precision, recall, F1-score per natural/attack
- **Alberi**: accuracy e weighted accuracy di ogni albero

## Utilizzo

### Avvio Singolo Client
```bash
cd federated/SmartGrid
python clientRF.py <client_id>
```

Esempio:
```bash
python clientRF.py 1
```

Il client ID deve essere tra 1 e 13 (corrispondenti ai file data1.csv - data13.csv).

### Avvio Multiplo (tutti i client)
Per avviare tutti i client in parallelo, si può usare uno script simile a `run_clientsDNN.py`:
```bash
# Esempio per avviare 3 client in background
python clientRF.py 1 &
python clientRF.py 2 &
python clientRF.py 3 &
```

## Struttura del Client

### Classe `SmartGridRFClient`
Implementa `fl.client.NumPyClient` con tre metodi principali:

#### 1. `get_parameters(config)`
- Restituisce gli alberi serializzati del Random Forest locale
- Include metadati per ogni albero (accuracy, weighted accuracy)
- Formato compatibile con Flower (lista di numpy arrays)

#### 2. `fit(parameters, config)`
- Addestra un nuovo Random Forest sui dati locali
- Valuta accuratezza di ogni albero individualmente
- Serializza e restituisce gli alberi con metadati
- Restituisce: `(parameters, num_samples, metrics)`

#### 3. `evaluate(parameters, config)`
- Ricostruisce il Random Forest globale dai parametri ricevuti
- Valuta le performance sul validation set locale
- Calcola metriche complete e confusion matrix
- Restituisce: `(loss, num_samples, metrics)`

## Funzioni Principali

### Preprocessing
- `load_client_smartgrid_data(client_id)`: Carica e preprocessa i dati
- `clean_data_for_pca(X)`: Pulizia inf/NaN
- `fit_clip_outliers_iqr(X, k)`: Calcola limiti per clipping outlier
- `transform_clip_outliers_iqr(X, lower, upper)`: Applica clipping
- `remove_near_constant_features(X, ...)`: Rimuove feature costanti
- `apply_pca(X, client_id)`: Applica PCA con numero fisso di componenti

### Modello
- `create_random_forest_model()`: Crea Random Forest con configurazione ottimale
- `set_reproducibility_seeds()`: Imposta semi per riproducibilità

### Serializzazione
- `serialize_tree(tree)`: Serializza un singolo albero
- `deserialize_tree(tree_dict)`: Deserializza un singolo albero
- `serialize_random_forest_trees(rf_model, X_val, y_val)`: Serializza tutti gli alberi con metadati
- `deserialize_random_forest_trees(trees_data)`: Ricostruisce Random Forest da alberi serializzati

## Output e Logging

Il client fornisce logging dettagliato durante:
- Caricamento dati e preprocessing
- Creazione e training del modello
- Serializzazione alberi
- Valutazione e metriche

Esempio output:
```
=== AVVIO CLIENT RF 1 ===
=== PREPROCESSING FEDERATO RF ===
Pulizia inf/NaN: ABILITATA
Clipping outlier: ABILITATA
[Client 1] Distribuzione: 3866 attacchi (77.8%), 1100 naturali
[Client 1] Suddivisione: 3476 training, 1490 validation

[Client] === CREAZIONE RANDOM FOREST ===
[Client] N. estimatori: 100
[Client] Class weight: balanced

[Client 1] === ROUND DI ADDESTRAMENTO RF ===
[Client 1] Train Accuracy: 1.0000
[Client 1] Train F1: 1.0000, Balanced Acc: 1.0000

Serializzazione 100 alberi...
✅ 100 alberi serializzati
[Client 1] ✅ Invio 100 alberi al server
```

## Compatibilità

### Con il Framework Esistente
- Stessa struttura di directory dei dati (`data/SmartGrid/data*.csv`)
- Stesso formato delle labels (`marker != "Natural"`)
- Stessa pipeline di preprocessing del clientDNN
- Stesso sistema di riproducibilità (RANDOM_SEED = 42)

### Con Flower
- Implementa `fl.client.NumPyClient`
- Comunicazione via numpy arrays
- Connessione a `localhost:8080` (configurabile)
- Compatibile con strategie di aggregazione custom

## Metodi di Aggregazione

Supporta i metodi descritti nel paper:
1. **Simple Voting (SV)**: Ogni albero ha peso uguale
2. **Weighted Voting (WV)**: Peso basato su accuracy
3. **Sorting DTs per RF based on Accuracy (S_DTs_A)**: Ordinamento per accuracy
4. **Sorting DTs per RF based on Weighted Accuracy (S_DTs_WA)**: Ordinamento per weighted accuracy

I metadati `accuracy` e `weighted_accuracy` per ogni albero permettono al server di implementare questi metodi.

## Gestione Errori

Il client include gestione robusta degli errori per:
- Caricamento dati inesistenti o corrotti
- Errori durante preprocessing (fallback disponibili)
- Errori di serializzazione/deserializzazione
- Problemi di comunicazione con il server
- Validazione dei parametri ricevuti

Ogni errore produce logging dettagliato e traceback per il debug.

## Riproducibilità

Il client garantisce riproducibilità attraverso:
- Seed fisso `RANDOM_SEED = 42` per numpy e random
- Chiamate a `set_reproducibility_seeds()` in punti critici
- Seed per scikit-learn Random Forest
- Seed per PCA se abilitata
- Split train/validation deterministico

## Performance

Metriche tipiche su SmartGrid:
- **Training accuracy**: ~99-100% (overfitting su RF profondo)
- **Validation accuracy**: ~95-97%
- **F1-score**: ~97-98%
- **Balanced accuracy**: ~85-90%
- **Training time**: ~5-10 secondi per 100 alberi
- **Serializzazione**: ~1-2 secondi per 100 alberi

## Differenze con Client DNN

| Aspetto | ClientDNN | ClientRF |
|---------|-----------|----------|
| Modello | Neural Network (TensorFlow/Keras) | Random Forest (scikit-learn) |
| Epoche | 15 epoche locali | Training singolo |
| Early Stopping | Sì | No (non applicabile) |
| Callbacks | ReduceLROnPlateau, EarlyStopping | Nessuno |
| Class Weights | Passati a fit() | Gestiti automaticamente da RF |
| Parametri | Pesi neurali (numpy arrays) | Alberi serializzati |
| Loss | Binary crossentropy | Simulata (1 - accuracy) |
| Ottimizzatore | Adam/AdamW | Non applicabile |
| Dropout | Sì | No |

## Requisiti

- Python 3.7+
- flwr >= 1.0.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- (opzionale) tensorflow >= 2.8.0 (solo se PCA abilitata con seed TF)

## File Correlati

- `federated/SmartGrid/clientDNN.py`: Client DNN originale
- `centralized/SmartGrid/centralizedRF.py`: Versione centralizzata Random Forest
- `data/SmartGrid/data*.csv`: Dataset per ogni client

## Note Implementative

### Scelte di Design

1. **Serializzazione con Pickle**: Scelto per semplicità e compatibilità nativa con scikit-learn
2. **Formato Parametri Flower**: Array concatenati [metadata + tree_bytes] per ogni albero
3. **Accuracy per Albero**: Calcolata su validation set per supportare aggregazione pesata
4. **Ricostruzione RF**: Workaround per scikit-learn che non supporta nativamente creazione da alberi esistenti
5. **Loss Simulata**: Usato (1 - accuracy) come proxy per loss in evaluate()

### Limitazioni Conosciute

1. **Dimensione Parametri**: Alberi profondi possono generare parametri di grandi dimensioni
2. **Compatibilità Versioni**: Pickle richiede stessa versione scikit-learn su client e server
3. **Memoria**: Deserializzazione di molti alberi può richiedere memoria significativa
4. **n_features_in_**: Tutti i client devono avere stesso numero di feature dopo preprocessing

## Troubleshooting

### Errore: "File non trovato"
Verificare che il file `data/SmartGrid/data{client_id}.csv` esista.

### Errore: "PCA output shape inconsistente"
Disabilitare PCA o verificare che `PCA_COMPONENTS` sia compatibile con il numero di feature.

### Errore: "Deserializzazione fallita"
Verificare che client e server usino la stessa versione di scikit-learn.

### Warning: "class has only one sample"
Normale se un client ha solo una classe nei dati. Le metriche che richiedono entrambe le classi saranno 0.0.

## Riferimenti

- Paper: "Random Forest Based on Federated Learning for Intrusion Detection"
- Flower Framework: https://flower.ai/
- Scikit-learn Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
