# Federated Learning su Smart Grid ⚡️

Questo progetto implementa un sistema di **Federated Learning** applicato a un dataset di Smart Grid per la **rilevazione di attacchi informatici**. L'obiettivo è simulare un ambiente distribuito con più client che addestrano modelli localmente, contribuendo poi a un modello globale senza condividere i dati sensibili.

Il progetto include anche una **valutazione robusta della sicurezza adversarial** tramite attacchi White-Box, Black-Box e Gray-Box utilizzando la libreria **ART (Adversarial Robustness Toolbox)**.

---

## 📁 Struttura del Progetto

```
federated-learning-tirocinio/
├── attacks/                           # 🛡️ Modulo attacchi adversarial e difese
│   ├── defense_config.py              # Configurazione difesa adversarial training
│   ├── defense_utils.py               # Utility per difesa (vincoli fisici, feature importance)
│   ├── evaluation.py                  # Metriche di valutazione attacchi (ASR, L0, L2, L-inf)
│   ├── hsj_blackbox.py                # Attacco Black-Box query-based (HopSkipJump)
│   ├── hsj_graybox_transfer.py        # Attacco Gray-Box con modello surrogato
│   ├── hsj_whitebox.py                # Attacco White-Box sul modello federato
│   ├── test_adversarial_defense.py    # Script test efficacia difesa
│   └── utils.py                       # Utility comuni (caricamento modello, preprocessing)
│
├── centralized/                       # 🏢 Addestramento centralizzato (baseline)
│   └── SmartGrid/
│       ├── centralizedDNN.py          # Modello DNN centralizzato
│       ├── centralizedRF.py           # Modello Random Forest centralizzato
│       └── results/                   # Risultati addestramento centralizzato
│
├── federated/                         # 🌐 Addestramento federato
│   └── SmartGrid/
│       ├── clientDNN.py               # Client Flower per DNN
│       ├── clientRF.py                # Client Flower per Random Forest
│       ├── serverDNN.py               # Server Flower per DNN con FedAvg
│       ├── serverRF.py                # Server Flower per Random Forest con aggregazione alberi
│       ├── run_clientsRF.py           # Script per avvio multiprocesso client RF
│       ├── models/                    # Modelli federati salvati (.pkl, .h5)
│       ├── results/                   # Risultati training federato
│       └── results_EDA/               # Report EDA dataset SmartGrid
│
├── data/                              # 📊 Dataset suddivisi per client
│   └── SmartGrid/                     # Dataset SmartGrid Attack (client 1-15)
│
├── scripts/                           # 🔧 Script utility
│   ├── analyze_pca_components.py      # Analisi componenti PCA
│   ├── debug_rf_serialization.py      # Debug serializzazione RF
│   ├── optuna_optimizer.py            # Hyperparameter tuning con Optuna
│   ├── test_rf_integration.py         # Test integrazione RF federato
│   └── results_optuna/                # Risultati ottimizzazione Optuna
│
├── requirements.txt                   # Dipendenze Python
├── .gitignore                         # File esclusi dal versionamento
└── README.md                          # Questo file
```

---

## 🧠 Tecnologie Utilizzate

### Framework e Librerie Principali
- **[Flower](https://flower.dev/)** — Framework Federated Learning
- **[TensorFlow/Keras](https://www.tensorflow.org/)** — Deep Neural Networks (DNN)
- **[scikit-learn](https://scikit-learn.org/)** — Random Forest e preprocessing
- **[ART (Adversarial Robustness Toolbox)](https://adversarial-robustness-toolbox.org/)** — Attacchi adversarial (HopSkipJump)
- **[Optuna](https://optuna.org/)** — Hyperparameter optimization
- **pandas**, **numpy** — Manipolazione dati

### Modelli Implementati
1. **Deep Neural Network (DNN)** — Rete neurale profonda con regolarizzazione L2 e dropout
2. **Random Forest (RF)** — Ensemble di alberi decisionali con aggregazione federata

### Preprocessing Configurabile
- Pulizia valori infiniti/NaN
- Clipping outlier (IQR-based)
- Imputazione mediana
- Standardizzazione (StandardScaler)
- Riduzione dimensionalità (PCA)
- Rimozione feature quasi-costanti

---

## ⚙️ Setup Ambiente

### 1. Clona la repository

```bash
git clone https://github.com/CataldoCarmine/federated-learning-tirocinio.git
cd federated-learning-tirocinio
```

### 2. Installa le dipendenze

```bash
pip install -r requirements.txt
```

**Dipendenze principali:**
- `flwr` — Federated Learning
- `torch` — PyTorch (per alcune operazioni)
- `scikit-learn` — ML classico
- `pandas`, `numpy` — Data manipulation
- `tensorflow` — Deep Learning
- `adversarial-robustness-toolbox` — Attacchi adversarial

### 3. Prepara i dati

I dati devono essere posizionati in `data/SmartGrid/` con la seguente struttura:

```
data/SmartGrid/
├── smartgrid_client_1.csv
├── smartgrid_client_2.csv
├── ...
└── smartgrid_client_15.csv
```

**Dataset utilizzato:** SmartGrid Attack Dataset (15 client, 78,377 campioni totali, 128 feature)

---

## 🚀 Esecuzione

### **Addestramento Federato Random Forest**

#### 1. Avvia il server (in un terminale)
```bash
cd federated/SmartGrid/
python serverRF.py
```

#### 2. Avvia i client (in un altro terminale)

**Opzione A: Singolo client**
```bash
python clientRF.py <client_id>  # es: python clientRF.py 1
```

**Opzione B: Multipli client in parallelo**
```bash
python run_clientsRF.py  # Avvia automaticamente client 2-12 e 14-15
```

> **Nota:** Il client 1 e 13 sono riservati come **test set** per la valutazione.

---

### **Addestramento Federato DNN**

#### 1. Avvia il server
```bash
cd federated/SmartGrid/
python serverDNN.py
```

#### 2. Avvia i client
```bash
python clientDNN.py <client_id>
```

---

### **Addestramento Centralizzato (Baseline)**

Per confronto con l'approccio federato:

```bash
cd centralized/SmartGrid/

# Random Forest centralizzato
python centralizedRF.py

# DNN centralizzato
python centralizedDNN.py
```

---

## 🛡️ Valutazione Robustezza Adversarial

Il progetto include una suite completa di **attacchi adversarial** per testare la robustezza dei modelli federati.

### **Attacco White-Box** (Accesso completo al modello)

```bash
python attacks/hsj_whitebox.py \
    --model-path federated/SmartGrid/models/federated_rf_global_<timestamp>.pkl \
    --save-results
```

**Parametri:**
- `max_iter=100` — Iterazioni convergenza accurata
- `max_eval=10000` — Budget query generoso
- `norm=2` — Norma L2 (distanza euclidea)

---

### **Attacco Black-Box** (Solo accesso API predizioni)

```bash
python attacks/hsj_blackbox.py \
    --target-model-path federated/SmartGrid/models/federated_rf_global_<timestamp>.pkl \
    --save-results
```

**Parametri:**
- `max_iter=10` — Convergenza rapida (silenzioso)
- `max_eval=500` — Budget ridotto (evita rilevamento)
- Vincoli fisici SmartGrid applicati automaticamente

---

### **Attacco Gray-Box Transfer** (Modello surrogato)

```bash
python attacks/hsj_graybox_transfer.py \
    --target-model-path federated/SmartGrid/models/federated_rf_global_<timestamp>.pkl \
    --surrogate-clients 7 11 \
    --save-results
```

**Workflow:**
1. Addestra RF surrogato su dati pubblici (client 7, 11)
2. Genera adversarial examples sul surrogato
3. Testa transferability sul modello federato target

---

### **Test Difesa Adversarial Training**

```bash
# Test baseline (SENZA difesa)
python attacks/test_adversarial_defense.py --client-id 1 --disable-defense

# Test CON difesa adversarial training
python attacks/test_adversarial_defense.py --client-id 1

# Confronto automatico baseline vs robusto
python attacks/test_adversarial_defense.py --client-id 1 --compare
```

**Difese implementate:**
- ✅ Adversarial Training con cache intelligente
- ✅ Vincoli fisici SmartGrid (range valori realistici)
- ✅ Feature importance per vincoli adattivi

---

## 📊 Configurazione Preprocessing

Modifica i flag in `serverRF.py`, `clientRF.py` (o DNN) per personalizzare il preprocessing:

```python
# ============== FLAGS GLOBALI PER CONTROLLO PREPROCESSING ==============
ENABLE_CLEAN_INF_NAN = True           # Pulizia inf/NaN
ENABLE_CLIPPING_OUTLIERS = False      # Clipping outlier per quantili (IQR)
ENABLE_IMPUTATION = True              # Imputazione mediana
ENABLE_SCALING = True                 # StandardScaler (mean=0, std=1)
ENABLE_REMOVE_NEAR_CONSTANT_FEATURES = False  # Rimozione feature quasi-costanti
ENABLE_PCA = False                    # PCA per riduzione dimensionalità

# CONFIGURAZIONE PCA
PCA_COMPONENTS = 21                   # Numero componenti principali
PCA_RANDOM_SEED = 42                  # Seed per riproducibilità
```

---

## 📈 Risultati

### **Metriche Valutate**

#### Federated Learning:
- **Accuracy**, **Balanced Accuracy**
- **F1-Score** (totale, per classe)
- **Precision** e **Recall** (Natural/Attack)
- **AUC** (Area Under ROC Curve)

#### Attacchi Adversarial:
- **ASR (Attack Success Rate)** — % esempi Attack evasi → Natural
- **Perturbazione L0** — Numero feature modificate
- **Perturbazione L2** — Distanza euclidea media
- **Perturbazione L-inf** — Massimo cambiamento per feature

### **Report Generati Automaticamente**

Tutti gli esperimenti salvano report dettagliati in:

- `federated/SmartGrid/results/` — Training federato
- `centralized/SmartGrid/results/` — Training centralizzato
- `attacks/results/` — Attacchi adversarial (generato automaticamente)

**Esempio report attacco:**
```
===============================================================================
REPORT ATTACCO ADVERSARIAL - HopSkipJump White-Box
===============================================================================

STATISTICHE ATTACCO:
- Attack Success Rate (ASR): 87.5%
- Esempi adversarial generati: 350/400

METRICHE PERTURBAZIONE:
- Perturbazione L2 media: 0.0234
- Perturbazione L-inf media: 0.0089
- Feature modificate (L0): 12.3

PERFORMANCE MODELLO:
- Accuracy originale (su Attack): 95.2%
- Accuracy adversarial: 12.5%
```

---

## 🔧 Script Utility

### **Analisi PCA**
```bash
python scripts/analyze_pca_components.py
```
Analizza varianza spiegata per determinare numero ottimale componenti PCA.

### **Ottimizzazione Hyperparameter**
```bash
python scripts/optuna_optimizer.py
```
Usa Optuna per trovare configurazione ottimale DNN (learning rate, dropout, L2).

### **Test Integrazione Random Forest**
```bash
python scripts/test_rf_integration.py
```
Verifica compatibilità aggregazione alberi nel federato.

---

## 📝 Note Importanti

### **Riproducibilità**
Tutti gli esperimenti usano **seed fisso (42)** per garantire risultati riproducibili:
- NumPy, TensorFlow, scikit-learn, Python random
- `tf.config.experimental.enable_op_determinism()` per TensorFlow

### **Suddivisione Dati**
- **Client 1, 13:** Riservati come **test set** (mai usati in training)
- **Client 2-12, 14-15:** Usati per training federato
- **Client 7, 11:** Suggeriti per modello surrogato in attacchi Gray-Box

### **Configurazione Difese**
Modifica `attacks/defense_config.py` per controllare adversarial training:

```python
DEFENSE_CONFIG = {
    'ENABLE_ADVERSARIAL_TRAINING': True,  # ON/OFF difesa
    'EPSILON': 0.05,                       # Budget perturbazione
    'MAX_ADVERSARIAL_SAMPLES': 500,        # Campioni per training
    'ADVERSARIAL_RATIO': 0.5,              # 50% adv / 50% puliti
}
```

---

## 📚 Riferimenti

Questo progetto è parte di un **tirocinio universitario** focalizzato su:
- Federated Learning per privacy-preserving ML
- Robustezza adversarial in sistemi distribuiti
- Applicazioni Smart Grid per rilevazione intrusioni

---

## 👤 Autore

**Carmine Cataldo**  
📧 [GitHub: @CataldoCarmine](https://github.com/Carm1neBread)
