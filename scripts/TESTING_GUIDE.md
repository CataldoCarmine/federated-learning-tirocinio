# Random Forest Serialization - Quick Test Guide

## Test Rapidi

### 1. Test di Debug (Identifica il Problema)
```bash
cd /path/to/federated-learning-tirocinio
python3 scripts/debug_rf_serialization.py
```

**Output atteso**: Tutti i test passano, mostra che il problema era il mismatch client/server

**Durata**: ~5 secondi

---

### 2. Test End-to-End (Verifica la Soluzione)
```bash
cd /path/to/federated-learning-tirocinio
python3 scripts/test_rf_serialization_e2e.py
```

**Output atteso**: 
```
✅ TUTTI I TEST COMPLETATI CON SUCCESSO!
🎉 Il problema di serializzazione è RISOLTO!
```

**Durata**: ~10 secondi

---

### 3. Test di Integrazione (Opzionale - Server + Client Reali)
```bash
cd /path/to/federated-learning-tirocinio
python3 scripts/test_rf_integration.py
```

**Nota**: Richiede i dati SmartGrid. Avvia server e un client per 30 secondi.

**Durata**: ~40 secondi

---

## Test Completo del Sistema

### Passo 1: Avvia il Server

In un terminale:
```bash
cd federated/SmartGrid
python serverRF.py
```

Aspetta finché vedi:
```
🌳 Server Random Forest in attesa di client su localhost:8080...
```

### Passo 2: Avvia i Client

#### Opzione A: Un Client alla Volta (per debug)

In terminali separati:
```bash
cd federated/SmartGrid
python clientRF.py 1
```

Cerca nell'output:
- ✅ `Invio X alberi al server` → Serializzazione funziona
- ✅ `Modello aggregato ricevuto dal server` → Deserializzazione funziona
- ✅ `Training completato!` → Training funziona

#### Opzione B: Tutti i Client Insieme (per test completi)

In un nuovo terminale:
```bash
cd federated/SmartGrid
python run_clientsRF.py
```

Questo avvierà i client 1-13 in parallelo.

---

## Cosa Cercare nei Log

### Client - Serializzazione Corretta ✅
```
[Client] === SERIALIZZAZIONE ALBERI ===
[Client] ✅ Albero 1 serializzato (XXXX bytes)
[Client 1] Invio 65 alberi al server
[Client 1] Primo albero: shape=(XXXX,), dtype=uint8
```

### Server - Deserializzazione Corretta ✅
```
[Server] Ricevuti 65 parametri dal client
[Server] Tipo parametri ricevuti: <class 'numpy.ndarray'>
[Server] ✅ Albero 1 deserializzato correttamente (acc=0.XXX)
[Server] Deserializzati 65 alberi validi su 65
```

### Server - Aggregazione Corretta ✅
```
[Server] ✅ Random Forest globale creato con 100 alberi
[Server] Attributi configurati: n_features=XXX, n_classes=2
[Server] ✅ Aggregazione Random Forest completata
```

### Client - Ricezione Modello Globale ✅
```
[Client 1] Tipo parametro ricevuto: <class 'numpy.ndarray'>
[Client 1] Convertito numpy array in bytes: XXXXX bytes
[Client 1] ✅ Modello aggregato ricevuto dal server
[Client 1] Nuovo modello ha 100 alberi
```

---

## Risoluzione Problemi

### Errore: "unpickling stack underflow"
❌ **Causa**: Vecchia versione del codice con il bug
✅ **Soluzione**: Assicurati di usare la versione corretta del branch

### Errore: "ModuleNotFoundError: No module named 'numpy'"
❌ **Causa**: Dipendenze non installate
✅ **Soluzione**: 
```bash
pip install -r requirements.txt
# oppure
pip install numpy scikit-learn pandas flwr
```

### Errore: "File data1.csv non trovato"
❌ **Causa**: Dati SmartGrid non presenti
✅ **Soluzione**: Assicurati che i file CSV siano in `data/SmartGrid/`

### Server si blocca in attesa di client
✅ **Normale**: Il server aspetta almeno 2 client prima di iniziare
✅ **Soluzione**: Avvia almeno 2 client

---

## Verifica Rapida della Fix

Se vuoi solo verificare che la serializzazione funzioni:

```bash
# Test minimo (5 secondi)
python3 scripts/debug_rf_serialization.py

# Test completo (10 secondi)
python3 scripts/test_rf_serialization_e2e.py
```

Se entrambi mostrano `✅ SUCCESSO`, la fix è corretta!

---

## File Modificati

I file principali modificati per risolvere il bug:

1. **`federated/SmartGrid/clientRF.py`**
   - `get_parameters()`: Ora restituisce numpy arrays direttamente
   - `set_parameters()`: Ora gestisce numpy arrays correttamente

2. **`federated/SmartGrid/serverRF.py`**
   - `deserialize_trees_from_client()`: Usa `tobytes()` su numpy arrays
   - `serialize_global_model()`: Restituisce numpy arrays
   - `create_global_random_forest()`: Imposta attributi sklearn corretti

3. **Nuovi script di test**:
   - `scripts/debug_rf_serialization.py`
   - `scripts/test_rf_serialization_e2e.py`
   - `scripts/test_rf_integration.py`

---

## Report Completo

Per maggiori dettagli tecnici, consulta:
```
scripts/SERIALIZATION_FIX_REPORT.md
```
