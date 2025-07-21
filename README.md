# Federated Learning su Smart Grid ⚡️

Questo progetto implementa un sistema di Federated Learning applicato a un dataset di smart grid per la rilevazione di attacchi informatici. L'obiettivo è simulare un ambiente distribuito con più client che collaborano all'addestramento di un modello di classificazione, preservando la privacy dei dati locali.

## 📁 Struttura del Progetto

- `client/` — Codice per i client FL
- `server/` — Codice per il server FL
- `data/` — Dataset CSV suddiviso per client
- `utils/` — Funzioni ausiliarie per preprocessing e metriche
- `requirements.txt` — Dipendenze Python
- `.gitignore` — File e cartelle esclusi dal versionamento
- `README.md` — Questo file
- `main.py` — Entry point per l'esecuzione

## 🧠 Tecnologie Utilizzate

- [Flower](https://flower.dev/) per il framework di Federated Learning
- `scikit-learn`, `pandas`, `tensorflow`, `numpy` per la parte ML
- Python 3.10+

## ⚙️ Setup Ambiente

### 1. Clona la repository

```bash
git clone https://github.com/CataldoCarmine/federated-learning-tirocinio.git
cd federated-learning-tirocinio
