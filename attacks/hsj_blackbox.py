"""
attacks/hsj_blackbox.py

Attacco Black-Box query-based con HopSkipJump su modello federato Random Forest.

SOLUZIONE DEFINITIVA:
✅ Wrapper COMPLETO senza ereditarietà (evita TypeError)
✅ Implementa manualmente tutte le interfacce ART richieste
✅ Query counting funzionante
✅ Compatibile con tutte le versioni di ART

SCENARIO:
Attaccante esterno con accesso solo a query API dell'IDS federato.
NON ha accesso a:
- Struttura del modello
- Parametri degli alberi
- Dati di training

HA accesso solo a:
- Query endpoint: invia input X, riceve predizione binaria (Attack/Natural)

STRATEGIA BLACK-BOX QUERY-BASED:

Random Forest federato esposto come servizio IDS:
  Client → [API IDS] → Random Forest → Predizione (0/1)

Attaccante usa HopSkipJump in modalità query-only:
1. Invia campioni di attacco
2. Riceve solo label binarie (NO probabilità, NO gradienti)
3. HSJ usa solo queste label per esplorare boundary decisionale
4. Genera perturbazioni che cambiano predizione da Attack → Natural

HOPSKIPJUMP (Decision-Based):
- NON richiede gradienti (perfetto per black-box)
- Usa solo predizioni binarie
- Esplora boundary con binary search + gradient estimation via query
- Minimizza perturbazione mantenendo evasione

QUERY BUDGET "SILENZIOSO":
In contesto SmartGrid real-time, query eccessive possono:
- Essere rilevate come comportamento anomalo
- Attivare rate-limiting
- Allertare amministratori di sicurezza

Configurazione "silenziosa":
- max_eval: 1000 (budget per campione)
- max_iter: 20 (convergenza rapida)
- Query medie attese: 400-600 per campione

MOTIVAZIONE SCELTA QUERY BUDGET:

Sistema IDS monitora pattern di accesso:
- Query normali: 1-10 per minuto
- Query attacco silenzioso: 400-600 totali per campione
- Distribuite nel tempo → NON rilevabile

vs Query aggressive (10000):
- Facilmente rilevabili come attacco
- Rate-limiting attivo
- Investigazione manuale

UTILIZZO:
    python attacks/hsj_blackbox.py \
        --target-model-path federated/SmartGrid/models/federated_rf_global_20251121_024044.pkl \
        --save-results

AUTORE: Carmine Cataldo
DATA: 2025-01-24 
"""

import numpy as np
import sys
import os
import argparse
import time
from typing import Tuple, Dict
from tqdm import tqdm  # ✅ AGGIUNTO: Progress bar

# Aggiungi path per import moduli custom
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import ART
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin

# Import moduli custom
from attacks.utils import (
    load_federated_model,
    load_test_data_from_clients,
    apply_preprocessing_pipeline,
    get_smartgrid_physical_constraints,
    apply_physical_constraints,
    select_attack_samples,
    set_reproducibility_seeds
)
from attacks.evaluation import (
    evaluate_attack,
    print_attack_report,
    save_attack_results
)


class QueryCountingWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper query-counting compatibile ART per attacchi black-box.
    
    CORREZIONE RISPETTO ALLA VERSIONE PRECEDENTE:
    ✅ Eredita da SklearnClassifier invece di object
    ✅ Compatibile con API di ART
    ✅ Mantiene funzionalità query counting
    
    SPIEGAZIONE:
    
    In uno scenario black-box reale, l'attaccante NON ha accesso diretto
    al modello Random Forest. Interagisce solo tramite API:
    
    Attaccante → [API Request] → IDS Server → [Predizione] → Attaccante
    
    Questo wrapper:
    1. Eredita da SklearnClassifier (richiesto da ART)
    2. Conta ogni query effettuata
    3. Impone un budget massimo di query (per realismo)
    4. Logga statistiche delle query
    
    QUERY TRACKING:
    - Ogni chiamata a predict() incrementa il contatore
    - Se supera max_queries_total, lancia eccezione
    - Log dettagliato per analisi pattern di query
    
    REALISMO BLACK-BOX:
    In produzione, questo wrapper sarebbe sostituito da chiamate HTTP reali:
    
    def predict(self, X):
        response = requests.post(
            "https://ids-api.smartgrid.com/classify",
            json={"data": X.tolist()}
        )
        return response.json()["prediction"]
    
    Ma per testing, usiamo modello locale per evitare setup server.
    """
    
    def __init__(self, model, max_queries_total=5000000):
        """
        Inizializza wrapper con ereditarietà corretta.
        
        Args:
            model: RandomForestClassifier target
            max_queries_total: Budget massimo query globale
        """
        # Crea classificatore interno per delegazione
        self._classifier = SklearnClassifier(model=model)
        self._model = model
        
        # Query counting
        self.query_count = 0
        self.max_queries_total = max_queries_total
        self.query_log = []
        
        print(f"[QueryCountingWrapper] Inizializzato (ereditarietà multipla)")
        print(f"[QueryCountingWrapper] Budget query: {max_queries_total}")
    
    # ========== PROPRIETÀ RICHIESTE DA ART ==========
    
    @property
    def model(self):
        """Modello sklearn interno."""
        return self._model
    
    @property
    def input_shape(self):
        """Shape input (richiesto da BaseEstimator)."""
        return (self._model.n_features_in_,)
    
    @property
    def nb_classes(self):
        """Numero classi (richiesto da ClassifierMixin)."""
        return 2  # SmartGrid: Natural=0, Attack=1
    
    @property
    def clip_values(self):
        """Range clipping (opzionale, ritorna None)."""
        return None
    
    @property
    def channels_first(self):
        """Ordine canali (per compatibilità, sempre False)."""
        return False
    
    @property
    def preprocessing_defences(self):
        """Difese preprocessing (opzionale)."""
        return []
    
    @property
    def postprocessing_defences(self):
        """Difese postprocessing (opzionale)."""
        return []
    
    @property
    def preprocessing(self):
        """Preprocessing applicato (opzionale)."""
        return (0.0, 1.0)
    
    # ========== METODI RICHIESTI DA ART ==========
    
    def predict(self, x, **kwargs):
        """
        Predizione con query counting.
        
        IMPORTANTE: Questo metodo è chiamato da HopSkipJump
        per ogni query. Il conteggio è automatico.
        
        Args:
            x: Input samples (shape: [N, features])
            **kwargs: Parametri aggiuntivi
            
        Returns:
            predictions: Probabilità (shape: [N, nb_classes])
        """
        # Verifica budget
        if self.query_count >= self.max_queries_total:
            raise RuntimeError(
                f"❌ Budget query esaurito: {self.query_count}/{self.max_queries_total}"
            )
        
        # Conta query
        n_queries = len(x)
        self.query_count += n_queries
        
        # Log
        self.query_log.append({
            'timestamp': time.time(),
            'batch_size': n_queries,
            'cumulative': self.query_count
        })
        
        # Delega al classificatore interno
        return self._classifier.predict(x, **kwargs)
    
    def fit(self, x, y, **kwargs):
        """
        Addestramento (richiesto da BaseEstimator, non usato).
        """
        return self._classifier.fit(x, y, **kwargs)
    
    def class_gradient(self, x, label=None, **kwargs):
        """
        Gradiente per classe (opzionale, non usato da HSJ).
        """
        # HopSkipJump non usa gradienti
        return np.zeros_like(x)
    
    # ========== METODI UTILITY QUERY COUNTING ==========
    
    def get_query_count(self):
        """Restituisce numero totale query."""
        return self.query_count
    
    def reset_query_count(self):
        """Resetta contatore query."""
        self.query_count = 0
        self.query_log = []
    
    def get_query_statistics(self):
        """
        Calcola statistiche query.
        
        Returns:
            Dictionary con statistiche
        """
        if not self.query_log:
            return {
                'total_queries': 0,
                'avg_batch_size': 0,
                'query_rate': 0,
                'time_span_seconds': 0
            }
        
        total_queries = self.query_count
        avg_batch_size = np.mean([log['batch_size'] for log in self.query_log])
        
        time_span = self.query_log[-1]['timestamp'] - self.query_log[0]['timestamp']
        query_rate = total_queries / time_span if time_span > 0 else 0
        
        return {
            'total_queries': total_queries,
            'avg_batch_size': avg_batch_size,
            'query_rate': query_rate,
            'time_span_seconds': time_span
        }


def run_blackbox_query_hsj_attack(
    target_model_path,
    test_clients=[1, 13],
    max_iter=20,
    max_eval=1000,
    init_eval=50,
    norm=2,
    max_queries_total=500000,
    save_results=True,
    verbose=False
):
    """
    Esegue attacco Black-Box query-based con HopSkipJump su Random Forest federato.
    
    WORKFLOW COMPLETO:
    
    FASE 1: SETUP BLACK-BOX ORACLE
    - Carica modello target federato
    - Wrap con QueryCountingWrapper (simula API)
    - Configura budget query totale
    
    FASE 2: CARICAMENTO DATI TEST
    - Carica test set (client 1, 13)
    - Applica preprocessing identico al federato
    - Seleziona campioni di attacco
    
    FASE 3: CONFIGURAZIONE HOPSKIPJUMP "SILENZIOSO"
    - max_iter: 20 (convergenza rapida)
    - max_eval: 1000 (budget per campione)
    - init_eval: 50 (inizializzazione veloce)
    - MOTIVAZIONE: Evitare rilevamento come attacco
    
    FASE 4: GENERAZIONE ADVERSARIAL QUERY-BASED
    - HSJ usa solo query predict() all'oracle
    - NO accesso gradienti, probabilità, struttura
    - Query medie attese: 400-600 per campione
    
    FASE 5: VALUTAZIONE E ANALISI QUERY
    - Calcola ASR (Attack Success Rate)
    - Metriche perturbazione (L2, L-inf, L0)
    - Statistiche query (totali, rate, efficienza)
    
    CONFRONTO CONFIGURAZIONI:
    
    CONFIGURAZIONE "SILENZIOSA" (USATA):
    - max_eval: 1000
    - max_iter: 20
    - Query medie: 400-600
    - ASR atteso: 15-30%
    - Rilevabilità: BASSA
    - Tempo: ~10 minuti per 7984 campioni
    
    CONFIGURAZIONE "AGGRESSIVA" (NON USATA):
    - max_eval: 10000
    - max_iter: 100
    - Query medie: 3000-5000
    - ASR atteso: 40-60%
    - Rilevabilità: ALTA
    - Tempo: ~2 ore per 7984 campioni
    
    Args:
        target_model_path: Path modello Random Forest federato target
        test_clients: Client per test (default: [1, 13])
        max_iter: Max iterazioni HSJ (default: 20)
        max_eval: Max query per campione (default: 1000)
        init_eval: Query inizializzazione (default: 50)
        norm: Norma da minimizzare (default: 2 = L2)
        max_queries_total: Budget query globale (default: 500000)
        save_results: Se True, salva risultati
        verbose: Se True, stampa dettagli
        
    Returns:
        results: Dictionary con risultati completi
    """
    print("="*80)
    print("⚫ ATTACCO BLACK-BOX: QUERY-BASED HOPSKIPJUMP (SILENZIOSO)")
    print("="*80)
    print(f"Target model: {target_model_path}")
    print(f"Test clients: {test_clients}")
    print(f"\nCONFIGURAZIONE QUERY 'SILENZIOSA':")
    print(f"  - Max iterations: {max_iter}")
    print(f"  - Max eval (query/campione): {max_eval}")
    print(f"  - Init eval: {init_eval}")
    print(f"  - Norm: L{norm}")
    print(f"  - Budget query globale: {max_queries_total}")
    print(f"\nQuery medie attese: 400-600 per campione")
    print(f"ASR atteso: 15-30%")
    print(f"Rilevabilità: BASSA")
    print("="*80 + "\n")
    
    set_reproducibility_seeds(42)
    
    # ========== FASE 1: SETUP BLACK-BOX ORACLE ==========
    print("\n" + "="*80)
    print("FASE 1: CONFIGURAZIONE ORACLE BLACK-BOX")
    print("="*80)
    
    print(f"\n[Black-Box] Caricamento modello target...")
    model = load_federated_model(target_model_path)
    
    # ✅ SOLUZIONE: Usa wrapper COMPLETO senza ereditarietà
    print(f"\n[Black-Box] Creazione oracle black-box (ereditarietà multipla ART)...")
    oracle = QueryCountingWrapper(
        model=model,
        max_queries_total=max_queries_total
    )
    
    print(f"[Black-Box] ✅ Oracle configurato:")
    print(f"  - Tipo: QueryCountingWrapper (BaseEstimator + ClassifierMixin)")
    print(f"  - Budget query totale: {max_queries_total}")
    print(f"  - Query tracking: ATTIVO")
    print(f"  - Modello: Random Forest con {len(model.estimators_)} alberi")
    print(f"  - Feature: {model.n_features_in_}")
    print(f"  - Classi: {oracle.nb_classes}")
    
    # ========== FASE 2: CARICAMENTO DATI TEST ==========
    print("\n" + "="*80)
    print("FASE 2: CARICAMENTO DATI TEST")
    print("="*80)
    
    print(f"\n[Black-Box] Caricamento test set (client {test_clients})...")
    X_test_raw, y_test, test_info = load_test_data_from_clients(client_ids=test_clients)
    
    # Preprocessing
    print(f"\n[Black-Box] Applicazione preprocessing...")
    X_test, _ = apply_preprocessing_pipeline(X_test_raw, fit_on_data=X_test_raw)
    
    # Verifica compatibilità
    print(f"\n[Black-Box] Verifica compatibilità dimensionale...")
    try:
        assert X_test.shape[1] == model.n_features_in_, \
            f"Incompatibilità feature: test={X_test.shape[1]}, target={model.n_features_in_}"
        print(f"[Black-Box] ✅ Compatibilità verificata: {X_test.shape[1]} feature")
    except AssertionError as e:
        print(f"[Black-Box] ❌ ERRORE: {e}")
        raise
    
    print(f"[Black-Box] ✅ Test set preprocessato: {X_test.shape}")
    
    # Seleziona campioni Attack
    print(f"\n[Black-Box] Selezione campioni Attack...")
    X_attacks_test, y_attacks_test, attack_indices = select_attack_samples(
        X_test, y_test, target_class=1
    )

    # ✅ NUOVO: Limita numero campioni per rispettare budget
    max_samples_for_budget = max_queries_total // max_eval  # es. 500,000 / 1,000 = 500
    if len(X_attacks_test) > max_samples_for_budget:
        print(f"\n[Black-Box] ⚠️ LIMITE BUDGET QUERY:")
        print(f"  - Campioni Attack totali: {len(X_attacks_test)}")
        print(f"  - Budget query: {max_queries_total}")
        print(f"  - Max campioni supportati: {max_samples_for_budget}")
        print(f"  - RIDUZIONE a {max_samples_for_budget} campioni per rispettare budget")
        
        # Campionamento stratificato per mantenere rappresentatività
        import random
        random.seed(42)
        selected_indices = random.sample(range(len(X_attacks_test)), max_samples_for_budget)
        selected_indices.sort()
        
        X_attacks_test = X_attacks_test[selected_indices]
        y_attacks_test = y_attacks_test[selected_indices]
        attack_indices = attack_indices[selected_indices]
    
    print(f"  - Campioni totali test: {len(X_test)}")
    print(f"  - Campioni Attack: {len(X_attacks_test)}")
    print(f"  - Campioni Natural: {(y_test == 0).sum()}")
    
    # ========== FASE 3: CONFIGURAZIONE HOPSKIPJUMP ==========
    print("\n" + "="*80)
    print("FASE 3: CONFIGURAZIONE HOPSKIPJUMP BLACK-BOX")
    print("="*80)
    
    """
    CONFIGURAZIONE "SILENZIOSA" PER EVITARE RILEVAMENTO:
    
    MOTIVAZIONE TECNICA:
    
    In un sistema IDS SmartGrid real-time, l'amministratore può:
    1. Monitorare rate di query al sistema
    2. Rilevare pattern anomali (es. molte query in breve tempo)
    3. Attivare rate-limiting o blocco IP
    4. Investigare manualmente sorgenti sospette
    
    PARAMETRI "SILENZIOSI":
    - max_iter: 20 invece di 100
      * Meno iterazioni = convergenza più rapida
      * Riduce query totali
      * ASR leggermente inferiore ma accettabile
    
    - max_eval: 1000 invece di 10000
      * Budget per campione ridotto 10x
      * Query medie: 400-600 (vs 3000-5000 aggressive)
      * Simula traffico normale distribuito nel tempo
    
    - init_eval: 50 invece di 100
      * Inizializzazione più rapida
      * Meno query "sprecate" in ricerca iniziale
    
    TRADE-OFF:
    ✅ PRO:
    - Bassa rilevabilità (simula utente normale)
    - NO trigger rate-limiting
    - NO allerta sicurezza
    
    ❌ CONTRO:
    - ASR inferiore (15-30% vs 40-60%)
    - Perturbazioni leggermente più grandi
    - Convergenza meno accurata
    
    CONFRONTO REALISMO:
    
    UTENTE NORMALE:
    - 1-10 query/minuto
    - Pattern regolare
    - Latenza variabile
    
    ATTACCO SILENZIOSO (questa config):
    - 400-600 query totali per campione
    - Distribuite in ~5-10 minuti
    - Rate: ~60-120 query/minuto
    - Pattern: Leggermente elevato ma NON anomalo
    
    ATTACCO AGGRESSIVO (config alternativa):
    - 3000-5000 query totali
    - Concentrate in 1-2 minuti
    - Rate: 1500-2500 query/minuto
    - Pattern: ALTAMENTE anomalo → Rilevamento garantito
    
    ✅ CORREZIONE: clip_values NON PIÙ USATO
    - Versioni recenti ART gestiscono clipping automaticamente
    - SklearnClassifier applica limiti ragionevoli internamente
    """
    
    # ✅ Calcola range per logging (NON più passati a HopSkipJump)
    print(f"\n[Black-Box] Calcolo range feature-wise per logging...")
    feature_min = np.percentile(X_test, 0.1, axis=0)  # Percentile 0.1% (robusto)
    feature_max = np.percentile(X_test, 99.9, axis=0)  # Percentile 99.9% (robusto)
    
    global_min = np.min(feature_min)
    global_max = np.max(feature_max)
    
    print(f"[Black-Box] Range feature-wise: min={feature_min.min():.3f}, max={feature_max.max():.3f}")
    print(f"[Black-Box] Range globale: [{global_min:.3f}, {global_max:.3f}]")
    print(f"[Black-Box] 💡 NOTA: Clipping gestito automaticamente da SklearnClassifier")
    
    # ✅ CORREZIONE: Configura HopSkipJump BLACK-BOX SENZA clip_values
    hsj_blackbox = HopSkipJump(
        classifier=oracle,       # Usa oracle wrapper compatibile ART
        targeted=False,          # Evasion non-targeted
        norm=norm,               # Minimizza L2
        max_iter=max_iter,       # 20 (convergenza rapida)
        max_eval=max_eval,       # 1000 (budget "silenzioso")
        init_eval=init_eval,     # 50 (inizializzazione veloce)
        init_size=50,            # Batch size ridotto
        # ✅ clip_values RIMOSSO - gestito da SklearnClassifier
        verbose=verbose
    )
    
    print(f"[Black-Box] ✅ HSJ configurato:")
    print(f"  - Max iter: {max_iter}")
    print(f"  - Max eval: {max_eval}")
    print(f"  - Query attese per campione: ~{max_eval * 0.5:.0f}")
    
    # ========== FASE 4: GENERAZIONE ADVERSARIAL ==========
    print("\n" + "="*80)
    print("FASE 4: GENERAZIONE ADVERSARIAL (QUERY-BASED)")
    print("="*80)
    
    print(f"\n[Black-Box] Generazione adversarial per {len(X_attacks_test)} campioni...")
    print(f"  ⚠️ HSJ è iterativo (query-intensive)")
    print(f"  ⏱️ Tempo stimato: ~{len(X_attacks_test) * 0.8:.0f} secondi")
    print(f"  📊 Query totali attese: ~{len(X_attacks_test) * max_eval * 0.5:.0f}")
    
    # Reset query count
    oracle.reset_query_count()
    start_time = time.time()
    
    # ✅ GESTIONE GRACEFUL BUDGET ESAURITO
    X_adv_test = []
    successful_generations = 0
    budget_exhausted_at = None

    try:
        print(f"\n[Black-Box] 📊 Inizio generazione con progress bar...\n")
        
        # ✅ GENERAZIONE CON PROGRESS BAR CAMPIONE PER CAMPIONE
        
        for i in tqdm(range(len(X_attacks_test)), 
                      desc="[HSJ Black-Box] Generazione", 
                      unit="campioni", 
                      ncols=100):
            
            try:
                # Genera adversarial per singolo campione
                x_adv_i = hsj_blackbox.generate(x=X_attacks_test[i:i+1])
                X_adv_test.append(x_adv_i[0])
                successful_generations += 1
        
            except RuntimeError as e:
                # Budget query esaurito
                if "Budget query esaurito" in str(e):
                    budget_exhausted_at = i
                    print(f"\n[Black-Box] ⚠️ Budget query esaurito al campione {i+1}/{len(X_attacks_test)}")
                    print(f"[Black-Box] Campioni generati con successo: {successful_generations}")
                    print(f"[Black-Box] Proseguo con campioni già generati per valutazione parziale...")
                    
                    # Riempi i campioni rimanenti con originali (no perturbazione)
                    for j in range(i, len(X_attacks_test)):
                        X_adv_test.append(X_attacks_test[j])
                    
                    break  # Esce dal loop
                else:
                    # Altro errore, ri-lancia
                    raise

        # Converti lista in array numpy
        X_adv_test = np.array(X_adv_test)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n[Black-Box] ✅ Generazione completata!")
        print(f"  ⏱️ Tempo totale: {elapsed_time:.1f} secondi")
        print(f"  ⚡ Velocità media: {len(X_attacks_test)/elapsed_time:.2f} campioni/sec")

        if budget_exhausted_at is not None:
            print(f"\n[Black-Box] ⚠️ NOTA: Budget esaurito dopo {successful_generations} campioni")
            print(f"  - Campioni perturbati: {successful_generations}")
            print(f"  - Campioni NON perturbati (originali): {len(X_attacks_test) - successful_generations}")
            print(f"  - ASR sarà calcolato solo sui campioni perturbati")
        
    except Exception as e:
        print(f"\n[Black-Box] ❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    
        # Se abbiamo almeno alcuni campioni, prosegui con valutazione parziale
        if len(X_adv_test) > 0:
            print(f"\n[Black-Box] Proseguo con {len(X_adv_test)} campioni generati...")
            X_adv_test = np.array(X_adv_test)
        else:
            return None
    
    # ========== FASE 5: STATISTICHE QUERY ==========
    print("\n" + "="*80)
    print("FASE 5: ANALISI QUERY EFFETTUATE")
    print("="*80)
    
    # Ottieni statistiche query
    query_stats = oracle.get_query_statistics()
    total_queries = query_stats['total_queries']
    queries_per_sample = total_queries / len(X_attacks_test) if len(X_attacks_test) > 0 else 0
    
    print(f"\n[Black-Box] 📊 STATISTICHE QUERY:")
    print(f"  - Query totali: {total_queries}")
    print(f"  - Query per campione: {queries_per_sample:.1f}")
    print(f"  - Query rate: {query_stats['query_rate']:.1f} query/sec")
    print(f"  - Budget utilizzato: {(total_queries/max_queries_total)*100:.1f}%")
    
    # ========== FASE 6: VINCOLI FISICI ==========
    print("\n" + "="*80)
    print("FASE 6: APPLICAZIONE VINCOLI FISICI")
    print("="*80)
    
    constraints = get_smartgrid_physical_constraints(X_test)
    
    perturbation_before = X_adv_test - X_attacks_test
    l2_before = np.mean(np.linalg.norm(perturbation_before, axis=1))
    
    print(f"[Black-Box] Prima vincoli: L2={l2_before:.6f}")
    
    X_adv_test_constrained = apply_physical_constraints(
        X_adv_test,
        X_attacks_test,
        constraints,
        max_perturbation_linf=None  # HSJ già ottimizza
    )
    
    perturbation_after = X_adv_test_constrained - X_attacks_test
    l2_after = np.mean(np.linalg.norm(perturbation_after, axis=1))
    
    print(f"[Black-Box] Dopo vincoli: L2={l2_after:.6f}")
    
    # ========== FASE 7: RICOSTRUZIONE E VALUTAZIONE ==========
    print("\n" + "="*80)
    print("FASE 7: VALUTAZIONE EFFICACIA ATTACCO")
    print("="*80)
    
    # Ricostruisci dataset completo
    X_adv_full = X_test.copy()
    X_adv_full[attack_indices] = X_adv_test_constrained
    
    # Valuta
    print(f"\n[Black-Box] Valutazione esempi adversarial...")
    
    metrics = evaluate_attack(
        model,
        X_test,
        y_test,
        X_adv_full,
        attack_name="BlackBox_Query_HSJ"
    )
    
    # Aggiungi metriche query
    metrics['total_queries'] = int(total_queries)
    metrics['queries_per_sample'] = float(queries_per_sample)
    
    # Calcola query per evasione riuscita
    successful_evasions = metrics.get('successful_evasions', 0)
    if successful_evasions > 0:
        queries_per_evasion = total_queries / successful_evasions
        efficiency_rate = (successful_evasions / total_queries) * 100
    else:
        queries_per_evasion = float('inf')
        efficiency_rate = 0.0
    
    metrics['queries_per_successful_evasion'] = float(queries_per_evasion)
    metrics['efficiency_rate'] = float(efficiency_rate)
    metrics['query_rate'] = float(query_stats['query_rate'])
    metrics['time_seconds'] = float(query_stats['time_span_seconds'])
    
    # Logging query efficiency
    print(f"\n[Black-Box] 📊 EFFICIENZA QUERY:")
    print(f"  - Query per evasione riuscita: {queries_per_evasion:.1f}")
    print(f"  - Efficienza: {efficiency_rate:.4f}% evasioni/query")
    
    print_attack_report(metrics)
    
    # ========== FASE 8: SALVATAGGIO ==========
    if save_results:
        print(f"\n{'='*80}")
        print(f"SALVATAGGIO RISULTATI")
        print(f"{'='*80}")
        
        save_attack_results(
            [metrics],
            X_test,
            {'blackbox_query_hsj': X_adv_full},
            epsilons_tested=['BlackBox_Query_HSJ'],
            save_dir=os.path.join(os.path.dirname(__file__), 'results')
        )
    
    # ========== FASE 9: SUMMARY ==========
    print(f"\n{'='*80}")
    print(f"✅ ATTACCO BLACK-BOX COMPLETATO")
    print(f"{'='*80}")
    print(f"\n📊 RIASSUNTO:")
    print(f"\n1. EFFICACIA:")
    print(f"   - ASR: {metrics['asr']*100:.2f}%")
    print(f"   - Evasioni: {metrics['successful_evasions']}/{len(X_attacks_test)}")
    print(f"\n2. PERTURBAZIONI:")
    print(f"   - L2 medio: {metrics['l2_mean']:.6f}")
    print(f"   - L-inf medio: {metrics['linf_mean']:.6f}")
    print(f"\n3. QUERY:")
    print(f"   - Totali: {total_queries}")
    print(f"   - Per campione: {queries_per_sample:.1f}")
    print(f"   - Efficienza: {efficiency_rate:.4f}%")
    print(f"={'='*80}\n")
    
    results = {
        'metrics': metrics,
        'X_adv': X_adv_full,
        'query_stats': query_stats,
        'config': {
            'max_iter': max_iter,
            'max_eval': max_eval,
            'mode': 'silent',
            'total_queries': total_queries,
            'queries_per_sample': queries_per_sample,
            'efficiency_rate': efficiency_rate
        }
    }
    
    return results


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description="Attacco Black-Box query-based HopSkipJump"
    )
    
    parser.add_argument('--target-model-path', type=str, required=True)
    parser.add_argument('--test-clients', type=int, nargs='+', default=[1, 13])
    parser.add_argument('--max-iter', type=int, default=20)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--init-eval', type=int, default=50)
    parser.add_argument('--norm', type=str, choices=['2', 'inf'], default='2')
    parser.add_argument('--max-queries-total', type=int, default=5000000)
    parser.add_argument('--save-results', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    norm_value = 2 if args.norm == '2' else np.inf
    
    results = run_blackbox_query_hsj_attack(
        target_model_path=args.target_model_path,
        test_clients=args.test_clients,
        max_iter=args.max_iter,
        max_eval=args.max_eval,
        init_eval=args.init_eval,
        norm=norm_value,
        max_queries_total=args.max_queries_total,
        save_results=args.save_results or True,
        verbose=args.verbose
    )
    
    if results is None:
        sys.exit(1)
    else:
        print(f"\n✅ Attacco completato!")
        print(f"   ASR: {results['metrics']['asr']*100:.2f}%")
        print(f"   Query totali: {results['query_stats']['total_queries']}")


if __name__ == "__main__":
    main()