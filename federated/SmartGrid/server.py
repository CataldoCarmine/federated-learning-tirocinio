import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import os

def get_smartgrid_evaluate_fn():
    """
    Crea una funzione di valutazione globale per il server SmartGrid.
    Usa un subset dei dati per valutare il modello aggregato.
    
    Returns:
        Funzione di valutazione che può essere usata dal server
    """
    
    def load_global_test_data():
        """
        Carica un dataset globale di test per la valutazione del server.
        Usa alcuni file client non utilizzati durante il training federato.
        """
        print("Caricamento dataset globale per valutazione server...")
        
        # Directory in cui si trova questo script (es. federated/SmartGrid/server.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Costruzione robusta dei path ai file CSV
        test_clients = [14, 15]
        df_list = []

        for client_id in test_clients:
            file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
                print(f"  - Caricato data{client_id}.csv: {len(df)} campioni")
            except FileNotFoundError:
                print(f"  - File data{client_id}.csv non trovato, saltato")
                continue

        if not df_list:
            print("  - ATTENZIONE: Nessun file di test globale trovato!")
            print("  - Usando il primo client disponibile come fallback...")

            fallback_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", "data1.csv")
            try:
                df_fallback = pd.read_csv(fallback_path)
                df_list = [df_fallback.sample(n=min(200, len(df_fallback)), random_state=42)]
            except FileNotFoundError:
                raise FileNotFoundError("Impossibile caricare dati per valutazione globale")
        
        # Combina i dataframe
        df_global = pd.concat(df_list, ignore_index=True)
        
        # Prepara X e y
        X_global = df_global.drop(columns=["marker"])
        y_global = (df_global["marker"] != "Natural").astype(int)
        
        # === PREPROCESSING IDENTICO AI CLIENT ===
        
        # STEP 1: Gestione valori mancanti con imputazione
        X_global.replace([np.inf, -np.inf], np.nan, inplace=True)
        nan_count = X_global.isnull().sum().sum()
        
        if nan_count > 0:
            print(f"  - Valori NaN trovati: {nan_count}, applicazione imputazione...")
            imputer = SimpleImputer(strategy="median")
            X_global_imputed = imputer.fit_transform(X_global)
            X_global = pd.DataFrame(X_global_imputed, columns=X_global.columns)

        # STEP 2: Normalizzazione
        scaler_global = StandardScaler()
        X_global_scaled = scaler_global.fit_transform(X_global)
        
        # NOTA: Per la valutazione globale manteniamo i dati sbilanciati
        # per rispecchiare la distribuzione reale (nessun SMOTE sul test set)
        
        print(f"  - Dataset globale preparato: {len(X_global_scaled)} campioni")
        print(f"  - Attacchi: {y_global.sum()}, Naturali: {(y_global == 0).sum()}")
        print(f"  - Dataset di test globale mantiene distribuzione reale (sbilanciata)")
        
        return X_global_scaled, y_global
    
    # Carica i dati globali una sola volta
    try:
        X_global, y_global = load_global_test_data()
        input_shape = X_global.shape[1]
    except Exception as e:
        print(f"Errore nel caricamento dati globali: {e}")
        # Fallback: crea dati fittizi per evitare crash
        X_global = np.random.random((100, 128))
        y_global = np.random.randint(0, 2, 100)
        input_shape = 128
        print("Usando dati fittizi per valutazione globale")
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione chiamata ad ogni round.
        
        Args:
            server_round: Numero del round corrente
            parameters: Pesi del modello aggregato
            config: Configurazione
        
        Returns:
            Tuple con (loss, metriche)
        """
        print(f"\n=== VALUTAZIONE GLOBALE ROUND {server_round} ===")
        
        try:
            # Crea il modello per la valutazione (identico ai client)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_shape,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])
            
            # Compila il modello
            model.compile(
                optimizer="adam",
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=["accuracy", "precision", "recall"]
            )
            
            # Imposta i pesi aggregati
            model.set_weights(parameters)
            
            # Valutazione sul dataset globale
            results = model.evaluate(X_global, y_global, verbose=0)
            loss, accuracy, precision, recall = results
            
            # Calcola F1-score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Risultati valutazione globale:")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  - F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
            print(f"  - Campioni utilizzati: {len(X_global)}")
            
            # Predizioni per analisi dettagliata
            predictions_prob = model.predict(X_global, verbose=0)
            predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
            
            # Matrice di confusione
            tn = np.sum((y_global == 0) & (predictions_binary == 0))
            fp = np.sum((y_global == 0) & (predictions_binary == 1))
            fn = np.sum((y_global == 1) & (predictions_binary == 0))
            tp = np.sum((y_global == 1) & (predictions_binary == 1))
            
            print(f"  - Matrice confusione: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            
            print("=" * 60)
            sys.stdout.flush()
            
            return float(loss), {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "global_samples": int(len(X_global))
            }
            
        except Exception as e:
            print(f"Errore durante la valutazione globale: {e}")
            return 1.0, {"accuracy": 0.0, "error": str(e), "global_samples": 0}
    
    return evaluate

def print_client_metrics(fit_results):
    """
    Stampa le metriche dei client dopo ogni round di addestramento.
    
    Args:
        fit_results: Risultati dell'addestramento dai client
    """
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT ROUND ===")
    
    total_samples = 0
    total_weighted_accuracy = 0
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client {i+1}:")
        print(f"  - Campioni training: {client_samples}")
        
        if 'train_accuracy' in client_metrics:
            accuracy = client_metrics['train_accuracy']
            total_weighted_accuracy += accuracy * client_samples
            print(f"  - Train Accuracy: {accuracy:.4f}")
        
        if 'train_loss' in client_metrics:
            print(f"  - Train Loss: {client_metrics['train_loss']:.4f}")
        
        if 'train_precision' in client_metrics:
            print(f"  - Train Precision: {client_metrics['train_precision']:.4f}")
        
        if 'train_recall' in client_metrics:
            print(f"  - Train Recall: {client_metrics['train_recall']:.4f}")
    
    if total_samples > 0:
        avg_weighted_accuracy = total_weighted_accuracy / total_samples
        print(f"\nMedia pesata accuracy: {avg_weighted_accuracy:.4f}")
    
    print("=" * 40)

class SmartGridFedAvg(FedAvg):
    """
    Strategia FedAvg personalizzata per SmartGrid con logging migliorato.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega i risultati dell'addestramento e stampa metriche dettagliate.
        """
        print(f"\n=== AGGREGAZIONE ROUND {server_round} ===")
        print(f"Client partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        
        if failures:
            print("Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")
        
        # Stampa metriche dei client
        print_client_metrics(results)
        
        # Chiama l'aggregazione standard
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is not None:
            print(f"Aggregazione completata per round {server_round}")
        else:
            print(f"ATTENZIONE: Aggregazione fallita per round {server_round}")
        
        return aggregated_result

def main():
    """
    Funzione principale per avviare il server SmartGrid federato.
    """
    print("=== AVVIO SERVER FEDERATO SMARTGRID ===")
    print("Configurazione:")
    print("  - Numero di round: 5")
    print("  - Client minimi per training: 2")
    print("  - Client minimi per valutazione: 2")
    print("  - Client minimi disponibili: 2")
    print("  - Strategia: FedAvg personalizzata")
    print("  - Valutazione: Dataset globale (client 14-15)")
    print("=" * 50)
    
    # Configurazione del server
    config = fl.server.ServerConfig(num_rounds=5)
    
    # Strategia Federated Averaging personalizzata
    strategy = SmartGridFedAvg(
        fraction_fit=1.0,                    # Usa tutti i client disponibili per training
        fraction_evaluate=1.0,               # Usa tutti i client disponibili per valutazione
        min_fit_clients=2,                   # Numero minimo di client per iniziare training
        min_evaluate_clients=2,              # Numero minimo di client per valutazione
        min_available_clients=2,             # Numero minimo di client connessi
        evaluate_fn=get_smartgrid_evaluate_fn()  # Valutazione globale
    )
    
    print("Server in attesa di client...")
    print("Per connettere i client, esegui in terminali separati:")
    print("  python client.py 1")
    print("  python client.py 2")
    print("  python client.py 3")
    print("  ...")
    print("  python client.py 13")
    print("\nNOTA: Usa client ID 1-13 per training federato")
    print("      Client 14-15 sono riservati per valutazione globale")
    print("\nIl training inizierà quando almeno 2 client saranno connessi.")
    print("=" * 50)
    sys.stdout.flush()
    
    try:
        # Avvia il server
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )
    except Exception as e:
        print(f"Errore durante l'avvio del server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()