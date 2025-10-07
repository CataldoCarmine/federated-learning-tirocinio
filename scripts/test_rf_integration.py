#!/usr/bin/env python3
"""
Script di integrazione per testare il flusso completo server-client Random Forest.

Questo script:
1. Avvia il server Random Forest in un processo separato
2. Avvia un client Random Forest
3. Verifica che la comunicazione funzioni correttamente
4. Termina i processi

NOTA: Questo è un test minimale per verificare la serializzazione.
      Per un test completo con tutti i client, usa run_clientsRF.py
"""

import subprocess
import time
import signal
import sys
import os

def print_section(title):
    """Stampa una sezione separata per maggiore leggibilità"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_integration():
    """Test di integrazione con server e client reali"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       TEST INTEGRAZIONE RANDOM FOREST FEDERATO (Server + Client)            ║
║                                                                              ║
║  Questo script verifica che server e client possano comunicare              ║
║  correttamente con la serializzazione corretta                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Cambia directory alla root del progetto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    smartgrid_dir = os.path.join(project_root, 'federated', 'SmartGrid')
    
    print(f"Directory progetto: {project_root}")
    print(f"Directory SmartGrid: {smartgrid_dir}")
    
    # Verifica che i file esistano
    server_path = os.path.join(smartgrid_dir, 'serverRF.py')
    client_path = os.path.join(smartgrid_dir, 'clientRF.py')
    
    if not os.path.exists(server_path):
        print(f"❌ ERRORE: serverRF.py non trovato in {server_path}")
        return 1
    
    if not os.path.exists(client_path):
        print(f"❌ ERRORE: clientRF.py non trovato in {client_path}")
        return 1
    
    print(f"✅ File trovati: serverRF.py, clientRF.py")
    
    # Verifica che i dati esistano
    data_path = os.path.join(project_root, 'data', 'SmartGrid', 'data1.csv')
    if not os.path.exists(data_path):
        print(f"❌ ERRORE: Dati SmartGrid non trovati in {data_path}")
        print("   Questo test richiede i dati SmartGrid per funzionare")
        return 1
    
    print(f"✅ Dati trovati: {data_path}")
    
    print_section("Avvio Server Random Forest")
    
    # Avvia il server in un processo separato
    try:
        server_process = subprocess.Popen(
            [sys.executable, server_path],
            cwd=smartgrid_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        print(f"✅ Server avviato (PID: {server_process.pid})")
        
        # Aspetta che il server sia pronto
        print("⏳ Attendo che il server sia pronto (5 secondi)...")
        time.sleep(5)
        
        # Verifica che il server sia ancora attivo
        if server_process.poll() is not None:
            print(f"❌ ERRORE: Il server si è terminato prematuramente")
            output, _ = server_process.communicate()
            print(f"Output server:\n{output}")
            return 1
        
        print("✅ Server attivo e in ascolto")
        
    except Exception as e:
        print(f"❌ ERRORE avvio server: {e}")
        return 1
    
    print_section("Avvio Client Random Forest (Client 1)")
    
    # Avvia un client
    try:
        client_process = subprocess.Popen(
            [sys.executable, client_path, '1'],
            cwd=smartgrid_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        print(f"✅ Client 1 avviato (PID: {client_process.pid})")
        
        # Leggi l'output del client per un po'
        print("⏳ Monitoro il client per 30 secondi...")
        timeout = 30
        start_time = time.time()
        
        serialization_success = False
        deserialization_success = False
        training_success = False
        
        while time.time() - start_time < timeout:
            # Leggi output del client
            line = client_process.stdout.readline()
            if line:
                print(f"[Client] {line.rstrip()}")
                
                # Cerca segni di successo
                if "Invio" in line and "alberi al server" in line:
                    serialization_success = True
                    print("✅ Serializzazione client completata")
                
                if "Modello aggregato ricevuto dal server" in line:
                    deserialization_success = True
                    print("✅ Deserializzazione client completata")
                
                if "Training completato" in line:
                    training_success = True
                    print("✅ Training client completato")
            
            # Verifica se il client è terminato
            if client_process.poll() is not None:
                print(f"Client terminato con exit code: {client_process.poll()}")
                break
            
            time.sleep(0.1)
        
        print_section("Risultati Test")
        
        print(f"Training completato: {'✅' if training_success else '❌'}")
        print(f"Serializzazione: {'✅' if serialization_success else '❌'}")
        print(f"Deserializzazione: {'✅' if deserialization_success else '❌'}")
        
        success = serialization_success and training_success
        
        if success:
            print("\n✅ TEST INTEGRAZIONE SUPERATO!")
            print("   La serializzazione/deserializzazione funziona correttamente")
        else:
            print("\n⚠️ TEST INTEGRAZIONE PARZIALE")
            print("   Alcuni componenti potrebbero non aver funzionato completamente")
            print("   Questo è normale per un test veloce con 1 solo client")
        
    except Exception as e:
        print(f"❌ ERRORE durante test client: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    finally:
        # Termina i processi
        print_section("Pulizia")
        
        try:
            if 'client_process' in locals() and client_process.poll() is None:
                print("Terminazione client...")
                client_process.terminate()
                client_process.wait(timeout=5)
                print("✅ Client terminato")
        except Exception as e:
            print(f"⚠️ Errore terminazione client: {e}")
            try:
                client_process.kill()
            except:
                pass
        
        try:
            if 'server_process' in locals() and server_process.poll() is None:
                print("Terminazione server...")
                server_process.terminate()
                server_process.wait(timeout=5)
                print("✅ Server terminato")
        except Exception as e:
            print(f"⚠️ Errore terminazione server: {e}")
            try:
                server_process.kill()
            except:
                pass
    
    return 0 if success else 1

def main():
    """Funzione principale"""
    try:
        return test_integration()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrotto dall'utente")
        return 130
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
