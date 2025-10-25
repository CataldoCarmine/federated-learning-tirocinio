"""
Script di logging automatico per diagnosticare problemi nella versione Enhanced RF.
Estrae informazioni critiche da un round di esecuzione per analisi approfondita.

Uso:
1. Avvia questo script PRIMA di server e client
2. Avvia server: python serverRFtmp.py
3. Avvia client: python clientRFtmp.py 1 (e altri client)
4. Lo script salverà automaticamente i log in results/debug_enhanced/
"""

import os
import sys
import time
import subprocess
import signal
from datetime import datetime
from pathlib import Path

# Configurazione
DEBUG_DIR = Path("results/debug_enhanced")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SERVER_LOG = DEBUG_DIR / f"server_enhanced_{TIMESTAMP}.log"
CLIENT_LOGS = {}  # {client_id: Path}

# Numero di client da avviare
NUM_CLIENTS = 3  # Usa solo 3 client per debug veloce

# Numero di round da eseguire per debug
DEBUG_ROUNDS = 2

def setup_client_log(client_id):
    """Crea file di log per un client specifico"""
    log_path = DEBUG_DIR / f"client{client_id}_enhanced_{TIMESTAMP}.log"
    CLIENT_LOGS[client_id] = log_path
    return log_path

def extract_critical_sections(log_file, output_file):
    """
    Estrae le sezioni critiche da un file di log completo.
    Cerca pattern specifici che indicano le aree problematiche.
    """
    if not log_file.exists():
        print(f"⚠️  Log file non trovato: {log_file}")
        return
    
    critical_patterns = [
        "FEATURE ENGINEERING",
        "ESTRAZIONE ALBERI",
        "SERIALIZZAZIONE ALBERI",
        "DESERIALIZZAZIONE",
        "SELEZIONE ALBERI",
        "CREAZIONE RANDOM FOREST GLOBALE",
        "DIVERSITY",
        "ACCURACY REALI",
        "VERIFICATION",
        "DEBUG",
        "⚠️",
        "❌",
        "✅"
    ]
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    critical_lines = []
    for i, line in enumerate(lines):
        # Includi linea se contiene pattern critico
        if any(pattern in line for pattern in critical_patterns):
            # Includi anche 2 linee prima e 2 dopo per contesto
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            critical_lines.extend(lines[start:end])
            critical_lines.append("\n")  # Separatore
    
    # Scrivi output
    with open(output_file, 'w') as f:
        f.write(f"=== SEZIONI CRITICHE ESTRATTE DA {log_file.name} ===\n")
        f.write(f"Timestamp: {TIMESTAMP}\n")
        f.write(f"Totale linee originali: {len(lines)}\n")
        f.write(f"Linee critiche estratte: {len(critical_lines)}\n")
        f.write("=" * 80 + "\n\n")
        f.writelines(critical_lines)
    
    print(f"✅ Estratte {len(critical_lines)} linee critiche da {log_file.name}")
    print(f"   Salvato in: {output_file}")

def run_server():
    """Avvia il server con logging"""
    print(f"🌳 Avvio server Enhanced RF...")
    print(f"   Log: {SERVER_LOG}")
    
    with open(SERVER_LOG, 'w') as log:
        # Modifica il file serverRFtmp.py per limitare i round
        server_cmd = [sys.executable, "serverRFtmp.py"]
        process = subprocess.Popen(
            server_cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
    
    return process

def run_client(client_id):
    """Avvia un client con logging"""
    log_path = setup_client_log(client_id)
    print(f"🔹 Avvio client {client_id}...")
    print(f"   Log: {log_path}")
    
    with open(log_path, 'w') as log:
        client_cmd = [sys.executable, "clientRFtmp.py", str(client_id)]
        process = subprocess.Popen(
            client_cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
    
    return process

def generate_analysis_report():
    """
    Genera un report di analisi automatico dai log estratti.
    """
    report_path = DEBUG_DIR / f"analysis_report_{TIMESTAMP}.txt"
    
    with open(report_path, 'w') as report:
        report.write("=" * 80 + "\n")
        report.write("REPORT ANALISI AUTOMATICA - ENHANCED RF DEBUG\n")
        report.write("=" * 80 + "\n\n")
        
        report.write(f"Timestamp: {TIMESTAMP}\n")
        report.write(f"Numero client: {NUM_CLIENTS}\n")
        report.write(f"Round eseguiti: {DEBUG_ROUNDS}\n\n")
        
        # Sezione 1: Feature Engineering Check
        report.write("=" * 80 + "\n")
        report.write("1. FEATURE ENGINEERING CHECK\n")
        report.write("=" * 80 + "\n\n")
        
        for client_id, log_path in CLIENT_LOGS.items():
            report.write(f"--- Client {client_id} ---\n")
            if log_path.exists():
                with open(log_path, 'r') as f:
                    content = f.read()
                    
                    # Cerca informazioni feature engineering
                    if "Features originali:" in content:
                        for line in content.split('\n'):
                            if "Features originali:" in line or "Features aggiunte:" in line or "Features totali:" in line:
                                report.write(f"  {line.strip()}\n")
                    
                    # Cerca NaN/Inf dopo FE
                    if "NaN dopo FE:" in content:
                        for line in content.split('\n'):
                            if "NaN dopo FE:" in line or "Inf dopo FE:" in line:
                                report.write(f"  {line.strip()}\n")
            report.write("\n")
        
        # Sezione 2: Diversity Scores Check
        report.write("=" * 80 + "\n")
        report.write("2. DIVERSITY SCORES CHECK\n")
        report.write("=" * 80 + "\n\n")
        
        for client_id, log_path in CLIENT_LOGS.items():
            report.write(f"--- Client {client_id} ---\n")
            if log_path.exists():
                with open(log_path, 'r') as f:
                    content = f.read()
                    
                    # Cerca diversity scores
                    if "DIVERSITY CHECK:" in content:
                        found = False
                        for line in content.split('\n'):
                            if "DIVERSITY CHECK:" in line or ("Media:" in line and found) or ("Min:" in line and found) or ("Max:" in line and found) or ("Zero count:" in line and found):
                                report.write(f"  {line.strip()}\n")
                                found = True
                            elif found and line.strip() == "":
                                break
            report.write("\n")
        
        # Sezione 3: Server Selection Check
        report.write("=" * 80 + "\n")
        report.write("3. SERVER SELECTION CHECK\n")
        report.write("=" * 80 + "\n\n")
        
        if SERVER_LOG.exists():
            with open(SERVER_LOG, 'r') as f:
                content = f.read()
                
                # Cerca SELECTION CHECK
                if "SELECTION CHECK:" in content:
                    in_section = False
                    for line in content.split('\n'):
                        if "SELECTION CHECK:" in line:
                            in_section = True
                        if in_section:
                            report.write(f"  {line.strip()}\n")
                            if "Tree" in line and "combined=" in line:
                                continue
                            elif line.strip() == "":
                                break
        
        # Sezione 4: Verification Check
        report.write("\n" + "=" * 80 + "\n")
        report.write("4. VERIFICATION CHECK (Server)\n")
        report.write("=" * 80 + "\n\n")
        
        if SERVER_LOG.exists():
            with open(SERVER_LOG, 'r') as f:
                content = f.read()
                
                # Cerca VERIFICATION
                for line in content.split('\n'):
                    if "VERIFICATION:" in line or "Alberi con diversity REALE:" in line:
                        report.write(f"  {line.strip()}\n")
        
        report.write("\n" + "=" * 80 + "\n")
        report.write("FINE REPORT\n")
        report.write("=" * 80 + "\n")
    
    print(f"\n✅ Report di analisi generato: {report_path}")
    return report_path

def main():
    """
    Funzione principale per orchestrare il debug logging.
    """
    print("\n" + "=" * 80)
    print("🔍 DEBUG ENHANCED RF - LOGGING AUTOMATICO")
    print("=" * 80)
    print(f"Directory output: {DEBUG_DIR}")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Client da avviare: {NUM_CLIENTS}")
    print(f"Round debug: {DEBUG_ROUNDS}")
    print("=" * 80 + "\n")
    
    # Avvia server
    server_process = run_server()
    time.sleep(5)  # Attendi che il server sia pronto
    
    # Avvia client
    client_processes = []
    for client_id in range(1, NUM_CLIENTS + 1):
        process = run_client(client_id)
        client_processes.append(process)
        time.sleep(2)  # Delay tra client
    
    print(f"\n✅ Server e {NUM_CLIENTS} client avviati con logging")
    print(f"⏳ Attendo completamento di {DEBUG_ROUNDS} round...")
    print("   (Premi Ctrl+C per terminare anticipatamente)\n")
    
    try:
        # Attendi completamento (stima: ~60 secondi per round)
        wait_time = DEBUG_ROUNDS * 60 + 30
        for i in range(wait_time):
            time.sleep(1)
            if i % 10 == 0:
                print(f"   Tempo trascorso: {i}s / {wait_time}s")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interruzione manuale rilevata")
    
    finally:
        # Termina processi
        print("\n🛑 Terminazione processi...")
        server_process.terminate()
        for process in client_processes:
            process.terminate()
        
        time.sleep(2)
        
        # Forza terminazione se ancora attivi
        try:
            server_process.kill()
            for process in client_processes:
                process.kill()
        except:
            pass
        
        print("✅ Processi terminati")
    
    # Estrai sezioni critiche
    print("\n🔍 Estrazione sezioni critiche dai log...")
    
    # Estrai da server
    server_critical = DEBUG_DIR / f"server_critical_{TIMESTAMP}.txt"
    extract_critical_sections(SERVER_LOG, server_critical)
    
    # Estrai da client
    for client_id, log_path in CLIENT_LOGS.items():
        client_critical = DEBUG_DIR / f"client{client_id}_critical_{TIMESTAMP}.txt"
        extract_critical_sections(log_path, client_critical)
    
    # Genera report di analisi
    print("\n📊 Generazione report di analisi...")
    report_path = generate_analysis_report()
    
    # Riepilogo finale
    print("\n" + "=" * 80)
    print("✅ DEBUG LOGGING COMPLETATO")
    print("=" * 80)
    print(f"\nFile generati in: {DEBUG_DIR}/")
    print(f"\n📄 LOG COMPLETI:")
    print(f"   - Server: {SERVER_LOG.name}")
    for client_id in CLIENT_LOGS:
        print(f"   - Client {client_id}: {CLIENT_LOGS[client_id].name}")
    
    print(f"\n🔍 SEZIONI CRITICHE ESTRATTE:")
    print(f"   - Server: {server_critical.name}")
    for client_id in CLIENT_LOGS:
        client_critical_name = f"client{client_id}_critical_{TIMESTAMP}.txt"
        print(f"   - Client {client_id}: {client_critical_name}")
    
    print(f"\n📊 REPORT ANALISI:")
    print(f"   - {report_path.name}")
    
    print("\n💡 PROSSIMI PASSI:")
    print("   1. Leggi il report di analisi per identificare il problema")
    print("   2. Consulta le sezioni critiche per dettagli")
    print("   3. Se necessario, consulta i log completi")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    # Verifica di essere nella directory corretta
    if not Path("serverRFtmp.py").exists():
        print("❌ ERRORE: Esegui questo script dalla directory federated/SmartGrid/")
        sys.exit(1)
    
    main()