# run_clients.py
import multiprocessing
import os

def run_client(client_id):
    # Avvia il client con il suo ID
    os.system(f"python3 client.py {client_id}")

if __name__ == "__main__":
    num_clients = 13
    processes = []

    for client_id in range(1, num_clients + 1):  # da 1 a 13 inclusi
        p = multiprocessing.Process(target=run_client, args=(client_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()