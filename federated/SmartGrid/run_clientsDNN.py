# run_clients.py
import multiprocessing
import os

def run_client(client_id):
    # Avvia il client con il suo ID
    os.system(f"python3 clientDNN.py {client_id}")

if __name__ == "__main__":
    num_clients = 12
    processes = []

    num_clients2 = 15

    for client_id in range(2, num_clients + 1):  # da 2 a 12 inclusi
        p = multiprocessing.Process(target=run_client, args=(client_id,))
        p.start()
        processes.append(p)

    for client_id in range(14, num_clients2 + 1):  # da 14 a 15 inclusi
         p = multiprocessing.Process(target=run_client, args=(client_id,))
         p.start()
         processes.append(p)

    for p in processes:
        p.join()