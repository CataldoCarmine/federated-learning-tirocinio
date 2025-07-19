import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,           # percentuale di client usati per training (1.0 = 100%)
    fraction_evaluate=1.0,      # percentuale di client usati per valutazione
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy)
