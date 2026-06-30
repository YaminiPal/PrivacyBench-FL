import time
import copy
import numpy as np


class FLBenchmark:
    def __init__(self, trainer, server, clients):
        """
        Benchmarking layer for federated learning experiments.

        trainer: FLTrainer instance
        server: FLServer instance
        clients: dict of FLClient objects
        """

        self.trainer = trainer
        self.server = server
        self.clients = clients

        self.results = {}

    # ============================
    # RUN FULL FL EXPERIMENT
    # ============================
    def run_fl(self, rounds=10, local_epochs=1, lr=0.01, name="fl"):
        start = time.time()

        history = self.trainer.train(
            rounds=rounds,
            local_epochs=local_epochs,
            lr=lr
        )

        end = time.time()

        self.results[name] = {
            "history": history,
            "time": end - start
        }

        return history

    # ============================
    # CENTRALIZED TRAINING BASELINE
    # ============================
    def run_centralized(self, global_loader, model, epochs=5, lr=0.01, name="centralized"):
        import torch

        model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        model.train()

        start = time.time()

        for _ in range(epochs):
            for X, y in global_loader:
                optimizer.zero_grad()
                outputs = model(X)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()

        end = time.time()

        self.results[name] = {
            "model": model,
            "time": end - start
        }

        return model

    # ============================
    # SIMPLE METRIC EXTRACTION
    # ============================
    def compare_accuracy(self):
        summary = {}

        for name, result in self.results.items():
            if "history" in result:
                acc = result["history"]["accuracy"][-1]
                summary[name] = acc

        return summary

    # ============================
    # COMMUNICATION COST ESTIMATION
    # ============================
    def communication_cost(self, model):
        """
        Estimates communication cost in MB per round.
        """

        total_params = 0

        for param in model.state_dict().values():
            total_params += param.numel() * param.element_size()

        # convert bytes → MB
        return total_params / (1024 * 1024)

    # ============================
    # PRINT FINAL REPORT
    # ============================
    def report(self):
        print("\n📊 FINAL BENCHMARK REPORT")
        print("=" * 40)

        for name, result in self.results.items():
            print(f"\n🔹 {name.upper()}")

            if "history" in result:
                acc = result["history"]["accuracy"][-1]
                loss = result["history"]["loss"][-1]

                print(f"Accuracy: {acc:.4f}")
                print(f"Loss: {loss:.4f}")
                print(f"Time: {result['time']:.2f}s")

            else:
                print(f"Time: {result['time']:.2f}s")

        print("\n📡 Communication Cost (approx per round):")
        sample_model = self.server.model
        print(f"{self.communication_cost(sample_model):.2f} MB")