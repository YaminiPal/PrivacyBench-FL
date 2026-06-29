import torch
import numpy as np
from utils.logger import logger
from utils.helpers import has_nan_gradients


class NullPrivacyAccountant:
    def step(self, **kwargs):
        pass

    def get_epsilon(self):
        return 0.0


class FLTrainer:
    def __init__(
        self,
        server,
        clients,
        train_loaders,
        val_loaders,
        metrics_engine,
        early_stopper,
        privacy_accountant,
        client_selector=None,
        clipper=None,
        quantizer=None,
        comm_channel=None,
        dp_config=None,
        forgetting_engine=None,
        temporal_loaders=None,
        device="cpu",
    ):
        self.server = server
        self.clients = clients
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loader = self._build_pooled_val_loader(val_loaders)
        self.metrics = metrics_engine
        self.stopper = early_stopper
        self.accountant = privacy_accountant
        self.selector = client_selector
        self.clipper = clipper
        self.quantizer = quantizer
        self.comm_channel = comm_channel
        self.dp_config = dp_config or {}
        self.forgetting_engine = forgetting_engine
        self.temporal_loaders = temporal_loaders or {}
        self.device = torch.device(device)
        self.total_traffic_mb = 0.0
        self.client_participation = {cid: 0 for cid in train_loaders}
        self.clipped_fraction_history = []

    def _build_pooled_val_loader(self, val_loaders):
        """Pool all client validation sets for global evaluation."""
        xs, ys = [], []
        for loader in val_loaders.values():
            if loader is None:
                continue
            for X, y in loader:
                xs.append(X)
                ys.append(y)
        if not xs:
            return None
        from torch.utils.data import TensorDataset, DataLoader
        X_all = torch.cat(xs, dim=0)
        y_all = torch.cat(ys, dim=0)
        return DataLoader(TensorDataset(X_all, y_all), batch_size=32, shuffle=False)

    def train(
        self,
        rounds,
        local_epochs,
        lr,
        noise_multiplier,
        lambda_reg=1e-4,
        drift_config=None,
        drift_engine=None,
        batch_size=32,
    ):
        history = {
            "round": [], "accuracy": [], "loss": [], "epsilon": [],
            "traffic": [], "total_traffic_mb": [], "f1_score": [],
            "client_participation": [], "forgetting_T1": [], "forgetting_T3": [],
        }
        logger.info(f"Starting FL training for {rounds} rounds")

        for r in range(rounds):
            current_round = r + 1

            if (
                drift_config and drift_config.get("enabled")
                and current_round == drift_config.get("trigger_round", 4)
            ):
                logger.warning(
                    f"Drift triggered: {drift_config['mode']} at round {current_round}"
                )
                for cid in self.train_loaders:
                    if self.train_loaders[cid] is not None:
                        self.train_loaders[cid] = drift_engine.apply_drift(
                            self.train_loaders[cid]
                        )

            global_params = self.server.get_parameters()
            self._track_payload(global_params)

            all_client_ids = list(self.train_loaders.keys())
            if self.selector:
                active_clients = self.selector.select(
                    self.clients, self.train_loaders
                )
            else:
                active_clients = all_client_ids

            updates = []
            total_samples = 0
            clipped_count = 0
            client_losses = []

            for cid in active_clients:
                client = self.clients[cid]
                loader = self.train_loaders[cid]
                if loader is None or len(loader.dataset) == 0:
                    continue

                client.set_parameters(global_params)

                dp_engine = None
                if self.dp_config.get("enabled"):
                    try:
                        from privacy.dp import DPEngine
                        dp_engine = DPEngine(
                            noise_multiplier=noise_multiplier,
                            max_grad_norm=self.dp_config.get("clip_norm", 1.0),
                            delta=self.dp_config.get("delta", 1e-5),
                        )
                    except Exception as e:
                        logger.warning(f"DP engine unavailable for {cid}: {e}")

                try:
                    delta = client.train(
                        train_loader=loader,
                        epochs=local_epochs,
                        lr=lr,
                        lambda_reg=lambda_reg,
                        dp_engine=dp_engine,
                        clipper=self.clipper,
                    )
                except Exception as e:
                    logger.warning(f"Training failed for client {cid}: {e}")
                    continue

                if has_nan_gradients(client.model):
                    logger.warning(f"NaN gradients for client {cid}, skipping")
                    continue

                if self.selector:
                    noise_level = noise_multiplier if self.dp_config.get("enabled") else 0.0
                    self.selector.update_client_metrics(
                        cid, client.last_update_norm, noise_level
                    )

                if self.clipper and client.last_update_norm > self.clipper.clip_bound:
                    clipped_count += 1

                if self.comm_channel and self.quantizer:
                    packed = client.serialize_update(comm_channel=self.comm_channel)
                    delta = self.comm_channel.unpack_payload(packed)

                updates.append((len(loader.dataset), delta))
                total_samples += len(loader.dataset)
                self.client_participation[cid] = self.client_participation.get(cid, 0) + 1
                self._track_payload(delta)

            if self.clipper and len(active_clients) > 0:
                frac_clipped = clipped_count / len(active_clients)
                self.clipper.update_clipping_bound(frac_clipped)
                self.clipped_fraction_history.append(frac_clipped)

            if len(updates) == 0:
                logger.error("No valid updates in round")
                break

            aggregated = self._fedavg(updates)
            new_params = {
                k: global_params[k] + aggregated[k].to(self.device)
                for k in global_params
            }
            self.server.set_parameters(new_params)

            loss, acc, logits, targets = self._evaluate(return_raw=True)
            client_losses.append(loss)

            sample_rate = 0.0
            if total_samples > 0:
                sample_rate = min(batch_size / total_samples, 1.0)
            if self.dp_config.get("enabled"):
                self.accountant.step(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                    steps=local_epochs * len(active_clients),
                )
            eps = self.accountant.get_epsilon()

            round_stats = {"round": current_round, "loss": loss, "accuracy": acc}
            if self.metrics:
                round_stats = self.metrics.compute_global_metrics(
                    current_round, [logits], [targets], loss, client_losses
                )
                self.metrics.print_summary_table(round_stats)

            t1_acc, t3_acc = self._evaluate_temporal_chunks()
            if self.forgetting_engine and (t1_acc is not None or t3_acc is not None):
                scores = [t1_acc or 0.0, t3_acc or 0.0]
                self.forgetting_engine.record_eval_snapshot(scores)

            logger.info(
                f"Round {current_round} | Acc {acc:.4f} | Loss {loss:.4f} | "
                f"ε {eps:.4f} | Clients {len(active_clients)}/{len(all_client_ids)}"
            )

            history["round"].append(current_round)
            history["accuracy"].append(acc)
            history["loss"].append(loss)
            history["epsilon"].append(eps)
            history["traffic"].append(self.total_traffic_mb)
            history["total_traffic_mb"].append(self.total_traffic_mb)
            history["f1_score"].append(round_stats.get("f1_score", 0.0))
            history["client_participation"].append(len(active_clients))
            history["forgetting_T1"].append(t1_acc)
            history["forgetting_T3"].append(t3_acc)

            if self.stopper.should_stop(
                current_loss=loss, current_accuracy=acc, current_epsilon=eps
            ):
                logger.warning(f"Early stop: {self.stopper.stop_reason}")
                break

        if self.forgetting_engine:
            self.forgetting_engine.print_text_summary()
            history["forgetting_report"] = (
                self.forgetting_engine.generate_stability_report()
            )

        history["client_participation_total"] = self.client_participation
        return history

    def _fedavg(self, updates):
        total = sum(w for w, _ in updates)
        avg = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in updates[0][1].items()
        }
        for w, delta in updates:
            for k in avg:
                avg[k] += delta[k].to(torch.float32) * (w / total)
        return avg

    def _evaluate(self, return_raw=False):
        if self.test_loader is None:
            return 0.0, 0.0, None, None

        model = self.server.model.to(self.device)
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        total, correct, loss_sum = 0, 0, 0.0
        all_logits, all_targets = [], []

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                out = model(X)
                loss = loss_fn(out, y)
                loss_sum += loss.item() * len(X)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(X)
                if return_raw:
                    all_logits.append(out)
                    all_targets.append(y)

        acc = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / total if total > 0 else 0.0

        if return_raw and all_logits:
            return avg_loss, acc, torch.cat(all_logits), torch.cat(all_targets)
        return avg_loss, acc

    def _evaluate_temporal_chunks(self):
        """Phase 6: evaluate on T1 (old) and T3 (latest) data per client."""
        if not self.temporal_loaders:
            return None, None

        model = self.server.model.to(self.device)
        model.eval()
        t1_correct, t1_total = 0, 0
        t3_correct, t3_total = 0, 0

        with torch.no_grad():
            for cid, periods in self.temporal_loaders.items():
                if "T1" in periods and periods["T1"] is not None:
                    for X, y in periods["T1"]:
                        X, y = X.to(self.device), y.to(self.device)
                        pred = model(X).argmax(dim=1)
                        t1_correct += (pred == y).sum().item()
                        t1_total += len(y)
                if "T3" in periods and periods["T3"] is not None:
                    for X, y in periods["T3"]:
                        X, y = X.to(self.device), y.to(self.device)
                        pred = model(X).argmax(dim=1)
                        t3_correct += (pred == y).sum().item()
                        t3_total += len(y)

        t1_acc = t1_correct / t1_total if t1_total > 0 else None
        t3_acc = t3_correct / t3_total if t3_total > 0 else None
        return t1_acc, t3_acc

    def _track_payload(self, state_dict):
        bytes_used = sum(v.numel() * 4 for v in state_dict.values())
        self.total_traffic_mb += bytes_used / (1024 * 1024)
