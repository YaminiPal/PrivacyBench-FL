import torch
from torch.utils.data import DataLoader, TensorDataset

from continual_learning.drift_detection import FeatureDriftDetector
from federated.similarity import SimilarityAwareAggregator
from privacy.quantization import QuantizationEngine
from utils.helpers import has_nan_gradients
from utils.logger import logger


class FLTrainer:
    """Federated trainer with personalization, adaptive transport, and CL support."""

    def __init__(self, server, clients, train_loaders, val_loaders, metrics_engine,
                 early_stopper, privacy_accountant, client_selector=None, clipper=None,
                 quantizer=None, comm_channel=None, dp_config=None,
                 forgetting_engine=None, temporal_loaders=None,
                 temporal_train_loaders=None, aggregation_config=None,
                 personalization_config=None, drift_detection_config=None,
                 quantization_config=None, device="cpu"):
        self.server, self.clients = server, clients
        self.train_loaders, self.val_loaders = train_loaders, val_loaders
        self.test_loader = self._build_pooled_loader(val_loaders)
        self.metrics, self.stopper, self.accountant = metrics_engine, early_stopper, privacy_accountant
        self.selector, self.clipper = client_selector, clipper
        self.quantizer, self.comm_channel = quantizer, comm_channel
        self.dp_config = dp_config or {}
        self.forgetting_engine = forgetting_engine
        self.temporal_loaders = temporal_loaders or {}
        self.temporal_train_loaders = temporal_train_loaders or {}
        self.aggregation_config = aggregation_config or {}
        self.quantization_config = quantization_config or {}
        self.personalization_config = personalization_config or {}
        self.drift_detection_config = drift_detection_config or {}
        self.aggregator = SimilarityAwareAggregator(self.aggregation_config.get("similarity_floor", 0.05))
        self.drift_detector = FeatureDriftDetector(self.drift_detection_config.get("z_threshold", 2.5)) if self.drift_detection_config.get("enabled") else None
        self.device = torch.device(device)
        self.total_traffic_mb = 0.0
        self.client_participation = {cid: 0 for cid in train_loaders}

    @staticmethod
    def _build_pooled_loader(loaders):
        batches = [(X, y) for loader in loaders.values() if loader is not None for X, y in loader]
        if not batches:
            return None
        X, y = zip(*batches)
        return DataLoader(TensorDataset(torch.cat(X), torch.cat(y)), batch_size=32, shuffle=False)

    def _adaptive_bits(self, round_idx):
        if not self.quantizer:
            return None
        cfg = self.aggregation_config.get("quantization", {})
        # quantization settings are placed on the channel by main; read its schedule if present.
        cfg = getattr(self, "quantization_config", cfg)
        if not cfg.get("adaptive", False):
            return cfg.get("bits", self.quantizer.bits)
        return cfg.get("early_bits", 16) if round_idx <= cfg.get("early_rounds", 2) else cfg.get("late_bits", 4)

    def _set_round_quantization(self, round_idx):
        bits = self._adaptive_bits(round_idx)
        if bits is not None:
            self.quantizer.set_bits(bits)
        return bits

    def _activate_temporal_task(self, round_idx, total_rounds):
        """Train sequentially on T1, T2, then T3 when temporal data is provided."""
        if not self.temporal_train_loaders:
            return None
        tasks = ["T1", "T2", "T3"]
        task = tasks[min((round_idx - 1) * len(tasks) // total_rounds, len(tasks) - 1)]
        for cid, periods in self.temporal_train_loaders.items():
            if periods.get(task) is not None:
                self.train_loaders[cid] = periods[task]
        return task

    def train(self, rounds, local_epochs, lr, noise_multiplier, lambda_reg=1e-4,
              drift_config=None, drift_engine=None, batch_size=32):
        history = {key: [] for key in ("round", "accuracy", "loss", "epsilon", "traffic",
                   "total_traffic_mb", "f1_score", "client_participation", "forgetting_T1",
                   "forgetting_T3", "quantization_bits", "mean_update_similarity", "active_task")}

        for round_idx in range(1, rounds + 1):
            task = self._activate_temporal_task(round_idx, rounds)
            bits = self._set_round_quantization(round_idx)
            global_params = self.server.get_parameters()
            active = self.selector.select(self.clients, self.train_loaders) if self.selector else list(self.train_loaders)
            updates, drift_events = [], {}

            for cid in active:
                loader = self.train_loaders.get(cid)
                if loader is None or len(loader.dataset) == 0 or not self.accountant.can_participate(cid):
                    continue
                client_lr, client_epochs = lr, local_epochs
                if self.drift_detector:
                    event = self.drift_detector.observe(cid, loader)
                    drift_events[cid] = event
                    if event["detected"]:
                        client_lr *= self.drift_detection_config.get("lr_multiplier", 0.5)
                        client_epochs += self.drift_detection_config.get("extra_local_epochs", 1)
                        logger.warning(f"Detected drift on {cid} (score={event['score']:.2f}); adapting local training")

                client = self.clients[cid]
                client.set_parameters(global_params)
                dp_engine = None
                if self.dp_config.get("enabled"):
                    from privacy.dp import DPEngine
                    dp_engine = DPEngine(noise_multiplier=noise_multiplier,
                                         max_grad_norm=self.dp_config.get("clip_norm", 1.0),
                                         delta=self.dp_config.get("delta", 1e-5))
                try:
                    delta = client.train(loader, epochs=client_epochs, lr=client_lr,
                                         lambda_reg=lambda_reg, dp_engine=dp_engine, clipper=self.clipper)
                except Exception as exc:
                    logger.warning(f"Training failed for {cid}: {exc}")
                    continue
                if has_nan_gradients(client.model):
                    logger.warning(f"Invalid gradients for {cid}; update discarded")
                    continue
                if self.comm_channel:
                    delta = self.comm_channel.unpack_payload(client.serialize_update(comm_channel=self.comm_channel))
                updates.append((cid, len(loader.dataset), delta))
                self.client_participation[cid] += 1
                if self.dp_config.get("enabled"):
                    self.accountant.step_for_client(cid, noise_multiplier,
                                                   min(batch_size / len(loader.dataset), 1.0),
                                                   steps=client_epochs * len(loader))

            if not updates:
                logger.warning("No eligible client updates remain; ending training")
                break
            if self.clipper:
                self.clipper.update_clipping_bound(sum(self.clients[cid].last_update_norm > self.clipper.clip_bound for cid, _, _ in updates) / len(updates))
            if self.aggregation_config.get("strategy", "fedavg") == "similarity_aware":
                aggregate, similarities = self.aggregator.aggregate(updates)
            else:
                total = sum(n for _, n, _ in updates)
                aggregate = {k: sum(delta[k].float() * (n / total) for _, n, delta in updates) for k in updates[0][2]}
                similarities = {}
            # Personalized classifier heads are intentionally absent from aggregate;
            # retain the server reference head while synchronizing the shared encoder.
            self.server.set_parameters({k: global_params[k] + aggregate[k].to(self.device) for k in aggregate})

            loss, acc, logits, targets = self._evaluate(return_raw=True)
            stats = self.metrics.compute_global_metrics(round_idx, [logits], [targets], loss) if self.metrics else {}
            t1, t3 = self._evaluate_temporal_tasks()
            if self.forgetting_engine:
                self.forgetting_engine.record_eval_snapshot(
                    [float("nan") if score is None else score for score in (t1, t3)], task=task
                )
            if self.comm_channel:
                self.total_traffic_mb = self.comm_channel.get_bandwidth_report()["total_traffic_mb"]
            else:
                self.total_traffic_mb += sum(sum(v.numel() * 4 for v in d.values()) for _, _, d in updates) / 1048576
            eps = self.accountant.get_epsilon()
            values = {"round": round_idx, "accuracy": acc, "loss": loss, "epsilon": eps,
                      "traffic": self.total_traffic_mb, "total_traffic_mb": self.total_traffic_mb,
                      "f1_score": stats.get("f1_score", 0.0), "client_participation": len(updates),
                      "forgetting_T1": t1, "forgetting_T3": t3, "quantization_bits": bits,
                      "mean_update_similarity": sum(similarities.values()) / len(similarities) if similarities else 1.0,
                      "active_task": task}
            for key, value in values.items(): history[key].append(value)
            logger.info(f"Round {round_idx}: acc={acc:.4f}, eps={eps:.3f}, bits={bits}, clients={len(updates)}")
            if self.stopper.should_stop(loss, acc, eps): break

        history["client_participation_total"] = self.client_participation
        history["privacy_report"] = self.accountant.get_privacy_report()
        history["drift_events"] = drift_events if 'drift_events' in locals() else {}
        history["forgetting_report"] = self.forgetting_engine.generate_stability_report() if self.forgetting_engine else {}
        return history

    def _evaluate(self, return_raw=False):
        if self.test_loader is None: return 0.0, 0.0, None, None
        if self.personalization_config.get("enabled"):
            return self._evaluate_personalized()
        model, loss_fn = self.server.model.to(self.device), torch.nn.CrossEntropyLoss()
        model.eval(); total = correct = 0; loss_sum = 0.0; logits = []; targets = []
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device); out = model(X); loss_sum += loss_fn(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item(); total += len(y); logits.append(out); targets.append(y)
        return loss_sum / total, correct / total, torch.cat(logits), torch.cat(targets)

    def _evaluate_personalized(self):
        """Evaluate each validation shard with its own private classifier head."""
        loss_fn = torch.nn.CrossEntropyLoss(); total = correct = 0; loss_sum = 0.0; logits = []; targets = []
        with torch.no_grad():
            for cid, loader in self.val_loaders.items():
                if loader is None: continue
                model = self.clients[cid].model.to(self.device); model.eval()
                for X, y in loader:
                    X, y = X.to(self.device), y.to(self.device); out = model(X)
                    loss_sum += loss_fn(out, y).item() * len(y); correct += (out.argmax(1) == y).sum().item(); total += len(y)
                    logits.append(out); targets.append(y)
        return loss_sum / total, correct / total, torch.cat(logits), torch.cat(targets)

    def _evaluate_temporal_tasks(self):
        scores = {}
        with torch.no_grad():
            for task in ("T1", "T3"):
                correct = total = 0
                for cid, periods in self.temporal_loaders.items():
                    model = (self.clients[cid].model if self.personalization_config.get("enabled") else self.server.model).to(self.device)
                    model.eval()
                    for X, y in ([*periods[task]] if periods.get(task) else []):
                        pred = model(X.to(self.device)).argmax(1); correct += (pred == y.to(self.device)).sum().item(); total += len(y)
                scores[task] = correct / total if total else None
        return scores["T1"], scores["T3"]
