"""Similarity-aware aggregation for heterogeneous federated clients."""
import torch


class SimilarityAwareAggregator:
    """Down-weights client updates that disagree with the round consensus.

    This is intentionally a small extension of FedAvg: it uses only model
    updates already sent by clients and therefore does not require sharing raw
    examples or expensive pairwise optimal-transport computations.
    """

    def __init__(self, similarity_floor=0.05):
        self.similarity_floor = float(max(0.0, similarity_floor))
        self.last_similarities = {}

    @staticmethod
    def _flatten(update):
        return torch.cat([value.detach().float().reshape(-1).cpu() for value in update.values()])

    def aggregate(self, updates):
        """Return an aggregate delta and per-client cosine similarities.

        ``updates`` contains ``(client_id, num_examples, state_dict)`` tuples.
        Sample count remains a factor, while cosine alignment with the weighted
        consensus protects the global model from strongly divergent updates.
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty update list.")

        total_examples = sum(num_examples for _, num_examples, _ in updates)
        consensus = None
        for _, num_examples, update in updates:
            vector = self._flatten(update)
            weighted = vector * (num_examples / total_examples)
            consensus = weighted if consensus is None else consensus + weighted

        raw_weights, similarities = [], {}
        for client_id, num_examples, update in updates:
            vector = self._flatten(update)
            similarity = torch.nn.functional.cosine_similarity(
                vector.unsqueeze(0), consensus.unsqueeze(0), dim=1
            ).item()
            similarities[client_id] = similarity
            raw_weights.append(num_examples * max(self.similarity_floor, (similarity + 1.0) / 2.0))

        normalizer = sum(raw_weights)
        aggregate = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in updates[0][2].items()}
        for (_, _, update), weight in zip(updates, raw_weights):
            for key in aggregate:
                aggregate[key] += update[key].float() * (weight / normalizer)

        self.last_similarities = similarities
        return aggregate, similarities
