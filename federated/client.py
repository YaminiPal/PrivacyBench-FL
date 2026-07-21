import copy
import torch
from privacy.dp import DPEngine


class FLClient:
    def __init__(self, client_id, model, device="cpu"):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.last_update = None
        self.last_update_norm = 0.0

    def _base_model(self):
        if hasattr(self.model, "_module"):
            return self.model._module
        return self.model

    def _personalized_keys(self):
        model = self._base_model()
        return set(model.personalized_parameter_keys()) if hasattr(model, "personalized_parameter_keys") else set()

    def train(
        self,
        train_loader,
        epochs=1,
        lr=0.01,
        optimizer_cls=torch.optim.SGD,
        loss_fn=torch.nn.CrossEntropyLoss(),
        dp_engine=None,
        lambda_reg=1e-4,
        clipper=None,
    ):
        base = self._base_model()
        initial_state = {
            k: v.detach().cpu().clone()
            for k, v in base.state_dict().items()
        }

        working_model = copy.deepcopy(base)
        optimizer = optimizer_cls(working_model.parameters(), lr=lr)
        working_model.train()

        if dp_engine is not None:
            working_model, optimizer, train_loader = dp_engine.make_private(
                model=working_model,
                optimizer=optimizer,
                data_loader=train_loader,
            )

        for _ in range(epochs):
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = working_model(X)
                base_loss = loss_fn(output, y)
                l2_model = (
                    working_model._module
                    if hasattr(working_model, "_module")
                    else working_model
                )
                l2_penalty = (
                    l2_model.get_l2_penalty(lambda_reg=lambda_reg)
                    if hasattr(l2_model, "get_l2_penalty")
                    else 0.0
                )
                total_loss = base_loss + l2_penalty
                total_loss.backward()
                optimizer.step()

        trained_state = DPEngine.get_clean_state_dict(working_model)
        base.load_state_dict(trained_state)
        self.model = base

        personalized_keys = self._personalized_keys()
        self.last_update = {
            k: trained_state[k].detach().cpu() - initial_state[k]
            for k in initial_state if k not in personalized_keys
        }

        if clipper is not None:
            self.last_update, _ = clipper.clip_update(self.last_update)

        self.last_update_norm = self._compute_norm(self.last_update)
        return self.last_update

    def _compute_norm(self, update_dict):
        total_sq = sum(torch.sum(v ** 2).item() for v in update_dict.values())
        return torch.sqrt(torch.tensor(total_sq)).item()

    def set_parameters(self, global_state_dict):
        model = self._base_model()
        local_state = model.state_dict()
        for key, value in global_state_dict.items():
            if key not in self._personalized_keys():
                local_state[key] = value.detach().clone()
        model.load_state_dict(local_state)

    def serialize_update(self, quantizer=None, comm_channel=None):
        if self.last_update is None:
            raise ValueError("Client must run train() before serialize_update().")

        if comm_channel is not None:
            return comm_channel.package_payload(self.last_update)

        if quantizer is not None:
            return quantizer.quantize(self.last_update)

        return {k: v.numpy() for k, v in self.last_update.items()}
