"""Example TorchServe handler for PyTorch models."""

from __future__ import annotations

import json
from typing import Any

import torch


class FiberOpticHandler:
    """Minimal TorchServe handler."""

    def __init__(self) -> None:
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self, ctx: Any) -> None:
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        model_path = f"{model_dir}/model.pt"
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def preprocess(self, data: Any) -> torch.Tensor:
        batch = []
        for row in data:
            tensor = torch.tensor(row.get("body"), dtype=torch.float32)
            batch.append(tensor)
        return torch.stack(batch, dim=0).to(self.device)

    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(inputs)
            return torch.sigmoid(outputs)

    def postprocess(self, outputs: torch.Tensor) -> list[str]:
        probs = outputs.squeeze(-1).cpu().numpy().tolist()
        return [json.dumps({"probability": prob}) for prob in probs]


_service = FiberOpticHandler()


def handle(data: Any, ctx: Any) -> Any:
    if ctx is not None and _service.model is None:
        _service.initialize(ctx)
    inputs = _service.preprocess(data)
    outputs = _service.inference(inputs)
    return _service.postprocess(outputs)
