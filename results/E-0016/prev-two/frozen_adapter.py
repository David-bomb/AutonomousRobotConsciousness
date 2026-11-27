"""Публичная загрузка prev-two. Один вызов choose на каждый реальный физический ход."""
from hashlib import sha256
from io import BytesIO
import torch


def load(checkpoint:bytes,expected_sha256=None,ablate=False):
    if expected_sha256 is not None and sha256(checkpoint).hexdigest()!=expected_sha256:
        raise ValueError("Не совпадает SHA-256 выбранных весов")
    state=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    if "patch_score.0.weight" in state:
        from .patch_adapter import from_bytes
    elif "route_parent_mix" in state:
        from .route_adapter import from_bytes
    else:
        from .adapter import from_bytes
    return from_bytes(checkpoint,ablate=ablate)


from_bytes=load
