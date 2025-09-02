import random
import torch
import numpy as np



def get_base_model(model):
    base_model = getattr(model, "base_model", model)
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return base_model.model
    return base_model


def get_embedding_device(model):
    base_model = get_base_model(model)
    if hasattr(base_model, "embed_tokens"):
        return base_model.embed_tokens.weight.device
    if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "wte"):
        return base_model.transformer.wte.weight.device
    return next(model.parameters()).device


def set_deterministic_backend(seed:int=42) -> None:
    """ 
    Forces PyTorch to use only deterministic operations (e.g., disables non-deterministic GPU kernels).
    This is crucial for reproducibility: given the same inputs and model state, to get the same outputs every time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True