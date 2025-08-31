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