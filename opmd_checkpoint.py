"""Resolve checkpoint metadata without guessing which architecture produced it."""

from argparse import Namespace

from opmd_cli import MODEL_SPECS


def resolve_checkpoint(checkpoint, model_kind=None, **overrides):
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Expected a nonempty checkpoint dictionary")
    wrapped = "model" in checkpoint
    config = checkpoint.get("config", {}) if wrapped else {}
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary")
    recorded = checkpoint.get("model_kind") if wrapped else None
    config_kind = config.get("model_kind")
    if recorded and config_kind and recorded != config_kind:
        raise ValueError("Checkpoint model_kind and config.model_kind disagree")
    recorded = recorded or config_kind
    if wrapped and not recorded:
        raise ValueError("Checkpoint has no architecture metadata. Use a checkpoint produced by "
                         "one of the five numbered model entries.")
    if model_kind and recorded and model_kind != recorded:
        raise ValueError(f"Requested {model_kind}, but checkpoint records {recorded}")
    kind = recorded or model_kind
    if kind not in MODEL_SPECS:
        raise ValueError("Specify --model_kind for a bare five-architecture state_dict")
    state = checkpoint["model"] if wrapped else checkpoint
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model state is empty or invalid")
    if all(isinstance(k, str) and k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    values = {
        "dim": config.get("dim", 512),
        "heads": config.get("heads", 8),
        "size": config.get("size", 224),
        "max_text_len": config.get("max_text_len", 64),
        "pair_policy": config.get("pair_policy", "all"),
        "bert_path": config.get("bert_path", "google-bert/bert-base-chinese"),
    }
    for key, value in overrides.items():
        if value is not None:
            # Changing dimensions or preprocessing on a configured checkpoint is unsafe.
            if key in ["dim", "heads", "size", "max_text_len"] and key in config and value != config[key]:
                raise ValueError(f"--{key}={value} differs from checkpoint value {config[key]}")
            values[key] = value
    if values["dim"] <= 0 or values["heads"] <= 0 or values["dim"] % values["heads"]:
        raise ValueError("Invalid checkpoint dim/heads")
    if not 0 < values["size"] <= 224 or values["max_text_len"] <= 0:
        raise ValueError("Invalid checkpoint image/text size")
    if values["pair_policy"] not in ["all", "one_fixed"]:
        raise ValueError("Invalid checkpoint pair_policy")
    return kind, state, Namespace(**values), config
