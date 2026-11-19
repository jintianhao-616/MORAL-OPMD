"""Resolve checkpoint metadata without guessing which architecture produced it."""

from argparse import Namespace
from pathlib import Path

from opmd_cli import MODEL_SPECS, MODEL_WEIGHT_FILENAMES, INFERENCE_DEFAULTS


def model_kind_from_filename(path):
    """Match the five documented filenames; never infer identity from shapes.

    In particular, M4 and M5 share tensor shapes and cannot be distinguished
    from tensor dimensions alone. Keep the downloaded filenames unchanged.
    """
    name = Path(path).name if path is not None else None
    return next((kind for kind, filename in MODEL_WEIGHT_FILENAMES.items()
                 if filename == name), None)


def resolve_checkpoint(checkpoint, model_kind=None, checkpoint_path=None, **overrides):
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
    filename_kind = model_kind_from_filename(checkpoint_path)
    for identity in [recorded, model_kind]:
        if filename_kind and identity and filename_kind != identity:
            raise ValueError(f"Filename {Path(checkpoint_path).name} identifies {filename_kind}, "
                             f"but the requested/recorded model is {identity}")
    kind = recorded or model_kind or filename_kind
    if kind not in MODEL_SPECS:
        raise ValueError("Use M1_weights.pt through M5_weights.pt, or specify --model_kind "
                         "for a renamed bare state_dict")
    state = checkpoint["model"] if wrapped else checkpoint
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model state is empty or invalid")
    if all(isinstance(k, str) and k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    values = {name: config.get(name, default) for name, default in INFERENCE_DEFAULTS.items()}
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

