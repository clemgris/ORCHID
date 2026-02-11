import logging
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def process_rgb(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_obs_key = "rgb_static"
    rgb_obs = episode[rgb_obs_key]
    # expand dims for single environment obs
    if len(rgb_obs.shape) != 4:
        rgb_obs = np.expand_dims(rgb_obs, axis=0)
    assert len(rgb_obs.shape) == 4
    if window_size == 0 and seq_idx == 0:  # single file loader
        # To Square image
        if isinstance(rgb_obs, np.ndarray):
            seq_rgb_obs_ = torch.from_numpy(rgb_obs * 255).byte().permute(0, 3, 1, 2)
        elif isinstance(rgb_obs, torch.Tensor):
            seq_rgb_obs_ = (rgb_obs * 255).byte().permute(0, 3, 1, 2)
        else:
            raise TypeError(f"Unsupported type {type(rgb_obs)} for rgb_obs")
    else:  # episode loader
        if isinstance(rgb_obs, np.ndarray):
            seq_rgb_obs_ = (
                torch.from_numpy(rgb_obs[seq_idx : seq_idx + window_size] * 255)
                .byte()
                .permute(0, 3, 1, 2)
            )
        elif isinstance(rgb_obs, torch.Tensor):
            seq_rgb_obs_ = (
                (rgb_obs[seq_idx : seq_idx + window_size] * 255)
                .byte()
                .permute(0, 3, 1, 2)
            )
    # we might have different transformations for the different cameras
    if rgb_obs_key in transforms:
        seq_rgb_obs_ = T.Compose(transforms[rgb_obs_key])(seq_rgb_obs_)
    # shape: N_rgb_obs x (BxCxHxW)
    return {"rgb_static": seq_rgb_obs_}


def process_actions(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    # shape: (N_actions)
    action_keys = observation_space["actions"]
    if len(action_keys) != 1:
        raise NotImplementedError
    action_key = action_keys[0]
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if "actions" in transforms:
            action = transforms["actions"]((action, episode["robot_obs"]))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(
            episode[action_key][seq_idx : seq_idx + window_size]
        ).float()
    if seq_acts.dim() == 2:
        return {"actions": seq_acts[:, :]}
    elif seq_acts.dim() == 3:
        return {"actions": seq_acts[:, 0, :]}
    else:
        raise ValueError(f"Unsupported action dimension {seq_acts.dim()}")


def process_language(
    episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
) -> Dict[str, torch.Tensor]:
    seq_lang = {"lang": torch.empty(0), "task": torch.empty(0)}
    if with_lang:
        # lang = torch.from_numpy(episode["language"]).float()
        # if "language" in transforms:
        #     lang = transforms["language"](lang)
        seq_lang["lang"] = episode["language"].replace("_", " ")
        seq_lang["task"] = episode["task"].replace("_", " ")
    return seq_lang


def lookup_naming_pattern(
    dataset_dir: Path, save_format: str
) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())
