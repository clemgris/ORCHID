# === Standard Library ===
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

# === Third-party Libraries ===
import torch
from pytorch_lightning import seed_everything

# === Set Up Paths ===
ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))  # Top-level project root
sys.path.extend(
    [
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
    ]
)

# === Local Project Imports ===
# === CALVIN Imports ===
from calvin.calvin_env.calvin_env.envs.play_table_env import get_env
from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import (
    tasks,
)

# === DDPO-PyTorch Imports ===
DPPO_ROOT_PATH = Path(__file__).resolve().parents[2] / "ddpo-pytorch"
sys.path.insert(0, str(DPPO_ROOT_PATH))


# === Device Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def make_env(dataset_path):
    train_folder = Path(dataset_path) / "training"
    env = get_env(train_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


if __name__ == "__main__":
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )

    parser.add_argument(
        "--saving_path",
        type=str,
        required=True,
        help="Path to the CALVIN dataset.",
    )

    parser.add_argument(
        "--ann_folder_name",
        type=str,
        default="new_lang_annotations",
    )

    args = parser.parse_args()

    saving_path = Path(args.saving_path)
    training_folders = list(saving_path.glob("training_*"))

    auto_lang_ann = {
        "info": {"episodes": [], "indx": [], "length": [], "num_trials": []},
        "language": {"ann": [], "task": []},
    }

    # Merge lang_annotation
    for folder in training_folders:
        lang_file = folder / "lang_annotations/auto_lang_ann.npy"
        if lang_file.exists() is False:
            lang_file = folder / "lang_ann.npy"
        if os.path.exists(lang_file):
            ann = np.load(lang_file, allow_pickle=True).item()
            for keys in auto_lang_ann.keys():
                for sub_keys in auto_lang_ann[keys].keys():
                    auto_lang_ann[keys][sub_keys].extend(ann[keys][sub_keys])

    # Assert no overlap in episode indices
    all_start_end = auto_lang_ann["info"]["indx"]
    sorted_indices = sorted(all_start_end, key=lambda x: x[0])
    for i in range(1, len(sorted_indices)):
        assert sorted_indices[i][0] >= sorted_indices[i - 1][1], (
            "Overlap in episode indices detected!"
        )

    # Save
    lang_save_path = saving_path / "training" / args.ann_folder_name
    lang_save_path.mkdir(exist_ok=True)
    np.save(lang_save_path / "auto_lang_ann.npy", auto_lang_ann)

    # Print stats
    for tasks in tasks.keys():
        num_ann = sum(1 for task in auto_lang_ann["language"]["task"] if task == tasks)
        print(f"Number of annotations for task {tasks}: {num_ann}")

    # Move all the files with name start with 'episode_' in folders 'training_*' to training
    for folder in training_folders:
        episode_files = list(folder.glob("episode_*"))
        for file in episode_files:
            new_path = saving_path / "training" / file.name
            file.rename(new_path)
