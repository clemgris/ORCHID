# === Standard Library ===
import logging
import os
import sys
from pathlib import Path
from typing import List

# === Third-party Libraries ===
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import colored

# === Project Path Setup ===
ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
        str(ROOT_PATH / "Franka3BlocksEnv"),
        str(ROOT_PATH / "Franka3BlocksEnv/Franka3BlocksEnv"),
    ]
)

# === Local Imports ===
from state_buffer import StateBuffer

from methods.rollout import rollout_data_collection, rollout_data_collection_toy

# === Logger ===
logger = logging.getLogger(__name__)


NUM_SEQUENCES = 1000


def generate_new_data(
    model,
    env,
    debug_path=None,
    conf_dir=None,
    num_data=1000,
    task: str = None,
    saving_path: str = None,
    num_trials: int = 1,
    state_buffer: StateBuffer = None,
    start_idx: int = 0,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_folder: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.

    Returns:
        0 on success
    """
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    all_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable.yaml")

    ann_saving_path = os.path.join(saving_path, "lang_annotations/auto_lang_ann.npy")
    dirpath = os.path.dirname(ann_saving_path)
    os.makedirs(dirpath, exist_ok=True)

    if not os.path.exists(ann_saving_path):
        auto_lang_ann = {
            "info": {"episodes": [], "indx": [], "length": [], "num_trials": []},
            "language": {"ann": [], "task": []},
        }
    else:
        auto_lang_ann = np.load(ann_saving_path, allow_pickle=True).item()

    success_counter = 0
    suffled_idx = np.random.permutation(len(state_buffer.valid_idx[task]))
    for ii in suffled_idx:
        _, robot_obs, scene_obs = state_buffer.get(task, ii)
        done = False
        num_trial = 0
        while (not done) and (num_trial < num_trials):
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            success, length, (start, end), ann = rollout_data_collection(
                env,
                model,
                task_oracle,
                task,
                all_annotations,
                debug_path,
                saving_path,
                start_idx,
            )
            num_trial += 1
            if success:
                success_counter += 1
                print(colored("S", "green"), task, "trial", num_trial, end=" ")

                auto_lang_ann["info"]["indx"].append((start, end))
                auto_lang_ann["language"]["ann"].append(ann)
                auto_lang_ann["language"]["task"].append(task)
                auto_lang_ann["info"]["length"].append(length)
                auto_lang_ann["info"]["num_trials"].append(num_trial)

                done = True

            if success_counter % 5 == 0:
                print("Saved", success_counter, "episodes for the task", task)
                # Save language annotations
                np.save(
                    ann_saving_path,
                    auto_lang_ann,
                    allow_pickle=True,
                )
        if success_counter >= num_data:
            break

    print(
        "Created",
        success_counter,
        "successful episodes out of",
        len(state_buffer.valid_idx[task]),
        "initial states for the task",
        task,
        "at",
        saving_path,
    )

    return 0


def generate_new_data_toy(
    model,
    env,
    debug_path=None,
    num_data=1000,
    task: str = None,
    task_id: int = None,
    saving_path: str = None,
    num_trials: int = 1,
    state_buffer: List = None,
    start_idx: int = 0,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_folder: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.

    Returns:
        0 on success
    """
    ann_saving_path = os.path.join(saving_path, "lang_annotations/auto_lang_ann.npy")
    if not os.path.exists(ann_saving_path):
        ann_saving_path = os.path.join(saving_path, "auto_lang_ann.npy")
    dirpath = os.path.dirname(ann_saving_path)
    os.makedirs(dirpath, exist_ok=True)

    if not os.path.exists(ann_saving_path):
        auto_lang_ann = {
            "info": {"episodes": [], "indx": [], "length": [], "num_trials": []},
            "language": {"ann": [], "task": []},
        }
    else:
        auto_lang_ann = np.load(ann_saving_path, allow_pickle=True).item()

    success_counter = 0
    suffled_idx = np.random.permutation(len(state_buffer))
    if task_id == 3:
        reward_fn = (  # noqa: E731
            lambda x, y: env.compute_reward_stack(x, y, up_cube="A", bottom_cube="B")
            or env.compute_reward_stack(x, y, up_cube="A", bottom_cube="C")
            or env.compute_reward_stack(x, y, up_cube="B", bottom_cube="A")
            or env.compute_reward_stack(x, y, up_cube="B", bottom_cube="C")
            or env.compute_reward_stack(x, y, up_cube="C", bottom_cube="A")
            or env.compute_reward_stack(x, y, up_cube="C", bottom_cube="B")
        )
    else:
        reward_fn = env.get_reward_function(task_id)

    for ii in suffled_idx:
        scene_obs = state_buffer[ii]
        done = False
        num_trial = 0
        while (not done) and (num_trial < num_trials):
            success, length, (start, end), ann = rollout_data_collection_toy(
                env,
                model,
                reward_fn,
                task,
                task_id,
                torch.tensor(scene_obs),
                debug_path,
                saving_path,
                start_idx,
            )
            num_trial += 1
            if success:
                success_counter += 1
                print(colored("S", "green"), task, "trial", num_trial, end=" ")

                auto_lang_ann["info"]["indx"].append((start, end))
                auto_lang_ann["language"]["ann"].append(ann)
                auto_lang_ann["language"]["task"].append(task)
                auto_lang_ann["info"]["length"].append(length)
                auto_lang_ann["info"]["num_trials"].append(num_trial)

                done = True

            if success_counter % 5 == 0:
                print("Saved", success_counter, "episodes for the task", task)
                # Save language annotations
                np.save(
                    ann_saving_path,
                    auto_lang_ann,
                    allow_pickle=True,
                )
        if success_counter >= num_data:
            break

    print(
        "Created",
        success_counter,
        "successful episodes out of",
        len(state_buffer),
        "initial states for the task",
        task,
        "at",
        saving_path,
    )

    return 0
