# === Standard Library ===
import logging
import os
import sys
from pathlib import Path

# === Third-party Libraries ===
import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored

# === Project Path Setup ===
ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
    ]
)

# === Local Imports ===
# === CALVIN Imports ===

from methods.rollout import rollout_data_collection

# === DDPO-PyTorch Imports ===
ROOT_PATH = Path(__file__).resolve().parents[2] / "ddpo-pytorch"
sys.path.insert(0, str(ROOT_PATH))

from ddpo_pytorch.state_buffer import StateBuffer

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
        Dictionary with results
    """
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    all_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable.yaml")

    results = []
    ann_saving_path = os.path.join(saving_path, "lang_annotations/auto_lang_ann.npy")
    if not os.path.exists(os.path.dirname(ann_saving_path)):
        auto_lang_ann = {
            "info": {"episodes": [], "indx": [], "length": [], "num_trials": []},
            "language": {"ann": [], "task": []},
        }
        os.makedirs(os.path.dirname(ann_saving_path), exist_ok=False)
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

    return results
