import argparse
import multiprocessing as mp
import os
import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np
import tqdm

# === Set Up Paths ===

CALVIN_ROOT_PATH = Path(__file__).resolve().parents[2] / "HD-ExpIt"
sys.path.insert(0, str(CALVIN_ROOT_PATH))  # Top-level project root

sys.path.extend(
    [
        str(CALVIN_ROOT_PATH / "franka_3blocks_env_pybullet"),
    ]
)

# === CALVIN Imports ===
from franka_3blocks_env_pybullet.Franka3BlocksEnv import Franka3BlocksEnvPyBullet

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


mp.set_start_method("spawn", force=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Data path",
        default="data/franka3b_env_demos/training",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug images during training.",
    )

    parser.add_argument(
        "--replan",
        action="store_true",
        help="Replan subgoals every 64 steps.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        help="Mode for state buffer saving.",
        default="start_end_all",
        choices=[
            "end_all",
            "reset",
        ],
    )

    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Balance states between tasks in the buffer",
    )

    args = parser.parse_args()

    dict_path = os.path.join(args.data_path, "lang_annotations/auto_lang_ann.npy")
    if not os.path.exists(dict_path):
        dict_path = os.path.join(args.data_path, "auto_lang_ann.npy")
    dict_ann = np.load(dict_path, allow_pickle=True).item()

    # Initialize state buffer
    env_cfg = {"num_envs": 1, "max_episode_length": 500, "task_mode": "random"}
    env = Franka3BlocksEnvPyBullet(env_cfg, headless=True)

    buffer = []
    if args.mode == "end_all":
        for task, (start, end) in zip(
            dict_ann["language"]["task"], dict_ann["info"]["indx"]
        ):
            if task == "stack":
                pass  # Skip stacking tasks for now
            else:
                # load last frame
                episode_path = os.path.join(args.data_path, f"episode_{end:07d}.npz")
                frame = np.load(episode_path)
                scene_obs = frame["states"]
                buffer.append(scene_obs)
    elif args.mode == "reset":
        for _ in range(2000):
            obs = env.reset()
            buffer.append(obs["state"].cpu().numpy())
    else:
        raise NotImplementedError

    print("Total states collected for buffer:", len(buffer), "(for all tasks)")

    # Save buffer to file
    buffer_name = f"state_buffer_{args.mode}.pkl"
    buffer_save_path = os.path.join(args.data_path, buffer_name)
    pickle.dump(buffer, open(buffer_save_path, "wb"))
    print(f"Saved state buffer to {buffer_save_path}")


if __name__ == "__main__":
    # Calvin config
    main()
