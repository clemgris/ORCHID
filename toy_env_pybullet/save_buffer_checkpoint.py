import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

# import ddpo_pytorch.prompts
# import ddpo_pytorch.rewards
import tqdm

# === Set Up Paths ===

CALVIN_ROOT_PATH = Path(__file__).resolve().parents[2] / "avdc"
sys.path.insert(0, str(CALVIN_ROOT_PATH))  # Top-level project root

CALVIN_ROOT_PATH = Path(__file__).resolve().parents[2] / "AVDC"
sys.path.insert(0, str(CALVIN_ROOT_PATH))  # Top-level project root

sys.path.extend(
    [
        str(CALVIN_ROOT_PATH / "toy_env_pybullet"),
    ]
)

# === CALVIN Imports ===
from toy_env_pybullet.toyEnv import Franka3CubeEnvPyBullet

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


# === DDPO-PyTorch Imports ===
ROOT_PATH = Path(__file__).resolve().parents[2] / "ddpo-pytorch"
sys.path.insert(0, str(ROOT_PATH))


import multiprocessing as mp

mp.set_start_method("spawn", force=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Data path",
        default="/home/grislain/AVDC/data/toy_env_demos/training",
    )

    parser.add_argument(
        "--server",
        "-s",
        type=str,
        help="Server",
        default="hacienda",
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
        "--result_folder",
        type=str,
        help="Path to save debug images.",
        default="RL_training",
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

    dict_path = os.path.join(args.data_path, "auto_lang_ann.npy")
    dict_ann = np.load(dict_path, allow_pickle=True).item()

    # Initialize state buffer
    env_cfg = {"num_envs": 1, "max_episode_length": 500, "task_mode": "random"}
    env = Franka3CubeEnvPyBullet(env_cfg, headless=True)

    buffer = []

    if args.mode == "end_all":
        for start, end in dict_ann["info"]["indx"]:
            # load last frame
            episode_path = os.path.join(args.data_path, f"episode_{end:07d}.npz")
            frame = np.load(episode_path)
            scene_obs = frame["states"]
            buffer.append(scene_obs)

            # # DEBUG
            # obs = env.reset(scene_obs=torch.tensor(scene_obs))
            # torchvision.utils.save_image(
            #     torch.tensor(obs["pixels"]),
            #     "env_reset.png",
            # )
            # torchvision.utils.save_image(
            #     torch.tensor(frame["rgb_static"]).permute(2, 0, 1), "data.png"
            # )
            # breakpoint()
            # # DEBUG
    elif args.mode == "reset":
        for _ in range(2000):
            obs = env.reset()
            buffer.append(obs["state"].cpu().numpy())

            # # DEBUG
            # torchvision.utils.save_image(
            #     torch.tensor(obs["pixels"]),
            #     "env_reset.png",
            # )
            # breakpoint()
            # # DEBUG
    else:
        raise NotImplementedError

    print("Total states collected for buffer:", len(buffer), "(for all tasks)")

    # Save buffer to file
    buffer_name = f"state_buffer_{args.mode}.pkl"
    buffer_save_path = os.path.join(args.data_path, buffer_name)
    np.save(buffer_save_path, buffer)
    print(f"Saved state buffer to {buffer_save_path}")


if __name__ == "__main__":
    # Calvin config
    main()
