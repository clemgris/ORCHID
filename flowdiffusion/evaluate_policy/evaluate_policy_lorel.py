"""Data generation script for LORL."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import sys
from collections import Counter

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_path, "lorel/env"))


import gymnasium as gym
import lorl_env  # noqa: F401
import numpy as np
import torch
import torchvision
from model.hierarchical_model_lorel import HierarchicalModel
from omegaconf import DictConfig, OmegaConf
from pyvirtualdisplay import Display
from termcolor import colored
from tqdm import trange
from utils.vis import save_gif

NUM_EPISODES_PER_TASK = 10
MAX_STEPS = 20


def get_motion_desc_sawyer(s0, st, color):
    """Generate motion description for object
    based on initial and final XY position"""
    dl = st - s0
    descr = f"move {color} "
    dirs = []
    if dl[1] > 0.02:
        dirs.append("down ")
    elif dl[1] < -0.02:
        dirs.append("up ")
    if dl[0] > 0.02:
        dirs.append("left ")
    elif dl[0] < -0.02:
        dirs.append("right ")
    random.shuffle(dirs)
    if len(dirs) > 0:
        direction = "and ".join(dirs)
        descr += direction
    else:
        return []
    return [descr]


def check_task_accomplished(task, st0, st):
    """
    Check if a given task string (from TASKS) is satisfied by the state change from st0 to st.
    TASKS include:
      - open/close drawer
      - turn faucet left/right
      - move <color> mug <direction>
    """
    task = task.lower().strip()

    # Drawer
    if task == "open drawer":
        return st[14] - st0[14] < -0.02
    elif task == "close drawer":
        return st[14] - st0[14] > 0.02

    # Faucet
    elif task == "turn faucet left":
        return st[13] - st0[13] > np.pi / 10
    elif task == "turn faucet right":
        return st[13] - st0[13] < -np.pi / 10

    # Mug motion
    elif task.startswith("move white mug") or task.startswith("move black mug"):
        if "white" in task:
            s0, s1 = st0[9:11], st[9:11]
        else:
            s0, s1 = st0[11:13], st[11:13]
        dl = s1 - s0

        if "down" in task:
            return dl[1] > 0.02
        elif "up" in task:
            return dl[1] < -0.02
        elif "left" in task:
            return dl[0] > 0.02
        elif "right" in task:
            return dl[0] < -0.02

    return False  # Unknown or unmatched task


TASKS = [
    "open drawer",
    "close drawer",
    "turn faucet left",
    "turn faucet right",
    "move white mug down",
    "move black mug right",
]


def evaluate_policy(env, model, eval_folder, debug_path, args):
    results = Counter()
    tot_tasks = Counter()

    for task in TASKS:
        for _ in trange(NUM_EPISODES_PER_TASK):
            success, length = rollout(env, task, model, debug_path)
            results[task] += success
            tot_tasks[task] += 1
            print(f"{task}: {results[task]} / {tot_tasks[task]} ({length})")

    print("\nResults\n" + "-" * 60)
    for task in results:
        print(f"{task}: {results[task]} / {tot_tasks[task]}")

    print(f"SR: {sum(results.values()) / sum(tot_tasks.values()) * 100:.1f}%")

    # Save results
    with open(
        os.path.join(args.eval_folder, f"results.txt"), "w"
    ) as f:
        for task in results:
            f.write(f"{task}: {results[task]} / {tot_tasks[task]}\n")
        f.write(
            f"SR: {sum(results.values()) / sum(tot_tasks.values()) * 100:.1f}%\n"
        )


def rollout(env, task, model, debug_path):
    done = False
    step = 0
    obs_list = []
    subgoals = []
    im = env.reset()
    im = im[0]
    initial_im = im.copy()

    ## DEBUG
    # path = "/home/grislain/SkillDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_1k/training/data_with_dino_vit_features/data_0.npz"
    # episode = np.load(path, allow_pickle=True)
    ## DEBUG

    while not done:
        # Reset environment
        st0 = env.unwrapped.data.qpos[:].copy()
        action = model.step(im, task).cpu().numpy()
        # action = episode["actions"][step]  # DEBUG

        obs_list.append(torch.Tensor(im).permute(2, 0, 1))
        if args.save_failures and hasattr(model, "sub_goals"):
            sample_subgoals = (
                step % model.ref_traj_length == 0 if model.replan else step == 0
            )
            if sample_subgoals:
                subgoals.append(model.sub_goals[0])

        st = env.unwrapped.data.qpos[:].copy()
        # breakpoint()
        im, r, done, _ = env.step(action)
        step += 1
        success = check_task_accomplished(task, st0, st)

        if success:
            print(colored("S", "green"), end=" ")
            return True, step
        if step > MAX_STEPS:
            break
    print(colored("F", "red"), end=" ")
    if args.save_failures:
        # Create folder for this failed episode
        os.makedirs(debug_path + "/failures/", exist_ok=True)
        failure_idx = len(os.listdir(debug_path + "/failures/"))
        failed_episode_path = os.path.join(
            debug_path, f"failures/failed_{task.replace(' ', '_')}_{failure_idx}"
        )
        os.makedirs(failed_episode_path, exist_ok=True)

        # Save episode (as png)
        torchvision.utils.save_image(
            torch.stack(obs_list),
            os.path.join(
                failed_episode_path,
                "trajectory.png",
            ),
        )
        # Save subgoals
        for kk, subgoal in enumerate(subgoals):
            # model.save_image(
            #     subgoal,
            #     os.path.join(failed_episode_path, f"subgoals_{kk}.png"),
            # )
            torchvision.utils.save_image(
                torch.concatenate(
                    [torch.Tensor(initial_im).permute(2, 0, 1)[None], (subgoal + 1) / 2]
                ),
                os.path.join(failed_episode_path, f"subgoals_{kk}.png"),
            )
        # Save episode (as gif)
        save_gif(
            obs_list,
            os.path.join(failed_episode_path, "trajectory.gif"),
            duration=5.0,
            norm=False,
        )

    return False, step


def main(args):
    np.random.seed(args.seed)

    ### Data parameters


if __name__ == "__main__":
    # Start the virtual display
    display = Display(
        visible=0, size=(1400, 900)
    )  # 'visible=0' means no physical display required
    display.start()

    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )

    parser.add_argument(
        "--eval_folder",
        type=str,
        help="Where to log the evaluation results.",
        default="eval_long_horizon",
    )

    parser.add_argument(
        "--policy_checkpoint_num",
        type=int,
        help="Policy checkpoint num",
        default=1033,
    )

    parser.add_argument(
        "--policy_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/calvin/models/policy_huit",
    )

    parser.add_argument(
        "--high_level_checkpoint_num",
        type=int,
        help="High level checkpoint number",
        default=100,
    )

    parser.add_argument(
        "--high_level_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/calvin/models/results_huit_CLIP",
    )

    parser.add_argument(
        "--server",
        "-s",
        type=str,
        help="Server",
        default="hacienda",
    )

    parser.add_argument(
        "--debug_path",
        type=str,
        help="Path to save debug images.",
        default="/home/grislain/AVDC/debug_eval_lorel",
    )

    parser.add_argument(
        "--num_subgoals",
        type=int,
        help="Number of subgoals to generate.",
        default=5,
    )

    parser.add_argument(
        "--replan",
        action="store_true",
        help="Replan subgoals every n steps.",
    )

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    args.save_failures = True if args.debug_path else False

    # Load data config
    policy_data_config = OmegaConf.load(
        os.path.join(args.policy_results_folder, "data_config.yaml")
    )
    if args.server == "hacienda":
        policy_data_config.root = "/home/grislain/AVDC/lorel/data/dec_24_sawver_50k/training/data_with_dino_vit_features"
    high_level_data_config = OmegaConf.load(
        os.path.join(args.high_level_results_folder, "data_config.yaml")
    )

    config = DictConfig(
        {
            "policy": {
                "checkpoint_num": args.policy_checkpoint_num,
                "results_folder": args.policy_results_folder,
                **policy_data_config,
            },
            "high_level": {
                "checkpoint_num": args.high_level_checkpoint_num,
                "results_folder": args.high_level_results_folder,
                "use_oracle_subgoals": False,
                **high_level_data_config,
            },
            "debug_path": args.debug_path,
            "server": args.server,
            "num_subgoals": args.num_subgoals,
            "replan": args.replan,
            "device": args.device,
        }
    )

    feat_transforms_dict = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image_transforms_dict = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(
            #     mean=[0.5, 0.5, 0.5],
            #     std=[0.5, 0.5, 0.5],
            # ),
        ]
    )

    transforms = {
        "pixel": image_transforms_dict,
        "feat": feat_transforms_dict,
    }

    envname = "LorlEnv-v0"
    env = gym.make(envname)

    model = HierarchicalModel(config, transforms)

    evaluate_policy(
        env,
        model=model,
        eval_folder=args.eval_folder,
        debug_path=args.debug_path,
        args=args,
    )

    display.stop()
