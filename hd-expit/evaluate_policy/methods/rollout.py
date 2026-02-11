import logging
import os
import sys
from pathlib import Path

import torch
import torchvision
from termcolor import colored

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "hd-expit"),
        str(ROOT_PATH / "calvin/calvin_models"),
        str(ROOT_PATH / "franka_3blocks_env_pybullet"),
        str(ROOT_PATH / "franka_3blocks_env_pybullet/Franka3BlocksEnv"),
    ]
)

from franka_3blocks_env_pybullet.Franka3BlocksEnv import (
    TASK_INSTRUCTIONS as franka3blocks_annotations,
)
from utils.vis import save_gif

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


def rollout_with_oracle(
    env,
    model,
    episode,
    task_oracle,
    args,
    task,
):
    reset_info = episode["state_info"]
    obs = env.reset(
        robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"][0]
    )
    lang_annotation = episode["lang"]

    model.reset()
    start_info = env.get_info()
    obs_list = []
    subgoals = []
    for step in range(args.ep_len):
        # action = episode["actions"][step]
        action = model.step(obs, lang_annotation, episode)
        obs, _, _, current_info = env.step(action)
        if args.save_failures:
            obs_list.append(obs["rgb_obs"]["rgb_static"][0, 0])
        # Check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {task.replace(" ", "_")}
        )

        if args.save_failures and hasattr(model, "sub_goals"):
            sample_subgoals = (
                step % model.ref_traj_length == 0 if model.replan else step == 0
            )
            if sample_subgoals:
                subgoals.append(model.sub_goals[0, :, 0])

        if len(current_task_info) > 0:
            print(colored("S", "green"), end=" ")
            return True, step
    print(colored("F", "red"), end=" ")
    if args.save_failures:
        if args.save_failures:
            # Create folder for this failed episode
            os.makedirs(args.debug_path, exist_ok=True)
            failed_episode_path = os.path.join(
                args.debug_path, f"failed_{task.replace(' ', '_')}_{episode['idx']}"
            )
            os.makedirs(
                failed_episode_path,
                exist_ok=True,
            )

        # Save episode (as png)
        torchvision.utils.save_image(
            (torch.stack(obs_list) + 1) / 2,
            os.path.join(
                failed_episode_path,
                "trajectory.png",
            ),
        )
        # Save subgoals
        for kk, subgoal in enumerate(subgoals):
            model.save_image(
                subgoal,
                f"failed_{task.replace(' ', '_')}_{episode['idx']}/subgoals_{kk}.png",
            )
        # Save episode (as gif)
        save_gif(
            obs_list, os.path.join(failed_episode_path, "trajectory.gif"), duration=1.0
        )

    return False, step


def rollout(env, model, task_oracle, subtask, val_annotations, debug_path=None):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    obs_list = []
    subgoals = []
    init = []
    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if debug_path:
            obs_list.append(obs["rgb_obs"]["rgb_static"][0, 0])
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if debug_path:
            sample_subgoals = (
                step % model.ref_traj_length == 0 if model.replan else step == 0
            )
            if sample_subgoals:
                subgoals.append(model.sub_goals[0, :, 0])
                init.append(model.init_subgoal_gen[0])
        if len(current_task_info) > 0:
            print(colored("S", "green"), end=" ")
            return True, step
    print(colored("F", "red"), end=" ")

    if debug_path:
        # Create folder for this failed episode
        os.makedirs(debug_path + "/failures/", exist_ok=True)
        failure_idx = len(os.listdir(debug_path + "/failures/"))
        failed_episode_path = os.path.join(
            debug_path, f"failures/failed_{subtask.replace(' ', '_')}_{failure_idx}"
        )
        os.makedirs(
            failed_episode_path,
            exist_ok=True,
        )
        # Save episode (as png)
        torchvision.utils.save_image(
            (torch.stack(obs_list) / 255).float(),
            os.path.join(
                failed_episode_path,
                "trajectory.png",
            ),
        )
        # Save episode (as gif)
        save_gif(
            obs_list,
            os.path.join(failed_episode_path, "trajectory.gif"),
            duration=1.0,
        )
        # Save subgoals
        if not model.cfg.policy.datamodule.lang_dataset.get("without_guidance", False):
            for kk, subgoal in enumerate(subgoals):
                model.save_image(
                    torch.cat([init[kk].cpu(), subgoal.cpu()]),
                    os.path.join(
                        f"failures/failed_{subtask.replace(' ', '_')}_{failure_idx}",
                        f"subgoals_{kk}.png",
                    ),
                )
    return False, step


def rollout_toy(env, model, task, task_id, reward_fn, debug_path=None):
    """
    Run the actual rollout on one task (which is one natural language instruction).
    """
    # get lang annotation for task
    lang_annotation = franka3blocks_annotations[task]

    obs = env.reset(task_id=task_id)
    init_obs = obs.copy()

    model.reset()

    obs_list = []
    subgoals = []
    init = []
    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if debug_path:
            obs_list.append(obs["pixels"][0])
        # check if current step solves a task
        reward = reward_fn(init_obs, obs)
        if debug_path:
            sample_subgoals = (
                step % model.ref_traj_length == 0 if model.replan else step == 0
            )
            if sample_subgoals:
                subgoals.append(model.sub_goals[0, :, 0])
                init.append(model.init_subgoal_gen[0])
        if reward > 0.5:
            if debug_path:
                os.makedirs(debug_path + "/success/", exist_ok=True)
                success_idx = len(os.listdir(debug_path + "/success/"))
                success_episode_path = os.path.join(
                    debug_path,
                    f"success/succeed_{task.replace(' ', '_')}_{success_idx}",
                )
                os.makedirs(
                    success_episode_path,
                    exist_ok=True,
                )
                # Save episode (as png)
                torchvision.utils.save_image(
                    (torch.stack(obs_list)).float(),
                    os.path.join(
                        success_episode_path,
                        "trajectory.png",
                    ),
                )
                # Save episode (as gif)
                save_gif(
                    obs_list,
                    os.path.join(success_episode_path, "trajectory.gif"),
                    duration=1.0,
                    norm=False,
                )
            print(colored("S", "green"), end=" ")
            return True, step
    print(colored("F", "red"), end=" ")

    if debug_path:
        # Create folder for this failed episode
        os.makedirs(debug_path + "/failures/", exist_ok=True)
        failure_idx = len(os.listdir(debug_path + "/failures/"))
        failed_episode_path = os.path.join(
            debug_path, f"failures/failed_{task.replace(' ', '_')}_{failure_idx}"
        )
        os.makedirs(
            failed_episode_path,
            exist_ok=True,
        )
        # Save episode (as png)
        torchvision.utils.save_image(
            (torch.stack(obs_list)).float(),
            os.path.join(
                failed_episode_path,
                "trajectory.png",
            ),
        )
        # Save episode (as gif)
        save_gif(
            obs_list,
            os.path.join(failed_episode_path, "trajectory.gif"),
            duration=1.0,
            norm=False,
        )
        # Save subgoals
        if not model.cfg.policy.dataset.get("without_guidance", False):
            for kk, subgoal in enumerate(subgoals):
                model.save_image(
                    torch.cat([init[kk][None, ...].cpu(), subgoal.cpu()]),
                    os.path.join(
                        f"failures/failed_{task.replace(' ', '_')}_{failure_idx}",
                        f"subgoals_{kk}.png",
                    ),
                )
    return False, step
