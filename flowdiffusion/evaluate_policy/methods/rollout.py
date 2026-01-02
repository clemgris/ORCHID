import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from termcolor import colored

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
    ]
)

from toyEnv import TASK_INSTRUCTIONS as toy_annotations
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
    lang_annotation = toy_annotations[task]

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


def rollout_data_collection(
    env,
    model,
    task_oracle,
    subtask,
    annotations,
    debug_path=None,
    saving_path=None,
    start_idx=0,
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = random.choice(annotations[subtask])
    model.reset()
    start_info = env.get_info()
    success = False

    frames = []
    obs_list = []
    # Count the number of frames in the saving folder
    frame_idx = (
        sum(
            1
            for f in os.listdir(saving_path)
            if f.startswith("episode_") and os.path.isfile(f"{saving_path}/{f}")
        )
        + start_idx
    )
    for step in range(65):
        action = model.step(obs, lang_annotation)
        frame = {
            "actions": action.detach().cpu().numpy(),
            "rel_actions": None,
            "robot_obs": obs["raw_obs"]["robot_obs"],
            "scene_obs": obs["raw_obs"]["scene_obs"],
            "rgb_static": obs["raw_obs"]["rgb_obs"]["rgb_static"],
            "rgb_gripper": obs["raw_obs"]["rgb_obs"]["rgb_gripper"],
            "depth_static": obs["raw_obs"]["depth_obs"]["depth_static"],
            "depth_gripper": obs["raw_obs"]["depth_obs"]["depth_gripper"],
        }
        frames.append(frame)

        # Take a step in the environment
        obs, _, _, current_info = env.step(action)
        obs_list.append(obs["rgb_obs"]["rgb_static"][0, 0])

        # Check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )

        # If solved, save the frames and return
        if len(current_task_info) > 0:
            success = True
        if success and (step >= 35):
            # Last frame
            action = model.step(obs, lang_annotation)  # last action, not executed
            frame = {
                "actions": action.detach().cpu().numpy(),
                "rel_actions": None,
                "robot_obs": obs["raw_obs"]["robot_obs"],
                "scene_obs": obs["raw_obs"]["scene_obs"],
                "rgb_static": obs["raw_obs"]["rgb_obs"]["rgb_static"],
                "rgb_gripper": obs["raw_obs"]["rgb_obs"]["rgb_gripper"],
                "depth_static": obs["raw_obs"]["depth_obs"]["depth_static"],
                "depth_gripper": obs["raw_obs"]["depth_obs"]["depth_gripper"],
            }
            frames.append(frame)
            # Save the frames
            idx = frame_idx
            for frame in frames:
                idx += 1
                # Save the frame
                frame_path = os.path.join(
                    saving_path,
                    f"episode_{idx:07d}.npz",
                )
                np.savez(frame_path, **frame)

            print(colored("S", "green"), end=" ")
            if debug_path is not None:
                success_path = os.path.join(debug_path, "successes")
                os.makedirs(success_path, exist_ok=True)
                success_idx = len(os.listdir(success_path))
                save_gif(
                    obs_list,
                    os.path.join(
                        success_path, f"trajectory_{subtask}_{success_idx}.gif"
                    ),
                    duration=1.0,
                )

            return True, step, (max(frame_idx + 1, idx - 64), idx), lang_annotation

    print(colored("F", "red"), end=" ")
    if debug_path is not None:
        failures_path = os.path.join(debug_path, "failures")
        os.makedirs(failures_path, exist_ok=True)
        failure_idx = len(os.listdir(failures_path))
        os.makedirs(os.path.join(failures_path, f"failed_{failure_idx}"), exist_ok=True)
        save_gif(
            obs_list,
            os.path.join(
                failures_path, f"failed_{failure_idx}/trajectory_{subtask}.gif"
            ),
            duration=1.0,
        )
        gen_subgoals = model.sub_goals[0, :, 0]
        torchvision.utils.save_image(
            (gen_subgoals.reshape(8, 3, 96, 96) + 1) / 2,
            os.path.join(
                failures_path,
                f"failed_{failure_idx}/subgoals_{subtask}.png",
            ),
        )
    return False, step, (0, 0), None


def rollout_data_collection_toy(
    env,
    model,
    reward_fn,
    task,
    task_id,
    scene_obs,
    debug_path=None,
    saving_path=None,
    start_idx=0,
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    # get lang annotation for subtask
    lang_annotation = toy_annotations[task]
    model.reset()
    obs = env.reset(scene_obs=scene_obs)
    init_obs = obs.copy()
    success = False

    frames = []
    obs_list = []
    # Count the number of frames in the saving folder
    frame_idx = (
        sum(
            1
            for f in os.listdir(saving_path)
            if f.startswith("episode_") and os.path.isfile(f"{saving_path}/{f}")
        )
        + start_idx
    )
    count = 0
    for step in range(65):
        action = model.step(obs, lang_annotation)
        frame = {
            "actions": action.cpu().numpy(),
            "states": obs["state"].cpu().numpy(),
            "rgb_static": obs["pixels"][0].permute(1, 2, 0).cpu().numpy(),
        }
        frames.append(frame)

        # Take a step in the environment
        obs, _, _, current_info = env.step(action)
        obs_list.append(obs["pixels"][0])

        # Check if current step solves a task
        reward = reward_fn(init_obs, obs)

        # If solved, save the frames and return
        if reward > 0.5:
            count += 1
        if count > 5:
            success = True
        if success and (step >= 50):
            # Last frame
            action = model.step(obs, lang_annotation)  # last action, not executed
            frame = {
                "actions": action.cpu().numpy(),
                "states": obs["state"].cpu().numpy(),
                "rgb_static": obs["pixels"][0].permute(1, 2, 0).cpu().numpy(),
            }
            frames.append(frame)
            # Save the frames
            idx = frame_idx
            for frame in frames:
                idx += 1
                # Save the frame
                frame_path = os.path.join(
                    saving_path,
                    f"episode_{idx:07d}.npz",
                )
                np.savez(frame_path, **frame)

            print(colored("S", "green"), end=" ")
            if debug_path is not None:
                success_path = os.path.join(debug_path, "successes")
                os.makedirs(success_path, exist_ok=True)
                success_idx = len(os.listdir(success_path))
                save_gif(
                    obs_list,
                    os.path.join(success_path, f"trajectory_{task}_{success_idx}.gif"),
                    duration=1.0,
                    norm=False,
                )

            return True, step, (max(frame_idx + 1, idx - 64), idx), lang_annotation

    print(colored("F", "red"), end=" ")
    if debug_path is not None:
        failures_path = os.path.join(debug_path, "failures")
        os.makedirs(failures_path, exist_ok=True)
        failure_idx = len(os.listdir(failures_path))
        os.makedirs(os.path.join(failures_path, f"failed_{failure_idx}"), exist_ok=True)
        save_gif(
            obs_list,
            os.path.join(failures_path, f"failed_{failure_idx}/trajectory_{task}.gif"),
            duration=1.0,
            norm=False,
        )
        gen_subgoals = model.sub_goals[0, :, 0]
        torchvision.utils.save_image(
            (gen_subgoals.reshape(8, 3, 96, 96) + 1) / 2,
            os.path.join(
                failures_path,
                f"failed_{failure_idx}/subgoals_{task}.png",
            ),
        )
    return False, step, (0, 0), None
