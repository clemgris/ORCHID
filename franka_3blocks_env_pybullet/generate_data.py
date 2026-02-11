import argparse
import os
import random

import numpy as np
import torch
from Franka3BlocksEnv import Franka3BlocksEnvPyBullet
from omegaconf import OmegaConf
from PIL import Image

TASK_NAMES = [
    "lift_A",  # 0: Lift cube A
    "lift_B",  # 1: Lift cube B
    "lift_C",  # 2: Lift cube C
    "stack",  # 3: Stack one cube on another (any valid pair)
    "push_C_left",  # 4: Push cube C to the left
    "push_C_right",  # 5: Push cube C to the right
    "push_B_left",  # 6: Push cube B to the left
    "push_B_right",  # 7: Push cube B to the right
    "push_A_left",  # 8: Push cube A to the left
    "push_A_right",  # 9: Push cube A to the right
]

TASK_INSTRUCTIONS = {
    "lift_A": "Lift the red block.",
    "lift_B": "Lift the green block.",
    "lift_C": "Lift the blue block.",
    "stack": "Stack one block on top of another block.",
    "push_A_left": "Push the red block to the left.",
    "push_A_right": "Push the red block to the right.",
    "push_B_left": "Push the green block to the left.",
    "push_B_right": "Push the green block to the right.",
    "push_C_left": "Push the blue block to the left.",
    "push_C_right": "Push the blue block to the right.",
}


def collect_demonstrations(
    saving_path,
    num_trials_per_task=10,
    max_steps=64,
    target_num_episode=30,
    debug_path=None,
):
    """Collect expert demonstrations for all tasks."""

    # Create saving directory
    os.makedirs(os.path.join(saving_path, "training"), exist_ok=True)
    if debug_path:
        os.makedirs(debug_path, exist_ok=True)

    # Initialize environment
    env_cfg = {"num_envs": 1, "max_episode_length": 500, "task_mode": "random"}
    env = Franka3BlocksEnvPyBullet(env_cfg, headless=True)

    # Initialize annotation dictionary
    auto_lang_ann = {
        "info": {"indx": [], "length": []},
        "language": {"ann": [], "task": []},
    }

    stats = {
        "act_max_bound": -100 * np.ones(shape=(7,)),
        "act_min_bound": 100 * np.ones(shape=(7,)),
    }

    neutral_pos = torch.tensor([0.5, 0.0, 0.4])

    # Compute starting index based on existing files
    existing_files = [
        f
        for f in os.listdir(saving_path)
        if f.startswith("episode_") and f.endswith(".npz")
    ]
    start_idx = len(existing_files)

    # Collect demonstrations for each task
    for task_id in range(len(TASK_NAMES)):
        num_success = 0
        num_failures = 0

        print(f"\nCollecting demonstrations for task: {TASK_NAMES[task_id]}")
        if debug_path:
            os.makedirs(os.path.join(debug_path, TASK_NAMES[task_id]), exist_ok=True)

        for episode in range(num_trials_per_task):
            if task_id == 3:
                # Chose bottom and up cube
                (up_cube, bottom_cube) = random.sample(["A", "B", "C"], 2)
                expert_pol = lambda x, y: env.expert_stack(  # noqa: E731
                    x, y, up_cube=up_cube, bottom_cube=bottom_cube
                )
                reward_fn = lambda x, y: env.compute_reward_stack(  # noqa: E731
                    x, y, up_cube=up_cube, bottom_cube=bottom_cube
                )
            else:
                expert_pol = env.get_expert_policy(task_id=task_id)
                reward_fn = env.get_reward_function(task_id)
            frames = []
            success = False
            imgs = []

            frame_idx = (
                len(
                    [
                        f
                        for f in os.listdir(saving_path + "/training")
                        if f.startswith("episode_") and f.endswith(".npz")
                    ]
                )
                + start_idx
            )

            # Reset environment
            obs = env.reset(task_id=task_id)
            init_obs = obs.copy()

            count_done = 0

            for step in range(max_steps):
                # Check if task is complete
                if (count_done > 5) and step > 50:
                    # Return to neutral position
                    actions = env._go_to(obs["state"], neutral_pos, gripper_cmd=1)
                    success = True
                else:
                    # Execute expert policy
                    actions = expert_pol(init_obs, obs)

                # Store frame data
                frame = {
                    "actions": actions.cpu().numpy(),
                    "states": obs["state"].cpu().numpy(),
                    "rgb_static": obs["pixels"][0].permute(1, 2, 0).cpu().numpy(),
                }
                frames.append(frame)
                stats["act_max_bound"] = np.maximum(
                    stats["act_max_bound"], actions.cpu().numpy()
                )

                stats["act_min_bound"] = np.minimum(
                    stats["act_min_bound"], actions.cpu().numpy()
                )

                # Step environment
                obs, _, dones, info = env.step(actions)
                if debug_path:
                    img = obs["pixels"][0].permute(1, 2, 0).cpu().numpy()
                    pil_img = Image.fromarray((img * 255).astype(np.uint8))
                    imgs.append(pil_img)

                # Check reward
                rewards = reward_fn(init_obs, obs)
                if rewards > 0.5:
                    count_done += 1

            # Save successful episodes
            if task_id != 3:
                save_condition = success and (count_done > 3)
            else:
                save_condition = success and (rewards > 0.5) and (count_done > 3)
            if save_condition:
                # Add to database
                if num_success < target_num_episode:
                    idx = frame_idx
                    for frame in frames:
                        # Save the frame
                        frame_path = os.path.join(
                            saving_path, f"training/episode_{idx:07d}.npz"
                        )
                        np.savez(frame_path, **frame)
                        idx += 1

                    start = frame_idx
                    end = idx - 1

                    # Store annotations
                    auto_lang_ann["info"]["indx"].append((start, end))
                    auto_lang_ann["info"]["length"].append(len(frames))
                    auto_lang_ann["language"]["ann"].append(
                        TASK_INSTRUCTIONS[TASK_NAMES[task_id]]
                    )
                    auto_lang_ann["language"]["task"].append(TASK_NAMES[task_id])

                    print(f"  Episode {episode + 1}/{num_trials_per_task}: Success")

                    if debug_path:
                        imgs[0].save(
                            os.path.join(
                                debug_path,
                                TASK_NAMES[task_id],
                                f"success_{num_success}.gif",
                            ),
                            format="GIF",
                            append_images=imgs[1:],
                            save_all=True,
                            duration=100,  # ms per frame
                            loop=0,
                        )
                num_success += 1

            else:
                print(f"  Episode {episode + 1}/{num_trials_per_task}: Failed")
                num_failures += 1
                imgs[0].save(
                    os.path.join(
                        debug_path, TASK_NAMES[task_id], f"failure_{num_failures}.gif"
                    ),
                    format="GIF",
                    append_images=imgs[1:],
                    save_all=True,
                    duration=100,  # ms per frame
                    loop=0,
                )
            if num_success >= target_num_episode:
                break

        print(
            f"Task {TASK_NAMES[task_id]}: {num_success}/{num_trials_per_task} successes"
        )

    # Save annotations
    ann_path = os.path.join(saving_path, "training/auto_lang_ann.npy")
    np.save(ann_path, auto_lang_ann)

    # Save stats
    stats_path = os.path.join(saving_path, "training/statistics.yaml")
    stats = {
        "act_min_bound": stats["act_min_bound"].tolist(),
        "act_max_bound": stats["act_max_bound"].tolist(),
    }
    OmegaConf.save(stats, stats_path)

    print(f"\nAnnotations saved to {ann_path}")
    print(f"Total episodes collected: {len(auto_lang_ann['info']['indx'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect expert demonstrations for Franka 3-cube environment"
    )
    parser.add_argument(
        "--saving_path",
        type=str,
        default="data/franka3b_env_demos",
        help="Path to save collected demonstrations",
    )

    parser.add_argument(
        "--debug_path",
        type=str,
        default="data/debug_franka3b_env_demos",
        help="Path to visualise collected demonstrations",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Max trials to collect episodes per task",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=2,
        help="Number of episodes to collect per task",
    )
    parser.add_argument(
        "--max_steps", type=int, default=64, help="Maximum steps per episode"
    )

    args = parser.parse_args()

    assert args.num_trials >= args.num_episodes, (
        "num_trials must be greater than or equal to num_episodes"
    )

    collect_demonstrations(
        saving_path=args.saving_path,
        num_trials_per_task=args.num_trials,
        max_steps=args.max_steps,
        debug_path=args.debug_path,
        target_num_episode=args.num_episodes,
    )


if __name__ == "__main__":
    main()
