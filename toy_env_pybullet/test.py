import os
import random

import numpy as np
import torch
from PIL import Image
from toyEnv import Franka3CubeEnvPyBullet

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


"""Collect expert demonstrations for all tasks."""

# Initialize environment
cfg = {"num_envs": 1, "max_episode_length": 500, "task_mode": "random"}
env = Franka3CubeEnvPyBullet(cfg, headless=True)

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

task_id = 0

for episode in range(1):
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

    # Reset environment
    obs = env.reset(task_id=task_id)
    init_obs = obs.copy()

    count_done = 0

    for step in range(64):
        # Check if task is complete
        if count_done > 5:
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

        # Step environment
        obs, _, dones, info = env.step(actions)
        img = obs["pixels"][0].permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        imgs.append(pil_img)

        # Check reward
        rewards = reward_fn(init_obs, obs)
        if rewards > 0.5:
            count_done += 1

    imgs[0].save(
        os.path.join("debug_toy_env", TASK_NAMES[task_id], "example.gif"),
        format="GIF",
        append_images=imgs[1:],
        save_all=True,
        duration=100,  # ms per frame
        loop=0,
    )
