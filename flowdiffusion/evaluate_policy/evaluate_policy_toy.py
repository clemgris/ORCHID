import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
        str(ROOT_PATH / "toy_env_pybullet"),
    ]
)

from flowdiffusion.evaluate_policy.methods.evaluate_policy import (
    evaluate_policy_singlestep_toy,
)
from model.hierarchical_model_toy import HierarchicalModel
from toy_env_pybullet.toyEnv import Franka3CubeEnvPyBullet

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # seed_everything(0, workers=True)
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )

    parser.add_argument(
        "--policy_checkpoint_num",
        type=str,
        help="Policy checkpoint num",
        default=1406,
    )

    parser.add_argument(
        "--policy_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/debug_toy_LL",
    )

    parser.add_argument(
        "--high_level_checkpoint_num",
        type=str,
        help="High level checkpoint number",
        default=2,
    )

    parser.add_argument(
        "--high_level_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/debug_toy_HL",
    )

    parser.add_argument(
        "--high_level_sampling_timesteps",
        type=int,
        help="Number of sampling timesteps for high level model",
        default=100,
    )

    parser.add_argument(
        "--high_level_ddim_eta",
        type=float,
        help="DDIM eta for high level model",
        default=0.0,
    )

    parser.add_argument(
        "--server",
        "-s",
        type=str,
        help="Server",
        default="hacienda",
    )

    parser.add_argument(
        "--use_oracle_subgoals",
        action="store_true",
        help="Use oracle subgoals",
    )

    parser.add_argument(
        "--num_subgoals",
        type=int,
        default=8,
        help="Number of subgoals to generate.",
    )

    parser.add_argument(
        "--debug_path",
        type=str,
        default=None,
        help="Path to save debug images.",
    )

    parser.add_argument(
        "--eval_folder",
        type=str,
        default="eval",
        help="Folder to save evaluation results.",
    )

    parser.add_argument(
        "--replan",
        action="store_true",
        help="Replan subgoals every 64 steps.",
    )

    parser.add_argument(
        "--use_filtered_data",
        action="store_true",
        help="Use filtered data (expert sucesses) for evaluation.",
    )

    parser.add_argument(
        "--policy_model",
        type=str,
        default="diffusion",
        choices=["diffusion", "act"],
        help="Policy model to use.",
    )

    parser.add_argument(
        "--high_level_guidance",
        type=int,
        default=3,
        help="Guidance in high-level diffusion.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()
    args.save_failures = args.debug_path is not None

    # Set seed
    seed_everything(args.seed, workers=True)

    # Load data config
    policy_data_config = OmegaConf.load(
        os.path.join(args.policy_results_folder, "data_config.yaml")
    )
    if not args.use_oracle_subgoals:
        high_level_data_config = OmegaConf.load(
            os.path.join(args.high_level_results_folder, "data_config.yaml")
        )
    else:
        high_level_data_config = {}

    config = DictConfig(
        {
            "policy": {
                "model": args.policy_model,
                "checkpoint_num": args.policy_checkpoint_num,
                "results_folder": args.policy_results_folder,
                **policy_data_config,
            },
            "high_level": {
                "checkpoint_num": args.high_level_checkpoint_num,
                "results_folder": args.high_level_results_folder,
                "use_oracle_subgoals": args.use_oracle_subgoals,
                **high_level_data_config,
                "sampling_timesteps": args.high_level_sampling_timesteps,
                "ddim_sampling_eta": args.high_level_ddim_eta,
                "guidance": args.high_level_guidance,
            },
            "debug_path": args.debug_path,
            "server": args.server,
            "num_subgoals": args.num_subgoals,
            "replan": args.replan,
        }
    )

    print("Config:\n" + OmegaConf.to_yaml(config))

    # Save config
    os.makedirs(args.eval_folder, exist_ok=True)
    with open(
        os.path.join(args.eval_folder, "config.yaml"),
        "w",
    ) as f:
        OmegaConf.save(config, f)

    if args.use_oracle_subgoals:
        print("Using oracle subgoals")
    else:
        print("Using generated subgoals")

    device = torch.device("cuda:0")
    config.device = "cuda"

    if args.debug_path:
        # Create debug folder
        debug_path = Path(config.debug_path)
        os.makedirs(debug_path, exist_ok=True)

    env_cfg = {"num_envs": 1, "max_episode_length": 500, "task_mode": "random"}
    env = Franka3CubeEnvPyBullet(env_cfg, headless=True)
    model = HierarchicalModel(config)

    start_time = time.time()

    evaluate_policy_singlestep_toy(model, env, args)

    elapsed_time = time.time() - start_time
    print("Evaluation time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
