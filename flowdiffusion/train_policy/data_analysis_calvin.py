import argparse
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

import torch
from omegaconf import DictConfig, OmegaConf

sys.path.append(
    os.path.join(
        root_path,
        "calvin/calvin_models",
    )
)

from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,  # noqa: E402
)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Total GPUs available: {torch.cuda.device_count()}")


def main(args):
    results_folder = args.results_folder
    data_path = args.data_path

    if args.train_on == "lang":
        dataset_name = "lang_dataset"
        dataset_key = "lang"
    elif args.train_on == "vis":
        dataset_name = "vis_dataset"
        dataset_key = "vis"
    else:
        raise ValueError(f"Unknown dataset name {args.train_on}")
    print(f"Training on {dataset_name} dataset")

    goal = args.goal if args.goal == "pixel" else f"{args.goal}_{args.feat_patch_size}"
    obs = args.obs if args.obs == "pixel" else f"{args.obs}_{args.feat_patch_size}"

    cfg = DictConfig(
        {
            "root": data_path,
            "datamodule": {
                dataset_name: {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskActionDataset",
                    "key": dataset_key,
                    "save_format": "npz",
                    "batch_size": args.batch_size,
                    "min_window_size": 32,
                    "max_window_size": 65,
                    "proprio_state": {
                        "n_state_obs": 8,
                        "keep_indices": [[0, 7], [14, 15]],
                        "robot_orientation_idx": [3, 6],
                        "normalize": True,
                        "normalize_robot_orientation": True,
                    },
                    "obs_space": {
                        "rgb_obs": ["rgb_static", "rgb_gripper"]
                        if args.use_gripper
                        else ["rgb_static"],
                        "depth_obs": (
                            ["depth_static"]
                            if (args.use_depth and not args.use_gripper)
                            else ["depth_static", "depth_gripper"]
                            if (args.use_depth and args.use_gripper)
                            else []
                        ),
                        "state_obs": ["robot_obs"],
                        "actions": ["actions"],
                        "language": ["language"],
                    },
                    "num_subgoals": args.num_subgoals,
                    "pad": True,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                    "norm_feat": args.norm,
                    "prob_aug": args.data_aug_prob,
                    "feat_patch_size": args.feat_patch_size,
                    "without_guidance": args.without_guidance,
                    "goal": goal,
                    "obs": obs,
                },
            },
            "training_steps": args.training_steps,  # In gradient steps
            "save_every": 100,  # In gradient steps
            "use_text": args.use_text,
            "text_encoder": args.text_encoder,
        }
    )

    print("Config:\n" + OmegaConf.to_yaml(cfg))

    stats_path = os.path.join(data_path, "training/statistics.yaml")
    train_stats = OmegaConf.load(stats_path)

    train_stats_dict = {
        "action": {
            "max": torch.Tensor(train_stats.act_max_bound),
            "min": torch.Tensor(train_stats.act_min_bound),
        }
    }

    transforms = OmegaConf.load(
        os.path.join(
            root_path,
            "calvin/calvin_models/conf/datamodule/transforms/play_basic.yaml",
        )
    )

    data_module = CalvinDataModule(
        cfg.datamodule, transforms=transforms, root_data_dir=cfg.root
    )

    data_module.setup()

    train_set = data_module.train_datasets[dataset_key]
    valid_set = data_module.val_datasets[dataset_key]

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

    import numpy as np
    from tqdm import tqdm

    all_actions = []
    for i in tqdm(range(len(train_set))):
        sample = train_set[i]
        action = sample["action"]
        # project action in [0,1]
        action = (action - train_stats_dict["action"]["min"]) / (
            train_stats_dict["action"]["max"] - train_stats_dict["action"]["min"]
        )
        # project in [-1, 1]
        action = 2 * action - 1
        all_actions.append(action)

    # Convert to numpy array
    all_actions = np.array(all_actions)

    all_actions = all_actions.reshape(-1, all_actions.shape[-1])  # Ensure 2D shape

    print(f"Shape of all actions: {all_actions.shape}")
    print(f"Mean of all actions: {np.mean(all_actions, axis=0)}")
    print(f"Std of all actions: {np.std(all_actions, axis=0)}")
    print(f"Min of all actions: {np.min(all_actions, axis=0)}")
    print(f"Max of all actions: {np.max(all_actions, axis=0)}")

    # Count the number of actions where the 3 first dimension are < 1e-5
    num_zero_actions = np.sum(np.all(np.abs(all_actions[:, :3]) < 1e-5, axis=1))
    print(f"Number of actions with first 3 dimensions < 1e-5: {num_zero_actions}")

    # Plot the distribution of actions over each dimension
    import matplotlib.pyplot as plt

    plt.figure(figsize=(2 * all_actions.shape[1], 5))
    for i in range(all_actions.shape[1]):
        plt.subplot(1, all_actions.shape[1], i + 1)
        plt.hist(all_actions[:, i], bins=50, alpha=0.7)
        plt.title(f"Action Dimension {i}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("action_distribution_calvin.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--server", type=str, default="hacienda"
    )  # set to 'jz' to run on jean zay server
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument(
        "-c", "--checkpoint_num", type=int, default=None
    )  # set to checkpoint number to resume training or generate samples
    parser.add_argument(
        "--training_steps", type=int, default=500000
    )  # set to number of training steps
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset",
    )  # set to path to dataset
    parser.add_argument(
        "-r", "--results_folder", type=str, default="../results_policy_single/calvin"
    )  # set to path to results folder
    parser.add_argument(
        "--num_subgoals", type=int, default=8
    )  # set to number of subgoals
    parser.add_argument(
        "--train_on", type=str, default="lang"
    )  # set to train on language labelled dataset (38% "lang") or full dataset (100% "vis")
    parser.add_argument(
        "--data_aug_prob", type=float, default=0.0
    )  # set to probability of data augmentation (0.0 for no augmentation)
    parser.add_argument(
        "--use_depth", action="store_true"
    )  # set to True to use depth observations
    parser.add_argument(
        "--use_gripper", action="store_true"
    )  # set to True to use gripper observations
    parser.add_argument(
        "--goal",
        type=str,
        default="pixel",
        choices=["pixel", "dino_vit", "dino", "r3m"],
    )  # set to goal type for diffusion (pixel, dino_vit, dino, r3m)
    parser.add_argument(
        "--obs",
        type=str,
        default="pixel",
        choices=["pixel", "dino_vit", "dino", "r3m"],
    )  # set to observation type for diffusion (pixel, dino_vit, dino, r3m)
    parser.add_argument(
        "--feat_patch_size", type=int, default=16
    )  # set to feature patch size for dino features
    parser.add_argument(
        "--norm", type=str, default=None, choices=[None, "l2", "z_score", "min_max"]
    )  # set to normalisation type for features
    parser.add_argument(
        "--batch_size", type=int, default=32
    )  # set to batch size for training
    parser.add_argument(
        "--use_text", action="store_true"
    )  # set to True to use text observations (language annotations)
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="CLIP",
        choices=["CLIP", "Flan-t5", "Siglip"],
    )  # set to text encoder to use
    parser.add_argument("--without_guidance", action="store_true")
    # set to True to train without guidance (i.e. no target conditioning)
    args = parser.parse_args()
    main(args)
