import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from lorel.expert_dataset import ExpertActionDataset, ExpertDataset  # noqa: E402, F401

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Total GPUs available: {torch.cuda.device_count()}")


def main(args):
    results_folder = args.results_folder

    if args.server == "jz":
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/TrajectoryDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_50k/dec_24_sawyer_50k/training/data_with_dino_vit_features"
    else:
        data_path = "/home/grislain/SkillDiffuser/lorel/data/jul_26_sawyer_1k/jul_26_sawyer_1k/training/data_with_dino_vit_features"

    cfg = DictConfig(
        {
            "root": data_path,
            "skip_frames": 4,
            "diffuse_on": "pixel",
            "save_every": args.save_every,
            "evaluate_every": 1000,  # Evaluate every 1000 steps
        },
    )

    # Training set
    train_set = ExpertActionDataset(
        cfg.root, skip_frames=cfg.skip_frames, diffuse_on=cfg.diffuse_on
    )
    # Validation test
    val_set = ExpertActionDataset(
        cfg.root.replace("training", "validation"),
        skip_frames=cfg.skip_frames,
        diffuse_on=cfg.diffuse_on,
    )

    print(f"Number of training samples: {len(train_set)}")
    print(f"Number of validation samples: {len(val_set)}")

    all_actions = []
    for i in tqdm(range(len(train_set))):
        sample = train_set[i]
        action = sample["action"]
        all_actions.append(action)

    all_actions = np.array(all_actions)
    all_actions = all_actions.reshape(-1, all_actions.shape[-1])  # Ensure 2D shape

    print(f"Shape of all actions: {all_actions.shape}")
    print(f"Mean of all actions: {np.mean(all_actions, axis=0)}")
    print(f"Std of all actions: {np.std(all_actions, axis=0)}")
    print(f"Min of all actions: {np.min(all_actions, axis=0)}")
    print(f"Max of all actions: {np.max(all_actions, axis=0)}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for i in range(all_actions.shape[1]):
        plt.subplot(1, all_actions.shape[1], i + 1)
        plt.hist(all_actions[:, i], bins=50, alpha=0.7)
        plt.title(f"Action Dimension {i}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("action_distribution_lorel.png")
    plt.close()

    # Count the number of actions where the 3 first dimension are < 1e-5
    num_zero_actions = np.sum(np.all(np.abs(all_actions[:, :3]) < 1e-5, axis=1))
    print(
        f"Number of actions with first 3 dimensions < 1e-5: {num_zero_actions} / {len(all_actions)}"
    )

    # filter actions where the first 3 dimensions are < 1e-5
    filtered_actions = all_actions[np.any(np.abs(all_actions[:, :3]) >= 1e-5, axis=1)]
    print("_" * 50)

    # print stats for filtered actions
    print(f"Shape of filtered actions: {filtered_actions.shape}")
    print(f"Mean of filtered actions: {np.mean(filtered_actions, axis=0)}")
    print(f"Std of filtered actions: {np.std(filtered_actions, axis=0)}")
    print(f"Min of filtered actions: {np.min(filtered_actions, axis=0)}")
    print(f"Max of filtered actions: {np.max(filtered_actions, axis=0)}")

    # Plot the distribution of actions over each dimension
    filtered_actions = filtered_actions.reshape(
        -1, filtered_actions.shape[-1]
    )  # Ensure 2D shape

    plt.figure(figsize=(10, 6))
    for i in range(filtered_actions.shape[1]):
        plt.subplot(1, filtered_actions.shape[1], i + 1)
        plt.hist(filtered_actions[:, i], bins=50, alpha=0.7)
        plt.title(f"Filtered Action Dimension {i}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("filtered_action_distribution_lorel.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results_folder",
        type=str,
        default="results/train_policy_lorel",
    )  # set to results folder to save the model and logs
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
        "--text_encoder",
        type=str,
        default="CLIP",
        choices=["CLIP", "Flan-t5", "Siglip"],
    )  # set to text encoder to use
    parser.add_argument(
        "--use_text", action="store_true"
    )  # set to use text embeddings in the policy
    parser.add_argument(
        "--goal",
        type=str,
        default="pixel",
        choices=["pixel", "dino", "r3m"],
    )  # set to goal representation
    parser.add_argument(
        "--obs",
        type=str,
        default="pixel",
        choices=["pixel", "dino", "r3m"],
    )  # set to observation representation
    parser.add_argument(
        "--feat_patch_size",
        type=int,
        default=224,
    )  # set to feature patch size for DINO/R3M features
    parser.add_argument(
        "--use_gripper", action="store_true"
    )  # set to use gripper images in the policy
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
    )  # set to save the model every n steps
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )  # set to batch size for training
    args = parser.parse_args()
    main(args)
