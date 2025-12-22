import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

import torch
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    SiglipTextModel,
    SiglipTokenizer,
    T5EncoderModel,
)

sys.path.append(
    os.path.join(
        root_path,
        "toy_env_pybullet/dataset",
    )
)

from toy_env_pybullet.dataset.dataset import ToyActionDataset

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Total GPUs available: {torch.cuda.device_count()}")


def main(args):
    results_folder = args.results_folder
    data_paths = args.data_paths

    print("Training on ", len(data_paths), " datasets.")

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
            "root": data_paths,
            "dataset": {
                "batch_size": args.batch_size,
                "min_window_size": 32,
                "max_window_size": 65,
                "obs_space": {
                    "rgb_static": ["rgb_static"],
                    "states": ["states"],
                    "actions": ["actions"],
                    "language": ["language"],
                },
                "num_subgoals": args.num_subgoals,
                "pad": True,
                "lang_folder": "lang_annotations",
                "auto_lang_name": "auto_lang_ann",
                "goal": goal,
                "obs": obs,
                "norm_feat": args.norm,
            },
            "training_steps": args.training_steps,  # In gradient steps
            "save_every": 100,  # In gradient steps
            "text_encoder": args.text_encoder,
            "use_text": args.use_text,
        }
    )

    print("Config:\n" + OmegaConf.to_yaml(cfg))

    n_channels = 4 if args.use_depth else 3

    transforms = OmegaConf.load(
        os.path.join(
            root_path,
            "toy_env_pybullet/conf/transforms.yaml",
        )
    )

    cfg.dataset.transforms = transforms

    results_folder = Path(results_folder)

    if os.path.exists(results_folder):
        if not args.override:
            raise ValueError(
                f"Results folder {results_folder} already exists. Use --override to overwrite."
            )
    results_folder.mkdir(exist_ok=True, parents=True)
    print("Results folder:", results_folder)

    train_sets = []
    for path in data_paths:
        train_set = ToyActionDataset(path, cfg.dataset)
        train_sets.append(train_set)
    train_set = torch.utils.data.ConcatDataset(train_sets)

    print("Train data:", len(train_set))
    print("Valid data:", 0)

    training_steps = cfg.training_steps
    device = torch.device("cuda")
    log_freq = 10

    # Observation representation shape
    if obs == "pixel":
        obs_shape = [n_channels, 96, 96]
    elif "dino" in obs:
        assert args.feat_patch_size % 16 == 0, (
            f"Feature patch size {args.feat_patch_size} must be a multiple of 16 for DINO features."
        )
        obs_shape = [768, args.feat_patch_size, args.feat_patch_size]
    elif "r3m" in obs:
        assert args.feat_patch_size % 7 == 0, (
            f"Feature patch size {args.feat_patch_size} must be a multiple of 7 for R3M features."
        )
        obs_shape = [512, args.feat_patch_size, args.feat_patch_size]

    # Goal representation shape
    if goal == "pixel":
        goal_shape = [n_channels, 96, 96]
    elif "dino" in goal:
        assert args.feat_patch_size % 16 == 0, (
            f"Feature patch size {args.feat_patch_size} must be a multiple of 16 for DINO features."
        )
        goal_shape = [768, args.feat_patch_size, args.feat_patch_size]
    elif "r3m" in goal:
        assert args.feat_patch_size % 7 == 0, (
            f"Feature patch size {args.feat_patch_size} must be a multiple of 7 for R3M features."
        )
        goal_shape = [512, args.feat_patch_size, args.feat_patch_size]

    # Text encoder
    if args.text_encoder == "CLIP":
        if args.server == "jz":
            text_pretrained_model = (
                "/lustre/fsmisc/dataset/HuggingFace_Models/openai/clip-vit-base-patch32"
            )
        else:
            text_pretrained_model = "openai/clip-vit-base-patch32"

        tokenizer = CLIPTokenizer.from_pretrained(text_pretrained_model)
        text_encoder = CLIPTextModel.from_pretrained(text_pretrained_model)
        text_embed_dim = 512

    elif args.text_encoder == "Flan-t5":
        if args.server == "jz":
            text_pretrained_model = (
                "/lustre/fsmisc/dataset/HuggingFace_Models/google/flan-t5-base"
            )
        else:
            text_pretrained_model = "google/flan-t5-base"
        text_encoder = T5EncoderModel.from_pretrained(text_pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model)
        text_embed_dim = 768

    elif args.text_encoder == "Siglip":
        if args.server == "jz":
            text_pretrained_model = "/lustre/fsn1/projects/rech/fch/uxv44vw/models/google/siglip-base-patch16-224"
        else:
            text_pretrained_model = "google/siglip-base-patch16-224"
        tokenizer = SiglipTokenizer.from_pretrained(text_pretrained_model)
        text_encoder = SiglipTextModel.from_pretrained(text_pretrained_model)
        text_embed_dim = 768

    text_encoder = text_encoder.to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    text_encoder_num_params = sum(p.numel() for p in text_encoder.parameters())
    print(
        f"Number of parameters in text encoder {text_pretrained_model}: {text_encoder_num_params / 1e6:.2f}M"
    )

    # Diffusion Policy
    diff_cfg = DictConfig(
        {
            "n_obs_steps": 1,
            "horizon": int(
                np.ceil(cfg.dataset["max_window_size"] / cfg.dataset["num_subgoals"])
            )
            - 1,
            "input_shapes": {
                "observation.state": [0],
            },
            "output_shapes": {
                "action": [7],
            },
            "n_action_steps": 8,
            "input_normalization_modes": {},
            "output_normalization_modes": {"action": "min_max"},
            "crop_shape": None,
            "vision_backbone": "resnet18",
            "use_text": args.use_text,
            "text_embed_dim": text_embed_dim,
            "final_text_embed_dim": 64,
        }
    )

    if args.pretrained_encoder:
        diff_cfg["pretrained_backbone_weights"] = args.pretrained_encoder
        diff_cfg["use_group_norm"] = False

    if obs == "pixel":
        diff_cfg["input_shapes"]["observation.image_static"] = obs_shape
        if args.use_gripper:
            diff_cfg["input_shapes"]["observation.image_gripper"] = obs_shape
    else:
        diff_cfg["input_shapes"]["observation.feat_static"] = obs_shape
        if args.use_gripper:
            raise NotImplementedError(
                "Gripper features are not implemented for diffusion policy."
            )

    if goal == "pixel":
        diff_cfg["input_shapes"]["observation.image_goal_static"] = goal_shape
        if args.use_gripper:
            diff_cfg["input_shapes"]["observation.image_goal_gripper"] = goal_shape
    else:
        diff_cfg["input_shapes"]["observation.feat_goal_static"] = goal_shape
        if args.use_gripper:
            raise NotImplementedError(
                "Gripper features are not implemented for diffusion policy."
            )

    diff_cfg = DiffusionConfig(**diff_cfg)
    cfg["diff_cfg"] = diff_cfg

    # Load training statistics
    stats_path = os.path.join(data_paths[0], "statistics.yaml")
    train_stats = OmegaConf.load(stats_path)

    train_stats_dict = {
        "action": {
            "max": torch.Tensor(train_stats.act_max_bound),
            "min": torch.Tensor(train_stats.act_min_bound),
        }
    }

    cfg["stats_path"] = stats_path
    # Save cfg
    if args.checkpoint_num is not None:
        # Load checkpoint config which is a yaml
        checkpoint_cfg_path = os.path.join(results_folder, "data_config.yaml")
        checkpoint_cfg = OmegaConf.load(checkpoint_cfg_path)

        # Check if cfg and checkpoint_cfg align
        mismatching_keys = []
        allowed_mismatch = ["training_steps"]  # Allow mismatch for training steps
        for key in cfg.keys():
            if key not in checkpoint_cfg:
                mismatching_keys.append(key)
                print(f"Key {key} not in checkpoint config.")
            elif cfg[key] != checkpoint_cfg[key]:
                mismatching_keys.append(key)
                print(
                    f"Key {key} has different value in checkpoint config {checkpoint_cfg[key]} != {cfg[key]}"
                )
        assert all(key in allowed_mismatch for key in mismatching_keys), (
            f"Keys {mismatching_keys} are not in the allowed mismatch list {allowed_mismatch}"
        )
    else:
        with open(os.path.join(results_folder, "data_config.yaml"), "w") as file:
            file.write(OmegaConf.to_yaml(cfg))

    policy = DiffusionPolicy(diff_cfg, dataset_stats=train_stats_dict)

    if args.checkpoint_num is not None:
        checkpoint_path = os.path.join(
            results_folder, f"model-{args.checkpoint_num}.pt"
        )
        print(f"Loading checkpoint from {checkpoint_path}")
        policy.load_state_dict(torch.load(checkpoint_path))

    policy.train()
    policy.to(device)

    # Training parameters
    print("Number of training parameters:", sum(p.numel() for p in policy.parameters()))

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        train_set,
        num_workers=4 if args.server == "hacienda" else 8,
        batch_size=cfg.dataset["batch_size"],
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Run training loop.
    done = False
    step = (
        args.checkpoint_num * cfg.save_every if (args.checkpoint_num is not None) else 0
    )
    print(f"Starting training at step {step}")
    pbar = tqdm(total=training_steps, initial=step, desc="Training")

    while not done:
        for batch in dataloader:
            batch_text = batch.get("text", None)
            if args.use_text:
                batch_text_ids = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(device)
                batch["text"] = text_encoder(**batch_text_ids).last_hidden_state
            else:
                del batch["text"]
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")

            if step % cfg.save_every == 0:
                # Delete previous checkpoints
                if step != args.checkpoint_num:
                    past_saving_path = os.path.join(
                        results_folder, f"model-{step // cfg.save_every - 2}.pt"
                    )
                    if os.path.exists(past_saving_path):
                        if step // cfg.save_every - 2 != args.checkpoint_num:
                            os.remove(past_saving_path)
                    # Save model
                    saving_path = os.path.join(
                        results_folder, f"model-{step // cfg.save_every}.pt"
                    )
                    torch.save(policy.state_dict(), saving_path)
            pbar.set_postfix(loss=loss.item())
            step += 1
            pbar.update(1)
            if step >= training_steps:
                done = True
                break
    pbar.close()


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
        "--data_paths",
        type=str,
        nargs="+",
        default=["/home/grislain/AVDC/data/toy_env_demos/training"],
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
    parser.add_argument("--pretrained_encoder", type=str, default=None)
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
