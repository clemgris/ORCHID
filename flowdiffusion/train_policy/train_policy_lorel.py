import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    SiglipTextModel,
    SiglipTokenizer,
    T5EncoderModel,
)

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

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
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/TrajectoryDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_50k/data_with_dino_vit_features"
    else:
        data_path = "/home/grislain/SkillDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_1k/training/data_with_dino_vit_features"

    cfg = DictConfig(
        {
            "root": data_path,
            "skip_frames": 4,
            "diffuse_on": "pixel",
            "num_data": args.num_data,
            "save_every": args.save_every,
        },
    )

    results_folder = Path(results_folder)

    if os.path.exists(results_folder):
        if not args.override:
            raise ValueError(
                f"Results folder {results_folder} already exists. Use --override to overwrite."
            )
    results_folder.mkdir(exist_ok=True, parents=True)

    with open(os.path.join(results_folder, "data_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    # Training set
    train_set = ExpertActionDataset(
        cfg.root, skip_frames=cfg.skip_frames, diffuse_on=cfg.diffuse_on
    )

    stats_path = os.path.join(data_path, "../dataset_stats.pkl")
    train_stats = pickle.load(open(stats_path, "rb"))

    cfg["stats_path"] = stats_path

    training_steps = args.training_steps
    device = torch.device("cuda")
    log_freq = 10

    goal = args.goal if args.goal == "pixel" else f"{args.goal}_{args.feat_patch_size}"
    obs = args.obs if args.obs == "pixel" else f"{args.obs}_{args.feat_patch_size}"

    n_channels = 3

    # Observation representation shape
    if obs == "pixel":
        obs_shape = [n_channels, 64, 64]
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
        goal_shape = [n_channels, 64, 64]
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

    # Diffusion config
    diff_cfg = DictConfig(
        {
            "n_obs_steps": 1,
            "horizon": cfg.skip_frames,
            "input_shapes": {
                "observation.state": [0],
            },
            "output_shapes": {
                "action": [5],
            },
            "n_action_steps": cfg.skip_frames,
            "input_normalization_modes": {},
            "output_normalization_modes": {"action": "min_max"},
            "crop_shape": None,
            "vision_backbone": "resnet18",
            "use_text": args.use_text,
            "text_embed_dim": text_embed_dim,
            "final_text_embed_dim": 64,
            "down_dims": (512, 1024),
        }
    )

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

    # Save cfg
    with open(os.path.join(results_folder, "data_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    policy = DiffusionPolicy(diff_cfg, dataset_stats=train_stats)
    policy.train()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        train_set,
        num_workers=4 if args.server == "jz" else 1,
        batch_size=args.batch_size if hasattr(args, "batch_size") else 32,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
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
            batch = {
                k: v.to(device, non_blocking=True, dtype=torch.float32)
                for k, v in batch.items()
            }
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
        "--num_data", type=int, default=100
    )  # set to number of data points to use for training
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
    args = parser.parse_args()
    main(args)
