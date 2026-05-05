# 🌺 ORCHID

**Fine-tuning Hierarchical Diffusion Policy with Iterative Self-Training**

This repository contains the official code and checkpoints for the paper:

> **Online Self-Training for Co-Adaptation in Hierarchical Diffusion**

---

## Installation 🛠️

#### 1. Create Conda Env
```bash
conda create -n orchid python==3.10
```
#### 2. Install CALVIN

```bash
cd calvin/calvin_env
pip install -e .
cd ../calvin_models
pip install -r requirements.txt
```

#### 3. Install Requirements

```bash
cd orchid/controller
pip install -e .
cd ../../
pip install -r requirements.txt
```

## Evaluation 🎯

**Model checkpoints**

To maintain double-blind integrity, model checkpoints will be made available in the final version of this work.

**CALVIN**

Multi-task language-condition (MTLC) benchmark
```bash
python orchid/evaluate_policy/evaluate_policy_calvin.py \
    --policy_checkpoint_num 9999 \
    --policy_results_folder path_2_LL_iter0 \
    --high_level_checkpoint_num 199 \
    --high_level_results_folder path_2_HL_iter0 \
    --eval_folder path_2_eval_MTLC_iter0 \
    --policy_model diffusion \
    --replan \
    --seed 0
```

Long horizon multi-task language-condition (LH-MTLC) benchmark
```bash
python python orchid/evaluate_policy/evaluate_policy_long_horizon_calvin.py \
    --policy_checkpoint_num 9999 \
    --policy_results_folder path_2_LL_iter0 \
    --high_level_checkpoint_num 199 \
    --high_level_results_folder path_2_HL_iter0 \
    --eval_folder path_2_eval_LHMTLC_iter0 \
    --policy_model diffusion \
    --replan \
    --seed 0
```

* **Franka-3Blocks**
```bash
python orchid/evaluate_policy/evaluate_policy_franka3b.py \
    --policy_checkpoint_num 9999 \
    --policy_results_folder path_2_LL_iter0 \
    --high_level_checkpoint_num 199 \
    --high_level_results_folder path_2_HL_iter0 \
    --eval_folder path_2_eval_iter0 \
    --policy_model diffusion \
    --replan \
    --seed 0
```


## Iterative Fine-Tuning Pipeline 🔄

### A. Train on the Initial Dataset

#### 1. Prepare the Initial Dataset

Download or create an initial expert dataset:

* **CALVIN**: download the `D_D` dataset following the official CALVIN repository instructions (https://github.com/mees/calvin).
* **Franka-3Blocks**: generate an expert dataset using:

```bash
python franka_3blocks_env_pybullet/generate_data.py \
    --saving_path path_2_dataset0 \
    --num_trials 200 \
    --num_episodes 100
```

---

#### 2. Train the High-Level (HL) Policy

**CALVIN**
```bash
accelerate launch orchid/train_planner/train_calvin.py \
    --data_paths path_2_dataset0 \
    --train_num_steps 500000 \
    --batch_size 8 \
    --diff_objective pred_v \
    --text_encoder CLIP \
    --result_folder path_2_HL_iter0
```
**Franka3Blocks**
```bash
accelerate launch orchid/train_planner/train_franka3b.py \
    --data_paths path_2_dataset0 \
    --train_num_steps 500000 \
    --batch_size 8 \
    --diff_objective pred_v \
    --text_encoder CLIP \
    --result_folder path_2_HL_iter0
```

---

#### 3. Train the Low-Level (LL) Policy

Diffusion-based low-level policy (**CALVIN**):

```bash
python orchid/train_policy/train_policy_calvin.py \
    --data_paths path_2_dataset0 \
    --training_steps 1000000 \
    --batch_size 32 \
    --result_folder path_2_LL_iter0
```

ACT-based low-level policy (**CALVIN**):

```bash
python orchid/train/train_policy_calvin_act.py \
    --data_paths path_2_dataset0 \
    --training_steps 500000 \
    --batch_size 128 \
    --result_folder path_2_ACT_LL_iter0
```

Diffusion-based low-level policy (**Franka3Blocks**):

```bash
python orchid/train_policy/train_policy_franka3b.py \
    --data_paths path_2_dataset0 \
    --training_steps 1000000 \
    --batch_size 32 \
    --result_folder path_2_LL_iter0
```

---

### B. Generate New Data

#### 1. Extract Context Buffers

Contexts consist of the **initial environment state** and the **language instruction**.

Two types of contexts are used:

* **Replayed contexts**
* **Reset contexts**

**Replayed contexts**

**CALVIN**
```bash
python orchid/generate_data/save_buffer.py \
    --data_path path_2_dataset0 \
    --policy_checkpoint_num 9999 \
    --policy_results_folder path_2_LL_iter0 \
    --mode "end_all" 
```

**Franka3Blocks**
```bash
python franka_3blocks_env_pybullet/save_buffer.py \
    --data_path path_2_dataset0 \
    --mode "end_all" 
```

**Reset contexts**

**CALVIN**
```bash
python orchid/generate_data/save_buffer.py \
    --data_path path_2_dataset0 \
    --policy_checkpoint_num 9999 \
    --policy_results_folder path_2_LL_iter0 \
    --mode "reset" 
```

**Franka3Blocks**
```bash
python franka_3blocks_env_pybullet/save_buffer.py \
    --data_path path_2_dataset0 \
    --mode "reset" 
```

#### 2. Merge Context Buffers

**CALVIN**
```python
from generate_data.state_buffer import StateBuffer

buffer_path_1 = 'path_2_end_dataset1/training/state_buffer_end_all.pkl'
buffer_path_2 = 'path_2_reset_dataset1/training/state_buffer_end_all.pkl'

buffer1 = StateBuffer(tasks.keys(), max_size=1e6)
buffer1.load(buffer_path_1)

buffer2 = StateBuffer(tasks.keys(), max_size=1e6)
buffer2.load(buffer_path_2)

buffer_merged = StateBuffer(tasks.keys(), max_size=1e6)
buffer_merged.load(buffer_path_1)

for s in buffer2.buffer:
    buffer_merged.add(s)

merged_buffer_save_path = "path_2_dataset1/training/state_buffer_merged_end_all.pkl"
buffer_merged.save(merged_buffer_save_path)
print(f"Saved state buffer to {merged_buffer_save_path}")
```


**Franka3Blocks**
```python
import pickle

buffer_path_1 = 'path_2_end_dataset1/training/state_buffer_end_all.pkl'
buffer_path_2 = 'path_2_reset_dataset1/training/state_buffer_end_all.pkl'

with open(buffer_path_1, "rb") as f:
    data_1 = pickle.load(f)

with open(buffer_path_2, "rb") as f:
    data_2 = pickle.load(f)

data = data_1 + data_2

with open("path_2_dataset1/training/state_buffer_end_reset_all.pkl", "wb") as f:
    pickle.dump(data, f)
```

---

#### 3. Generate New Rollouts

**CALVIN**
```bash
python orchid/generate_data/generate_new_data_calvin.py \
    --policy_checkpoint_num 9999 \
    --policy_results_folder path_2_LL \
    --high_level_checkpoint_num 199 \
    --high_level_results_folder path_2_HL \
    --policy_model diffusion \
    --saving_path path_2_dataset1 \
    --num_data 100 \
    --num_trials 5 \
    --buffer_save_path path_2_merged_buffer
```

**Franka3Blocks**
```bash
python orchid/generate_data/generate_new_data_franka3b.py \
    --policy_checkpoint_num 9999 \
    --policy_results_folder path_2_LL \
    --high_level_checkpoint_num 199 \
    --high_level_results_folder path_2_HL \
    --policy_model diffusion \
    --saving_path path_2_dataset1 \
    --num_data 100 \
    --num_trials 5 \
    --buffer_save_path path_2_merged_buffer
```

---

### C. Dataset Aggregation

* **orchid**: train from scratch on the full aggregated dataset.

**CALVIN**
```bash
accelerate launch orchid/train_planner/train_calvin.py \
    --data_paths path_2_dataset0 path_2_dataset1 \
    --train_num_steps 750000 \
    --batch_size 8 \
    --diff_objective pred_v \
    --text_encoder CLIP \
    --result_folder path_2_HL_iter1
```

```bash
python orchid/train_policy/train_policy_calvin.py \
    --data_paths path_2_dataset0 path_2_dataset1\
    --training_steps 1500000 \
    --batch_size 32 \
    --result_folder path_2_LL_iter1
```

* **orchid-ft**: fine-tune starting from the previous iteration’s policy.

**CALVIN**
```bash
accelerate launch orchid/train_planner/ft_calvin.py \
    --data_paths path_2_dataset1 \
    --train_num_steps 50000 \
    --batch_size 8 \
    --diff_objective pred_v \
    --text_encoder CLIP \
    --result_folder path_2_HL_ft_iter1 \
    --pretrained_results_folder path_2_HL_iter0 \
    --checkpoint_num 199
```

```bash
python orchid/train_policy/ft_policy_calvin.py \
    --data_paths path_2_dataset1\
    --training_steps 200000 \
    --batch_size 32 \
    --result_folder path_2_LL_ft_iter1 \
    --pretrained_results_folder path_2_LL_iter0 \
    --checkpoint_num 9999
```

---

## Acknowledgements 📚

This repository builds upon and is inspired by the following projects:

- **CALVIN**: https://github.com/mees/calvin.git (commit `dd37755`)
- **AVDC**: https://github.com/flow-diffusion/AVDC.git (commit `176fbe1`)
- **LeRobot**: https://github.com/huggingface/lerobot.git (version `0.1.0`)
