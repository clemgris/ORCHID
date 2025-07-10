#!/bin/bash

#SBATCH --partition=hard
#SBATCH --job-name=collect_data
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

export MUJOCO_GL=egl

BASE_PATH="/home/grislain"

cd $BASE_PATH/SkillDiffuser/lorel

nvidia-smi

# Collect data
python collect_data.py --savepath $BASE_PATH/SkillDiffuser/lorel/data/dec_24_sawyer_50k/ --num_episodes 50000

# Compress data
gzip -k $BASE_PATH/SkillDiffuser/lorel/data/dec_24_sawyer_50k/data.hdf5