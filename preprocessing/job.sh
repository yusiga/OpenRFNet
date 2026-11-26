#!/bin/bash
#SBATCH -J job_ztr # 作业名
#SBATCH -p dgx1 # 提交到队列
#SBATCH -N 1 # 结点数
#SBATCH --ntasks-per-node=1 # 每个节点的进程数
#SBATCH --cpus-per-task=6 # 每个进程占用cpu核心数
#SBATCH -t 48:00:00 # 任务最大运行时间
#SBATCH --gres=gpu:1  # gpu数量

python -u ./create_encoder.py
#python -u ./clean_mat.py
#python -u ./STFT.py