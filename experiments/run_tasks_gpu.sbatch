#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a40:1
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/gpu_%A_%a.out
# Runs line SLURM_ARRAY_TASK_ID+1 of the task file given as $1, on a frozen copy of the code, on one GPU.
cd /coc/flash10/ksingara3/compositionality-theory/experiments
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
FROZEN=$(mktemp -d); cp *.py $FROZEN/
export OMP_NUM_THREADS=8 PYTHONPATH=$FROZEN HF_HUB_OFFLINE=1
echo "$LINE"; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
~/flash/miniconda3/envs/chunks/bin/python $FROZEN/$LINE
