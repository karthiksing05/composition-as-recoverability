#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 8:00:00
#SBATCH --requeue
export HF_HUB_OFFLINE=1
cd /coc/flash10/ksingara3/compositionality-theory/experiments
E19_LAYER=$L ~/flash/miniconda3/envs/chunks/bin/python e19_nonlinear.py --models allenai/OLMo-2-1124-7B --seeds $SLURM_ARRAY_TASK_ID --outdir results/e26L$L
