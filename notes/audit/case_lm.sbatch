#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/notes/audit/case_lm_%j.log
export HF_HUB_OFFLINE=1
cd /coc/flash10/ksingara3/compositionality-theory/experiments
~/flash/miniconda3/envs/chunks/bin/python case_lm.py
