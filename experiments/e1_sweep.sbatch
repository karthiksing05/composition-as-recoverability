#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:30:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1
#SBATCH --array=0-29
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e1_%A_%a.out
Vs=(4 8 16 32 64 128)
V=${Vs[$((SLURM_ARRAY_TASK_ID % 6))]}
S=$((SLURM_ARRAY_TASK_ID / 6))
OUT=/coc/flash10/ksingara3/compositionality-theory/experiments/results/e1/d16_V${V}_s${S}.json
[ -f "$OUT" ] && exit 0
export OMP_NUM_THREADS=4
cd /coc/flash10/ksingara3/compositionality-theory/experiments
~/flash/miniconda3/envs/chunks/bin/python -I e1_calibration.py --d 16 --V $V --seed $S --out $OUT
