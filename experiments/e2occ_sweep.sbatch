#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1
#SBATCH --array=0-14
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e2occ_%A_%a.out
Vs=(16 32 64)
V=${Vs[$((SLURM_ARRAY_TASK_ID % 3))]}
S=$((SLURM_ARRAY_TASK_ID / 3))
export OMP_NUM_THREADS=4
cd /coc/flash10/ksingara3/compositionality-theory/experiments
FROZEN=$(mktemp -d); cp e2_glyph.py $FROZEN/  # run a frozen copy so edits cannot affect a running job
~/flash/miniconda3/envs/chunks/bin/python $FROZEN/e2_glyph.py --archs slot+mlp+cnn+tf --ms 8+16+32+64 --Vs $V --seeds $S --steps 4000 --second occ --outdir results/e2occ
