#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/task_%A_%a.out
# Runs line SLURM_ARRAY_TASK_ID+1 of the task file given as $1, on a frozen copy of the code.
cd /coc/flash10/ksingara3/compositionality-theory/experiments
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
FROZEN=$(mktemp -d); cp *.py $FROZEN/
export OMP_NUM_THREADS=8
echo "$LINE"
~/flash/miniconda3/envs/chunks/bin/python $FROZEN/$LINE
