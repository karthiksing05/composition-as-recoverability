#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a40:1
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --array=0-11
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e33v2_%A_%a.out
cd /coc/flash10/ksingara3/compositionality-theory/experiments
MODELS=(microsoft/phi-4 Qwen/Qwen3-8B allenai/OLMo-2-1124-13B allenai/OLMo-2-1124-7B)
M=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}; S=$((SLURM_ARRAY_TASK_ID % 3))
FROZEN=$(mktemp -d); cp *.py $FROZEN/
export OMP_NUM_THREADS=8 PYTHONPATH=$FROZEN HF_HUB_OFFLINE=1
echo $M $S; hostname
~/flash/miniconda3/envs/chunks/bin/python $FROZEN/e33_lm_controls.py --models $M --seeds $S --outdir /coc/flash10/ksingara3/compositionality-theory/experiments/results/e33v2
