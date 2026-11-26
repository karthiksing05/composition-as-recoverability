#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a40:1
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e40_%x_%j.out
# usage: sbatch -J e40_<tag> e40_attr_binding.sbatch <model> <seeds, e.g. 0+1+2> [outdir] [--smoke]
cd /coc/flash10/ksingara3/compositionality-theory/experiments
M=$1; SEEDS=${2:-0+1+2}; OUT=${3:-/coc/flash10/ksingara3/compositionality-theory/experiments/results/e40}; EXTRA=$4
FROZEN=$(mktemp -d); cp *.py $FROZEN/
export OMP_NUM_THREADS=8 PYTHONPATH=$FROZEN HF_HUB_OFFLINE=1
echo $M $SEEDS $OUT; hostname
~/flash/miniconda3/envs/chunks/bin/python $FROZEN/e40_attr_binding.py --models $M --seeds $SEEDS --outdir $OUT $EXTRA
