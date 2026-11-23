#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e38_%x_%j.out
# usage: sbatch -J name e38.sbatch <script.py> <args>
cd /coc/flash10/ksingara3/compositionality-theory/experiments
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
~/flash/miniconda3/envs/chunks/bin/python "$@"
