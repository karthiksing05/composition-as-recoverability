#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e31_%x_%j.out
# usage: sbatch -J name e31_score.sbatch <args for e31_score.py>
cd /coc/flash10/ksingara3/compositionality-theory/experiments
export HF_HUB_OFFLINE=1
~/flash/miniconda3/envs/chunks/bin/python e31_score.py "$@"
