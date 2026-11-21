#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e31_%x_%j.out
# CPU job: usage sbatch -J name e31_py.sbatch script.py args
cd /coc/flash10/ksingara3/compositionality-theory/experiments
~/flash/miniconda3/envs/chunks/bin/python "$@"
