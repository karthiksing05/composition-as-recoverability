#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-7
#SBATCH -J e27
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/notes/audit/e27_%a.log
cd /coc/flash10/ksingara3/compositionality-theory/experiments
~/flash/miniconda3/envs/chunks/bin/python e27_rhm_input.py $SLURM_ARRAY_TASK_ID
