#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/notes/audit/agg_e19.txt
cd /coc/flash10/ksingara3/compositionality-theory/experiments
~/flash/miniconda3/envs/chunks/bin/python agg_e19.py
