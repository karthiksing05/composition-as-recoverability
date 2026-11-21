#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
#SBATCH -t 01:00:00
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e33_smoke_%j.out
cd /coc/flash10/ksingara3/compositionality-theory/experiments
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
~/flash/miniconda3/envs/chunks/bin/python e33_lm_controls.py --models gpt2-large --seeds 0 --outdir /coc/flash10/ksingara3/compositionality-theory/experiments/results/e33_smoke --small
