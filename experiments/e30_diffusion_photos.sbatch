#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e30_%x_%j.out
# usage: sbatch -J name e30_diffusion_photos.sbatch --model sd15 --task trained --seeds 0
cd /coc/flash10/ksingara3/compositionality-theory/experiments
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
FROZEN=$(mktemp -d); cp e30_diffusion_photos.py e4_pretrained.py e8_vision.py e17_encoder_behavior.py $FROZEN/
cd $FROZEN; mkdir -p /coc/flash10/ksingara3/compositionality-theory/experiments/results/e30
~/flash/miniconda3/envs/chunks/bin/python e30_diffusion_photos.py --outdir /coc/flash10/ksingara3/compositionality-theory/experiments/results/e30 "$@"
