#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e34_%x_%j.out
# usage: sbatch -J name e34_relative_places.sbatch --system clip-b32-img --seeds 0+1+2 [--smoke --outdir X]
cd /coc/flash10/ksingara3/compositionality-theory/experiments
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
FROZEN=$(mktemp -d); cp e34_relative_places.py e30_diffusion_photos.py e4_pretrained.py e8_vision.py e17_encoder_behavior.py agg_e34.py $FROZEN/
cd $FROZEN; mkdir -p /coc/flash10/ksingara3/compositionality-theory/experiments/results/e34
if [ "$1" == "--agg" ]; then
  ~/flash/miniconda3/envs/chunks/bin/python agg_e34.py "${@:2}"
else
  ~/flash/miniconda3/envs/chunks/bin/python e34_relative_places.py --outdir /coc/flash10/ksingara3/compositionality-theory/experiments/results/e34 "$@"
fi
