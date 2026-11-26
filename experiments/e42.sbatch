#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e42_%x_%j.out
# usage: sbatch -J name e42.sbatch --system clip-b32-img --seeds 0+1+2 [--smoke --outdir X] ; sbatch e42.sbatch --agg
cd /coc/flash10/ksingara3/compositionality-theory/experiments
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
FROZEN=$(mktemp -d); cp e42_moved_controls.py e42_agg.py e34_relative_places.py e30_diffusion_photos.py e4_pretrained.py e8_vision.py e17_encoder_behavior.py $FROZEN/
cd $FROZEN
if [ "$1" == "--agg" ]; then
  ~/flash/miniconda3/envs/chunks/bin/python e42_agg.py "${@:2}"
else
  ~/flash/miniconda3/envs/chunks/bin/python e42_moved_controls.py "$@"
fi
