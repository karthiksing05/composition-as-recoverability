#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e35_%x_%j.out
# usage: sbatch -J name e35.sbatch <script.py> <args>   (script = e35_dit.py | e35_generate.py | e35_text.py | e31_score.py | e31_analyze.py | agg_e30.py | agg_e34.py)
cd /coc/flash10/ksingara3/compositionality-theory/experiments
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
FROZEN=$(mktemp -d -p /coc/flash10/ksingara3/compositionality-theory/experiments/e35/frozen); cp e35_dit.py e35_generate.py e35_text.py e30_diffusion_photos.py e34_relative_places.py e4_pretrained.py e8_vision.py e17_encoder_behavior.py e31_common.py e31_score.py e31_analyze.py agg_e30.py agg_e34.py e37_gligen.py e37_analyze.py e37_plan.py e36_harness_lm.py e9_behavior.py e7_roles.py e11_tokens.py $FROZEN/ 2>/dev/null
cd $FROZEN
~/flash/miniconda3/envs/chunks/bin/python "$@"
