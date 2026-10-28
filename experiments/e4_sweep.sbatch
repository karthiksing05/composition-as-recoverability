#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:30:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1
#SBATCH --array=0-5
#SBATCH -o /coc/flash10/ksingara3/compositionality-theory/slurm/e4_%A_%a.out
Ms=(clip-b32 sd15-clip-l t5-small t5-base bert-base-uncased gpt2)
export OMP_NUM_THREADS=8
cd /coc/flash10/ksingara3/compositionality-theory/experiments
~/flash/miniconda3/envs/chunks/bin/python -c "import transformers;print(transformers.__version__)"
FROZEN=$(mktemp -d); cp e4_pretrained.py $FROZEN/  # run a frozen copy so edits cannot affect a running job
~/flash/miniconda3/envs/chunks/bin/python $FROZEN/e4_pretrained.py --models ${Ms[$SLURM_ARRAY_TASK_ID]} --seeds 0+1+2 --outdir results/e4
