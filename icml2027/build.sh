#!/bin/bash
# Build the ICML submission on a compute node (never on the login node), then check pages and fonts.
# usage: bash build.sh   (run from anywhere)
set -e
D=/coc/flash10/ksingara3/compositionality-theory
srun -p overcap -A overcap --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,ripl-s1,oppy,major,clippy,protocol,sky2,zhurong,zima,brainiac,spot,randotron,uniblab,qt-1,samantha,chappie,claptrap -c 2 --mem=8G -t 20 bash -c "
cd $D/icml2027 && ~/flash/miniconda3/envs/tex/bin/tectonic --keep-logs main.tex 2>&1 | grep -E 'error|Error|undefined' | head -20
~/flash/miniconda3/envs/chunks/bin/python $D/notes/audit/figcheck/icml_check.py $D/icml2027/main.pdf $D/notes/audit/icml_pages \${NPAGES:-10} 2>&1 | grep -v deprecated
~/flash/miniconda3/envs/chunks/bin/python $D/notes/audit/figcheck/fontlist.py $D/icml2027/main.pdf 2>&1 | grep -v deprecated | awk '{print \$3}' | sed -E 's/^[A-Z]{6}\+//' | sort -u | tr '\n' ' '
echo
grep -E 'Warning.*(undefined|multiply)' $D/icml2027/main.log | sort -u | head -20
" 2>&1 | grep -v '^srun'
