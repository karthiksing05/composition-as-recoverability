# Case studies, language models (relation "on", seed 0)

Written by experiments/case_lm.py --summary from experiments/results/case_lm/*.json.
Headlines are means over five relations and 3 seeds; these runs are one relation and one seed,
so the stored E11/E9 values for the same relation and seed are the exact check (`ref`).
Flag = |rerun - headline| > 0.1.

## EleutherAI/pythia-1.4b (layer L12, part code carrier)

| measure | rerun | ref (stored, on, s0) | headline | flag |
|---|---|---|---|---|
| question-state transfer (conv) | 0.018 | 0.018 | 0.085 | ok |
| token-averaged transfer (conv) | 0.020 | 0.020 | ~0 | |
| answer accuracy, E9 protocol (on) | 0.486 | 0.486 | 0.500 | ok |
| answer accuracy, figure question on test sentences | 0.505 (can 0.18, conv 0.83) | | | |

| sentence | answer (margin fig-gnd) | question-state readout (d_fig / d_gnd) | token-avg readout |
|---|---|---|---|
| a book on a bench | bench (-0.12) | book (2.01 / 3.55) | book |
| a bench under a book | book (+2.50) | bench (3.90 / 2.63) | bench |
| a cake on a plate | plate (-1.00) | cake (1.38 / 2.71) | cake |
| a plate under a cake | cake (+1.75) | plate (2.51 / 2.16) | plate |
| a lamp on a rug | lamp (+0.12) | lamp (1.81 / 4.38) | lamp |
| a rug under a lamp | lamp (+1.75) | rug (4.09 / 2.28) | rug |
| a leaf on a rock | rock (-1.63) | leaf (1.95 / 2.96) | leaf |
| a rock under a leaf | leaf (+3.75) | rock (2.92 / 2.66) | rock |
| a key on a map | map (-0.87) | key (2.04 / 2.94) | key |
| a map under a key | key (+0.88) | map (4.50 / 3.01) | map |
| a hat on a sofa | sofa (-0.12) | hat (1.48 / 5.03) | hat |
| a sofa under a hat | hat (+1.87) | sofa (3.82 / 2.87) | sofa |
| a cherry on a pie | pie (-0.75) | cherry (2.35 / 3.94) | cherry |
| a pie under a cherry | cherry (+0.12) | pie (4.06 / 2.51) | pie |
| a basket on a tractor | tractor (-0.12) | basket (1.40 / 3.98) | basket |
| a tractor under a basket | basket (+0.38) | tractor (3.44 / 2.32) | tractor |

## Qwen/Qwen3-8B (layer L27, part code carrier)

| measure | rerun | ref (stored, on, s0) | headline | flag |
|---|---|---|---|---|
| question-state transfer (conv) | 0.916 | 0.918 | 0.870 | ok |
| token-averaged transfer (conv) | 0.030 | 0.030 | ~0 | |
| answer accuracy, E9 protocol (on) | 0.974 | 0.974 | - |  |
| answer accuracy, figure question on test sentences | 0.999 (can 1.00, conv 1.00) | | | |

| sentence | answer (margin fig-gnd) | question-state readout (d_fig / d_gnd) | token-avg readout |
|---|---|---|---|
| a book on a bench | book (+8.00) | book (1.99 / 3.04) | book |
| a bench under a book | book (+9.37) | book (2.43 / 2.80) | bench |
| a cake on a plate | cake (+10.25) | cake (1.31 / 1.83) | cake |
| a plate under a cake | cake (+12.62) | cake (2.04 / 2.57) | plate |
| a lamp on a rug | lamp (+10.75) | lamp (1.74 / 2.68) | lamp |
| a rug under a lamp | lamp (+12.25) | lamp (1.70 / 2.57) | rug |
| a leaf on a rock | leaf (+10.25) | leaf (1.49 / 2.69) | leaf |
| a rock under a leaf | leaf (+15.38) | leaf (2.92 / 3.37) | rock |
| a key on a map | key (+9.38) | key (2.03 / 2.60) | key |
| a map under a key | key (+8.50) | key (2.21 / 2.33) | map |
| a hat on a sofa | hat (+11.63) | hat (3.02 / 3.04) | hat |
| a sofa under a hat | hat (+13.63) | hat (2.98 / 3.67) | sofa |
| a cherry on a pie | cherry (+13.62) | cherry (1.87 / 3.19) | cherry |
| a pie under a cherry | cherry (+8.88) | cherry (2.46 / 2.83) | pie |
| a basket on a tractor | basket (+12.37) | basket (1.67 / 3.81) | basket |
| a tractor under a basket | basket (+15.62) | basket (2.03 / 3.59) | tractor |

## allenai/OLMo-2-1124-7B (layer L16, part code carrier)

| measure | rerun | ref (stored, on, s0) | headline | flag |
|---|---|---|---|---|
| question-state transfer (conv) | 0.142 | 0.144 | 0.200 | ok |
| token-averaged transfer (conv) | 0.018 | 0.018 | ~0 | |
| answer accuracy, E9 protocol (on) | 0.833 | 0.833 | 0.770 | ok |
| answer accuracy, figure question on test sentences | 0.802 (can 0.75, conv 0.85) | | | |

| sentence | answer (margin fig-gnd) | question-state readout (d_fig / d_gnd) | token-avg readout |
|---|---|---|---|
| a book on a bench | bench (-0.30) | book (3.32 / 4.92) | book |
| a bench under a book | book (+3.22) | bench (4.50 / 3.75) | bench |
| a cake on a plate | cake (+1.61) | cake (1.94 / 3.03) | cake |
| a plate under a cake | cake (+3.77) | plate (3.75 / 3.46) | plate |
| a lamp on a rug | lamp (+2.71) | lamp (3.17 / 3.40) | lamp |
| a rug under a lamp | lamp (+3.96) | rug (3.89 / 2.21) | rug |
| a leaf on a rock | leaf (+2.05) | leaf (1.33 / 2.58) | leaf |
| a rock under a leaf | leaf (+2.26) | leaf (3.38 / 3.47) | rock |
| a key on a map | key (+1.72) | key (3.08 / 4.68) | key |
| a map under a key | key (+1.13) | map (4.67 / 4.45) | map |
| a hat on a sofa | hat (+0.66) | hat (2.10 / 4.68) | hat |
| a sofa under a hat | hat (+2.97) | hat (4.56 / 4.76) | sofa |
| a cherry on a pie | cherry (+1.07) | cherry (1.82 / 3.60) | cherry |
| a pie under a cherry | cherry (+1.56) | pie (3.76 / 2.60) | pie |
| a basket on a tractor | basket (+0.42) | basket (2.36 / 3.88) | basket |
| a tractor under a basket | basket (+2.17) | tractor (3.99 / 2.96) | tractor |

## gpt2-large (layer L18, part code carrier)

| measure | rerun | ref (stored, on, s0) | headline | flag |
|---|---|---|---|---|
| question-state transfer (conv) | 0.012 | 0.012 | 0.085 | ok |
| token-averaged transfer (conv) | 0.000 | 0.000 | ~0 | |
| answer accuracy, E9 protocol (on) | 0.495 | 0.503 | 0.500 | ok |
| answer accuracy, figure question on test sentences | 0.498 (can 0.09, conv 0.91) | | | |

| sentence | answer (margin fig-gnd) | question-state readout (d_fig / d_gnd) | token-avg readout |
|---|---|---|---|
| a book on a bench | book (+0.46) | book (2.47 / 4.10) | book |
| a bench under a book | book (+3.73) | bench (3.53 / 1.65) | bench |
| a cake on a plate | plate (-1.31) | cake (1.50 / 2.28) | cake |
| a plate under a cake | cake (+1.99) | plate (2.49 / 2.15) | plate |
| a lamp on a rug | rug (-4.05) | lamp (1.46 / 3.73) | lamp |
| a rug under a lamp | lamp (+3.41) | rug (3.13 / 2.16) | rug |
| a leaf on a rock | rock (-1.20) | leaf (1.83 / 3.16) | leaf |
| a rock under a leaf | leaf (+1.45) | rock (3.36 / 1.87) | rock |
| a key on a map | map (-1.73) | key (2.34 / 3.68) | key |
| a map under a key | key (+2.64) | key (2.78 / 2.96) | map |
| a hat on a sofa | sofa (-5.91) | hat (1.50 / 5.07) | hat |
| a sofa under a hat | hat (+1.06) | sofa (4.34 / 1.88) | sofa |
| a cherry on a pie | cherry (+1.12) | cherry (2.79 / 4.21) | cherry |
| a pie under a cherry | cherry (+4.09) | pie (3.33 / 2.01) | pie |
| a basket on a tractor | tractor (-5.19) | basket (1.67 / 3.47) | basket |
| a tractor under a basket | basket (+2.95) | tractor (3.45 / 1.59) | tractor |

## microsoft/phi-4 (layer L30, part code phrase)

| measure | rerun | ref (stored, on, s0) | headline | flag |
|---|---|---|---|---|
| question-state transfer (conv) | 0.974 | 0.974 | 0.980 | ok |
| token-averaged transfer (conv) | 0.026 | 0.024 | ~0 | |
| answer accuracy, E9 protocol (on) | 0.995 | 0.995 | 0.990 | ok |
| answer accuracy, figure question on test sentences | 1.000 (can 1.00, conv 1.00) | | | |

| sentence | answer (margin fig-gnd) | question-state readout (d_fig / d_gnd) | token-avg readout |
|---|---|---|---|
| a book on a bench | book (+11.25) | book (1.49 / 2.81) | book |
| a bench under a book | book (+5.00) | book (1.85 / 2.38) | bench |
| a cake on a plate | cake (+14.88) | cake (1.20 / 2.71) | cake |
| a plate under a cake | cake (+6.75) | cake (1.44 / 2.69) | plate |
| a lamp on a rug | lamp (+13.19) | lamp (1.52 / 2.99) | lamp |
| a rug under a lamp | lamp (+6.63) | lamp (1.38 / 2.73) | rug |
| a leaf on a rock | leaf (+9.81) | leaf (1.07 / 2.67) | leaf |
| a rock under a leaf | leaf (+5.13) | leaf (1.14 / 2.57) | rock |
| a key on a map | key (+12.75) | key (1.22 / 2.28) | key |
| a map under a key | key (+8.06) | key (1.17 / 1.99) | key |
| a hat on a sofa | hat (+13.13) | hat (1.34 / 2.96) | hat |
| a sofa under a hat | hat (+8.69) | hat (1.60 / 2.76) | sofa |
| a cherry on a pie | cherry (+13.37) | cherry (1.96 / 3.48) | cherry |
| a pie under a cherry | cherry (+4.88) | cherry (2.63 / 2.76) | pie |
| a basket on a tractor | basket (+12.56) | basket (1.46 / 4.07) | basket |
| a tractor under a basket | basket (+9.94) | basket (1.69 / 3.89) | tractor |

