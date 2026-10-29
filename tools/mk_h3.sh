#!/bin/bash
# H3 righting-force queue on the Jev-constructed init (+ controls on the random init). Needs runs/init_s/c_jev*.bin
set -e
cd "$(dirname "$0")/.."
C="d=256 L=4 B=64 lr=4e-3 lr_emb_mult=30 lr_head_mult=3 eval_every=250 probe=labels/probe.bin tokens=50000000"
I=runs/init_s; PY="uv run --quiet --with numpy --with scipy python"
$PY tools/anchor_variants.py $I/c_jev.bin $I/c_jev_anchor.bin $I/anc_all.bin all > /dev/null
$PY tools/anchor_variants.py $I/c_jev.bin $I/c_jev_anchor.bin $I/anc_in.bin input > /dev/null
$PY tools/anchor_variants.py $I/c_jev.bin $I/c_jev_anchor.bin $I/anc_out.bin output > /dev/null
$PY tools/anchor_variants.py $I/init.bin $I/c_jev_anchor.bin $I/anc_rand.bin all > /dev/null
J="init_from=$I/c_jev.bin"; F="labels=labels/train.bin n_lab=8 concept_layer=0"
cat > queues/h3.txt << EOT
h3_w1 $C $J anchor=$I/anc_all.bin anchor_lambda=1
h3_w10 $C $J anchor=$I/anc_all.bin anchor_lambda=10
h3_w100 $C $J anchor=$I/anc_all.bin anchor_lambda=100
h3_w10_crit $C $J anchor=$I/anc_all.bin anchor_lambda=10 anchor_sched=linear anchor_end=0.5
h3_w10_ema $C $J anchor=$I/anc_all.bin anchor_lambda=10 anchor_ema=0.002
h3_w10_in $C $J anchor=$I/anc_in.bin anchor_lambda=10
h3_w10_out $C $J anchor=$I/anc_out.bin anchor_lambda=10
h3_f01 $C $J $F concept_w=0.1
h3_f1 $C $J $F concept_w=1
h3_f1_crit $C $J $F concept_w=1 concept_sched=linear concept_end=0.5
h3_f1_rand $C $F concept_w=1
h3_w10_rand $C anchor=$I/anc_rand.bin anchor_lambda=10
EOT
cat queues/h3.txt | awk '{print $1}' | tr '\n' ' '
