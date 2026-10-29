#!/bin/bash
# Build constructed inits from calibrated params and write the H2 queue (construction vs controls, S model).
# usage: tools/mk_h2.sh "<common train args>"
set -e
cd "$(dirname "$0")/.."
COMMON="$1"; I=runs/init_s
PY="uv run --quiet --with numpy --with scipy python"
cons() {  # name labels calib_json
  P=$(uv run --quiet python -c "import json;p=json.load(open('$3'))['params'];print(' '.join(f'--{k} {v}' for k,v in p.items()))")
  $PY tools/construct.py --base $I/init.bin --out $I/$1.bin --mode $4 --labels $2 --big_kind cls --cls_name km64 $P --anchor_out $I/$1_anchor.bin --meta_out $I/$1_meta.json > /dev/null
}
$PY tools/construct.py --base $I/init.bin --out $I/uni.bin --mode unigram > /dev/null
cons stat jev runs/calib_stat_cls_s/best.json stat
for lab in jev kmeans shuf coarse; do [ -f runs/calib_c_${lab}_s/best.json ] && cons c_$lab $lab runs/calib_c_${lab}_s/best.json both; done
{
  echo "h2_base $COMMON"
  echo "h2_uni $COMMON init_from=$I/uni.bin"
  echo "h2_stat $COMMON init_from=$I/stat.bin"
  for lab in jev kmeans shuf coarse; do [ -f $I/c_$lab.bin ] && echo "h2_c_$lab $COMMON init_from=$I/c_$lab.bin"; done
} > queues/h2.txt
cat queues/h2.txt
