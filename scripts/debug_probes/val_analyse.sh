#!/bin/bash
# Turn the val chain's artifacts into the verdict. Host-side, no GPU, no container.
#
#   usage: ./scripts/debug_probes/val_analyse.sh [vis|deep] ...
#
# `vis`  the visibility verdict at the established load. Per fingertip:
#          SIGNAL          fixed build vs control, same forced camera, same frame  <- the question
#          FLOOR_CROSS     the fixed build vs ITSELF in a separate process         <- the verdict floor
#          FLOOR_WITHIN_*  two identical rungs inside one process                  <- the cheaper floor
#          REST            the two builds UNLOADED -- must sit at the floor, or the comparison is
#                          measuring something other than the response to load
#          SELFPRESS       one build's rest vs its own peak -- the scale of what a press does to the
#                          image at all, which is what says whether the crop is even looking at the
#                          fingertips
# `deep` the same, per overtravel depth, from the load ladder.
#
# Every comparison uses the SAME crops, and every crop's ratio is printed, so the headline cannot be
# crop-shopped after the fact.
set -uo pipefail
MODE=${1:-vis}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L=/mnt/home_lustre/sedlam56/projects/REALM/logs/gripper_squeeze
PY=/home/sedlam56/miniconda3/envs/behavior/bin/python
PD="$HERE/scripts/debug_probes/val_pixel_diff.py"
CROPS=full,z2,z4,z8

case $MODE in
vis)
  "$PY" "$HERE/scripts/debug_probes/val_summarise.py" --root "$L" \
      --run xflat=val_xflat_a --run xflat_rep=val_xflat_b --run v2=val_v2_a \
      --out "$L/val_summary.json" 2>&1 | tee "$L/val_summary.txt"

  for F in L R; do
    echo; echo "############ FINGER $F ############"
    P="$L/val"
    "$PY" "$PD" --crop-set "$CROPS" \
      --img xflat_a="${P}_xflat_a_nf1000a_${F}_open_peakRAW.png" \
      --img xflat_a_b="${P}_xflat_a_nf1000b_${F}_open_peakRAW.png" \
      --img xflat_b="${P}_xflat_b_nf1000a_${F}_open_peakRAW.png" \
      --img v2_a="${P}_v2_a_nf1000a_${F}_open_peakRAW.png" \
      --img v2_a_b="${P}_v2_a_nf1000b_${F}_open_peakRAW.png" \
      --img xflat_rest="${P}_xflat_a_nf1000a_${F}_open_restRAW.png" \
      --img v2_rest="${P}_v2_a_nf1000a_${F}_open_restRAW.png" \
      --cmp "SIGNAL_xflat_vs_v2=xflat_a/v2_a" \
      --cmp "FLOOR_CROSS_PROCESS=xflat_a/xflat_b" \
      --cmp "FLOOR_WITHIN_xflat=xflat_a/xflat_a_b" \
      --cmp "FLOOR_WITHIN_v2=v2_a/v2_a_b" \
      --cmp "REST_xflat_vs_v2=xflat_rest/v2_rest" \
      --cmp "SELFPRESS_xflat=xflat_rest/xflat_a" \
      --cmp "SELFPRESS_v2=v2_rest/v2_a" \
      --sbs "$L/val_SBS_${F}.png" --sbs-order xflat_a,v2_a --sbs-crop z4 \
      --sbs-title "fingertip $F pressed, jaws OPEN, authored nf=1000, identical forced camera" \
      --out "$L/val_diff_${F}.json" 2>&1 | tee "$L/val_diff_${F}.txt"
  done
  ;;

deep)
  "$PY" "$HERE/scripts/debug_probes/val_summarise.py" --root "$L" \
      --run xflat_deep=val_xflat_deep --run xflat_deep2=val_xflat_deep2 --run v2_deep=val_v2_deep \
      --out "$L/val_deep_summary.json" 2>&1 | tee "$L/val_deep_summary.txt"

  for F in L R; do
    for D in 0000 0010 0020 0040 0060 0080 0100 0140 0180 0200; do
      A="$L/val_xflat_deep_nf1000a_${F}_open_ot${D}mmRAW.png"
      [ -f "$A" ] || continue
      echo; echo "############ FINGER $F  OVERTRAVEL ${D} mm ############"
      "$PY" "$PD" --crop-set "$CROPS" \
        --img xflat="$A" \
        --img xflat2="$L/val_xflat_deep2_nf1000a_${F}_open_ot${D}mmRAW.png" \
        --img v2="$L/val_v2_deep_nf1000a_${F}_open_ot${D}mmRAW.png" \
        --cmp "SIGNAL_xflat_vs_v2=xflat/v2" \
        --cmp "FLOOR_CROSS_PROCESS=xflat/xflat2" \
        --sbs "$L/val_deep_SBS_${F}_ot${D}mm.png" --sbs-order xflat,v2 --sbs-crop z4 \
        --sbs-title "finger $F, ${D} mm overtravel past first contact, authored nf=1000" \
        --out "$L/val_deep_diff_${F}_ot${D}mm.json"
    done
  done | tee "$L/val_deep_diff.txt"
  ;;

*) echo "usage: val_analyse.sh [vis|deep]"; exit 2 ;;
esac
echo "VAL_ANALYSE_COMPLETE $MODE"
