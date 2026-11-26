#!/bin/bash
# 26 Nov revision: real-app sessions (scripts as in the 26 Nov rerun, plus train2d and SEED) on the new head,
# 1d3b563ed (old) and 2333252cb (first PR commit only), synthetic data only.
set -u; R=<work>/fixwork/c866rev; cd $R; . test/env.sh; export PY=$R/test/session/pyw; S=test/session; L=logs/session_summary.txt
run() { local name=$1; shift; local t0=$(date +%s); "$@" > logs/session_$name.log 2>&1; local rc=$?; echo "$name exit $rc ($(( $(date +%s) - t0 ))s)" >> $L; }
echo "new=9ee250e0d old=1d3b563ed launchfix=2333252cb $(date -u +%FT%TZ)" > $L
rm -rf work_*
EPOCHS=4 run new_train3d nice $S/run_session.sh tree_new work_new_train3d train $S/configs/tiny_unet_ink.yaml
CK=$(ls -t $(find work_new_train3d -name "*.pth") | head -1); echo "checkpoint3d $CK" >> $L
run new_infer_default nice $S/run_session.sh tree_new work_new_infer_d infer $CK
run new_infer_other nice $S/run_session.sh tree_new work_new_infer_o infer $CK $S/configs/tiny_unet_wide_ink.yaml
for seed in 0 0 1; do SEED=$seed run launchfix_infer_other_seed${seed}_$RANDOM nice $S/run_session.sh tree_launchfix work_lf_o infer $CK $S/configs/tiny_unet_wide_ink.yaml; done
EPOCHS=2 run new_train2d nice $S/run_session.sh tree_new work_new_train2d train2d $S/configs/tiny_unet_2d_ink.yaml
CK2=$(ls -t $(find work_new_train2d -name "*.pth") | head -1); echo "checkpoint2d $CK2" >> $L
EPOCHS=2 run old_train2d nice $S/run_session.sh tree_old work_old_train2d train2d $S/configs/tiny_unet_2d_ink.yaml
echo "old_train2d checkpoints: $(find work_old_train2d -name "*.pth" | wc -l)" >> $L
echo done >> $L
