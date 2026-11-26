#!/bin/bash
# 26 Nov second revision: real-app sessions (same harness as the first revision) on the new head,
# plus inference of the same checkpoint with 9ee250e0d for comparison. Synthetic data only.
set -u; R=<work>/fixwork/c866rev; cd $R; . test/env.sh; export PY=$R/test/session/pyw; S=test/session; L=logs_r2/session_summary.txt
run() { local name=$1; shift; local t0=$(date +%s); "$@" > logs_r2/session_$name.log 2>&1; local rc=$?; echo "$name exit $rc ($(( $(date +%s) - t0 ))s)" >> $L; }
echo "new=56b28fcb8 base=9ee250e0d $(date -u +%FT%TZ)" > $L
rm -rf work_r2_*
EPOCHS=4 run new_train3d nice $S/run_session.sh tree_56b28fcb8 work_r2_train3d train $S/configs/tiny_unet_ink.yaml
CK=$(ls -t $(find work_r2_train3d -name "*_final.pth") | head -1); echo "checkpoint3d $CK" >> $L
run new_infer_default nice $S/run_session.sh tree_56b28fcb8 work_r2_infer_d infer $CK
run new_infer_other nice $S/run_session.sh tree_56b28fcb8 work_r2_infer_o infer $CK $S/configs/tiny_unet_wide_ink.yaml
run base_infer_default nice $S/run_session.sh tree_new work_r2_base_infer_d infer $CK
EPOCHS=2 run new_train2d nice $S/run_session.sh tree_56b28fcb8 work_r2_train2d train2d $S/configs/tiny_unet_2d_ink.yaml
CK2=$(ls -t $(find work_r2_train2d -name "*_final.pth") | head -1); echo "checkpoint2d $CK2" >> $L
echo done >> $L
