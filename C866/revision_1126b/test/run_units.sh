#!/bin/bash
# per-commit napari_trainer tests (the final test files copied into every tree) + shared regression
# tree_new = git archive of 9ee250e0d, tree_main = 659e2fadb (from the first revision); its log was renamed unit_napari_trainer_9ee250e0d.log
set -u; R=<work>/fixwork/c866rev; cd $R; . test/env.sh; L=logs_r2
HEADTESTS=$R/tree_56b28fcb8/vesuvius/tests/napari_trainer
for t in 6198b3dc5 3c209902f 524f0ed14 new main; do
  rm -rf tree_$t/vesuvius/tests/napari_trainer && cp -a $HEADTESTS tree_$t/vesuvius/tests/napari_trainer
done
for t in 56b28fcb8 6198b3dc5 3c209902f 524f0ed14 new main; do
  test/run_tests.sh tree_$t $R/bt_$t tests/napari_trainer > $L/unit_napari_trainer_$t.log 2>&1
  echo "$t: $(tail -1 $L/unit_napari_trainer_$t.log)" >> $L/unit_summary.txt
done
REG="tests/models/configuration tests/models/run/test_inference_legacy_checkpoint.py tests/models/datasets/test_find_valid_patches_plain_array.py tests/models/build/test_primus_shapes.py"
for t in 56b28fcb8 main; do
  test/run_tests.sh tree_$t $R/bt_reg_$t $REG > $L/regression_$t.log 2>&1
  echo "regression $t: $(tail -1 $L/regression_$t.log)" >> $L/unit_summary.txt
done
echo done >> $L/unit_summary.txt
