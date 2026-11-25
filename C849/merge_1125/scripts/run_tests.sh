#!/bin/bash
# 2026-11-25: whole tests/neural_tracing/test_fiber_trace_3d.py on four trees, with the same prebuilt vc_reader
# (older than #2680, so it writes vc-delta3d for 3-D arrays only) and the same test-only shim for every tree
# (scripts/delta3d_shim.py: a 4-D vc-delta3d request is stored with blosc/zstd).
#   main      = upstream main 5933623f7
#   merged    = d9b62051f (b34eebca8 + merge of 5933623f7; conflict in infer.py resolved by keeping both checks)
#   oldhead   = b34eebca8 (the PR head before the merge)
#   maintests = main 5933623f7 code with the merged tree's test file
for spec in main:origin/main merged:d9b62051f oldhead:b34eebca8 maintests:origin/main:d9b62051f; do
  IFS=: read k r t <<< "$spec"; D=<tmp>/m2655_$k; rm -rf $D; mkdir -p $D
  git archive $r vesuvius lasagna | tar x -C $D
  [ -n "$t" ] && git show $t:vesuvius/tests/neural_tracing/test_fiber_trace_3d.py > $D/vesuvius/tests/neural_tracing/test_fiber_trace_3d.py
  (cd $D/vesuvius && PYTHONPATH=<shim dir>:$D/vesuvius/src:$D:$D/lasagna:<vc_reader> python -m pytest -q -p no:cacheprovider \
     -p delta3d_shim --tb=line tests/neural_tracing/test_fiber_trace_3d.py > logs/pytest_shim_$k.log 2>&1)
done
