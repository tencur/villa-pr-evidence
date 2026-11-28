#!/bin/sh
# Usage: run_tests.sh <villa git dir> <scratch dir> <log dir> [<vc_reader python dir>]
# Exports each tree's vesuvius/ with git archive and runs tests/models/run on it with the same
# Python (PYTHON, default python; CPU torch). vc_reader must be importable (4th argument is
# prepended to PYTHONPATH if given).
#   merge  = 3b3618644 (07e29fccd + origin/main ad0f8646d)
#   main   = ad0f8646d
#   bot    = 07e29fccd (PR head before the merge)
#   main+branch-tests = main's code with this branch's two test files copied over
set -u
REPO=$1; WORK=$2; LOGS=$3; VCR=${4:-}
PY=${PYTHON:-python}
MERGE=3b3618644; MAIN=ad0f8646d; BOT=07e29fccd
BRANCH_TESTS="test_tta_vector_heads.py test_finalize_multiclass_scale.py"
mkdir -p "$WORK" "$LOGS"

export_tree() {  # <rev> <dir>
    rm -rf "$2"; mkdir -p "$2"
    git -C "$REPO" archive "$1" vesuvius | tar x -C "$2"
}

run_tree() {  # <label> <dir>
    ( cd "$2/vesuvius" &&
      PYTHONPATH="$2/vesuvius/src${VCR:+:$VCR}" "$PY" -m pytest -p no:cacheprovider -q tests/models/run ) \
        > "$LOGS/pytest_$1.log" 2>&1
    tail -n 1 "$LOGS/pytest_$1.log"
}

export_tree $MERGE "$WORK/merge" && run_tree merge-$MERGE "$WORK/merge"
export_tree $MAIN  "$WORK/main"  && run_tree main-$MAIN  "$WORK/main"
export_tree $BOT   "$WORK/bot"   && run_tree bot-$BOT    "$WORK/bot"

# main's code with this branch's test files
export_tree $MAIN "$WORK/main_branch_tests"
for f in $BRANCH_TESTS; do
    cp "$WORK/merge/vesuvius/tests/models/run/$f" "$WORK/main_branch_tests/vesuvius/tests/models/run/$f"
done
run_tree main-$MAIN-with-branch-tests "$WORK/main_branch_tests"
# test_tta_vector_heads.py imports the branch's new module run/orientation.py, so on main it
# fails at collection. (a) the rest of the directory despite that error; (b) with only
# orientation.py (pure numpy/torch helpers, not called by main's code) added so the file imports.
( cd "$WORK/main_branch_tests/vesuvius" &&
  PYTHONPATH="$WORK/main_branch_tests/vesuvius/src${VCR:+:$VCR}" "$PY" -m pytest -p no:cacheprovider -q \
      --continue-on-collection-errors tests/models/run ) > "$LOGS/pytest_main-$MAIN-with-branch-tests-continue.log" 2>&1
tail -n 1 "$LOGS/pytest_main-$MAIN-with-branch-tests-continue.log"
cp "$WORK/merge/vesuvius/src/vesuvius/models/run/orientation.py" "$WORK/main_branch_tests/vesuvius/src/vesuvius/models/run/"
run_tree main-$MAIN-with-branch-tests-and-orientation "$WORK/main_branch_tests"

# git diff stat of the branch against main, before and after the merge
git -C "$REPO" diff --stat $MAIN $MERGE > "$LOGS/diffstat_main-$MAIN..merge-$MERGE.txt"
git -C "$REPO" diff --stat c8f3ba453 $BOT > "$LOGS/diffstat_base-c8f3ba453..bot-$BOT.txt"
