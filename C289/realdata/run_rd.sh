#!/bin/bash
# usage: run_rd.sh <CID> <label main|fix> <cmd...>   prints a "== CID on side" block
cid=$1; side=$2; shift 2
echo "== $cid on $side  ($(date -u +%FT%TZ))"
"$@" 2>&1 | grep -v -i "warning\|it/s\]" | grep -v "^<tifffile\." | tail -40
