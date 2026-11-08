#!/bin/bash
# 27_organ_report.py for every organ of one atlas, N at a time.
set -u
dir=$1; atlas=$2; par=${3:-6}
cd "$dir" || exit 1
mkdir -p logs tables
organs=$(ls ../"$atlas" | while read d; do
  [ -d "../$atlas/$d" ] && [ -f "../$atlas/$d/annotation_out/${atlas}_${d}_annotated.h5ad" ] && echo "$d"; done)
echo "$organs" | xargs -P "$par" -I{} sh -c \
  'python 27_organ_report.py "$1" > logs/report_"$1".log 2>&1 && echo "done $1" || echo "FAIL $1"' _ {}
