#!/bin/bash
# Run one per-organ step over every organ of one atlas, N at a time.
set -u
dir=$1; atlas=$2; script=$3; par=${4:-6}
cd "$dir" || exit 1
mkdir -p logs tables
ls ../"$atlas" | while read d; do
  [ -f "../$atlas/$d/annotation_out/${atlas}_${d}_annotated.h5ad" ] && echo "$d"; done |
xargs -P "$par" -I{} sh -c \
  "python $script \"\$1\" > logs/$(basename $script .py)_\"\$1\".log 2>&1 && echo \"done \$1\" || echo \"FAIL \$1\"" _ {}
