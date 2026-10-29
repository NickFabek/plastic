#!/bin/bash
# Run a queue of training jobs back to back. Queue file lines: <run_name> <train.exe args...>  (# = comment)
# A run whose log already has a DONE line is skipped, so a queue can be re-launched after a crash.
cd "$(dirname "$0")/.."
while read -r name args; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  if grep -q '^DONE' "runs/$name/log.txt" 2>/dev/null; then echo "SKIP $name"; continue; fi
  mkdir -p "runs/$name"
  ${BIN:-./build/train.exe} run_dir="runs/$name" $args > "runs/$name/log.txt" 2>&1
  echo "[$name] $(grep -E '^(DONE|ERROR|NAN)' "runs/$name/log.txt" | tail -n 1)"
done < "$1"
echo "QUEUE_DONE $1"
