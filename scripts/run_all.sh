#!/usr/bin/env bash
# Run the full Sec. 5 + Sec. 6 sweep in parallel, one thread per job.
#
# The models are tiny (25-500 hidden units, batch 32), so intra-op threading
# costs more than it buys -- a 100x2 model measured 59% SLOWER on 4 threads than
# on 1. Parallelism therefore comes from running many single-threaded jobs.
#
#   scripts/run_all.sh                      # every model x index x variant
#   scripts/run_all.sh -j 8                 # cap concurrency (default: cores-2)
#   scripts/run_all.sh -m lstm,hybrid       # subset of models
#   scripts/run_all.sh -v base              # subset of variants
#   scripts/run_all.sh -s 0,1,2             # shard 0 of 3 (for multi-machine)
#   scripts/run_all.sh -n                   # dry run: print the plan
#   scripts/run_all.sh -f                   # re-run jobs that already finished
#   scripts/run_all.sh -m lstm -f -- --trials 2 --epochs 2 --max-walks 1
#                                           # anything after -- goes to run_experiment.py
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

detect_cores() {
  if command -v nproc >/dev/null 2>&1; then nproc
  elif [ "$(uname)" = "Darwin" ]; then sysctl -n hw.physicalcpu
  else echo 4; fi
}

JOBS=""; MODELS="arima,lstm,hybrid"; VARIANTS=""; SHARD=""; DRY=0; FORCE=0
while getopts "j:m:v:s:nfh" o; do case $o in
  j) JOBS=$OPTARG ;; m) MODELS=$OPTARG ;; v) VARIANTS=$OPTARG ;; s) SHARD=$OPTARG ;;
  n) DRY=1 ;; f) FORCE=1 ;;
  h) sed -n '2,14p' "$0"; exit 0 ;;
  *) exit 2 ;;
esac; done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift
# A plain string, not an array: macOS ships bash 3.2, where "${ARR[*]}" on an
# EMPTY array is an "unbound variable" error under `set -u`. Arrays also cannot
# be exported to the xargs workers, so this has to be a string either way.
EXTRA_STR="$*"          # passed through to run_experiment.py

CORES=$(detect_cores)
[ -n "$JOBS" ] || JOBS=$(( CORES > 2 ? CORES - 2 : 1 ))

INDICES=("^GSPC" "^FTSE" "^FCHI")
ARIMA_VARIANTS=(base arima_orders_0_3 arima_bic)
NEURAL_VARIANTS=(base dropout_0.05 dropout_0.10 batch_16 batch_64)

# ---- build the job list -------------------------------------------------- #
PLAN=()
for idx in "${INDICES[@]}"; do
  for model in ${MODELS//,/ }; do
    if [ "$model" = "arima" ]; then vs=("${ARIMA_VARIANTS[@]}"); else vs=("${NEURAL_VARIANTS[@]}"); fi
    for v in "${vs[@]}"; do
      # -v filters variants; an unmatched name is silently skipped for models
      # that do not have it (the ARIMA and neural variant lists are disjoint).
      if [ -n "$VARIANTS" ]; then
        case ",$VARIANTS," in *",$v,"*) ;; *) continue ;; esac
      fi
      PLAN+=("$idx|$model|$v")
    done
  done
done

if [ -n "$SHARD" ]; then                      # -s k,n -> take every n-th job
  IFS=, read -r K N <<< "$SHARD"
  SHARDED=(); for i in "${!PLAN[@]}"; do
    [ $(( i % N )) -eq "$K" ] && SHARDED+=("${PLAN[$i]}")
  done
  PLAN=("${SHARDED[@]}")
fi

# ---- skip finished jobs unless -f ---------------------------------------- #
# run_experiment.py suffixes the results directory when a non-default mode is
# requested (_return, _rolling). The resume check has to mirror that, or it
# matches the DEFAULT-mode directory, decides the job is done, and silently
# skips every job -- reporting "queued=0/N" for a sweep that never ran.
neural_sfx=""; arima_sfx=""
case " $EXTRA_STR " in *" --target return "*) neural_sfx="_return" ;; esac
case " $EXTRA_STR " in *" --arima-forecast rolling "*) arima_sfx="_rolling" ;; esac

TODO=()
for job in "${PLAN[@]}"; do
  IFS='|' read -r idx model v <<< "$job"
  if [ "$model" = "arima" ]; then sfx="$arima_sfx"; else sfx="$neural_sfx"; fi
  done_marker="results/$idx/${model}_${v}${sfx}/metrics.csv"
  if [ "$FORCE" -eq 0 ] && [ -f "$done_marker" ]; then
    echo "skip (done): $idx $model $v${sfx:+ ($sfx)}"
  else
    TODO+=("$job")
  fi
done

echo
echo "cores=$CORES  concurrency=$JOBS  queued=${#TODO[@]}/${#PLAN[@]}"
[ ${#TODO[@]} -eq 0 ] && { echo "nothing to do"; exit 0; }
if [ "$DRY" -eq 1 ]; then printf '  %s\n' "${TODO[@]}"; exit 0; fi

mkdir -p logs
START=$(date +%s)

# One thread per worker: see the header. Set before torch is imported.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_one() {
  IFS='|' read -r idx model v <<< "$1"
  log="logs/${idx#^}_${model}_${v}.log"
  t0=$(date +%s)
  # unquoted on purpose: EXTRA_STR must word-split into separate arguments
  if "$PY" scripts/run_experiment.py --model "$model" --index "$idx" \
        --variant "$v" $EXTRA_STR >"$log" 2>&1; then
    echo "  ok   $idx $model $v  ($(( $(date +%s) - t0 ))s)"
  else
    echo "  FAIL $idx $model $v  -> $log"
    echo "$1" >> logs/.failed
  fi
}
export -f run_one; export PY EXTRA_STR

rm -f logs/.failed
printf '%s\n' "${TODO[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

echo
echo "elapsed: $(( ($(date +%s) - START) / 60 ))m"
if [ -s logs/.failed ]; then
  echo "FAILED ($(wc -l < logs/.failed)):"; sed 's/^/  /' logs/.failed
  echo "re-run just these with: scripts/run_all.sh -f   (finished jobs are skipped)"
  exit 1
fi
echo "all ${#TODO[@]} jobs succeeded"
