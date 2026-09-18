#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -W group_list=go39
#PBS -p 1023
#PBS -j oe
# Fifth batch driver (docs/experiments_latent_batch2_20runs.md §11). Launches the listed run ids of configs/cells_latent2.txt
# concurrently on one GPU node; every process trains to 250k (exact --resume from last.pt) and then writes its final evaluation
# (idempotent), so a job cut off by walltime is simply resubmitted with the same command.
#   qsub -N lat2-a -l walltime=06:30:00 -o runs/pbs/lat2_a.out -v MPS=1,RUNS=01+05+08+11+15+02+06+09+12+16 scripts/pbs_latent.sh   # 10 runs / GPU under MPS: 72 ms / update
#   qsub -N lat2-smoke -q debug-g -l walltime=00:20:00 -o runs/pbs/lat2_smoke.out -v RUNS=01+11+15+18,RUN_TAG=_tput,EXTRA=--stop_after+300 scripts/pbs_latent.sh
# EXTRA: extra train_latent.py arguments, '+'-separated (qsub -v splits on commas / spaces).
cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}" || exit 1
module load cuda/12.9
source /work/go39/b20033/code/generalization_venv/bin/activate
export HOME=/work/go39/b20033
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1
RUNS=${RUNS:-01}; CELLS_FILE=${CELLS_FILE:-configs/cells_latent2.txt}; DATA=${DATA:-data/chain_loop}; RUN_TAG=${RUN_TAG:-}; EXTRA=${EXTRA:-}
declare -A COND SEED DIR
while IFS= read -r line; do
  line=${line%%#*}; [ -z "${line// /}" ] && continue
  IFS='|' read -r id c s d <<< "$line"; id=$(echo "$id" | xargs); COND[$id]=$(echo "$c" | xargs); SEED[$id]=$(echo "$s" | xargs); DIR[$id]=$(echo "$d" | xargs)
done < "$CELLS_FILE"
echo "=== $(date) host=$(hostname) job=${PBS_JOBID:-none} runs=$RUNS tag='$RUN_TAG' extra='$EXTRA' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
[ -f "$DATA/latent_eval_manifest.json" ] || { echo "missing $DATA/latent_eval_manifest.json (run src/generate_latent_eval.py)"; exit 1; }
# MPS=1: start an NVIDIA MPS daemon so the packed processes share the GPU concurrently. Without it their CUDA contexts are
# time-sliced and these launch-bound tiny steps get no benefit from packing (measured 2026-09-18: 10 processes -> 141 ms / update each).
if [ "${MPS:-0}" = 1 ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_${PBS_JOBID:-$$}; export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${PBS_JOBID:-$$}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if nvidia-cuda-mps-control -d; then echo "MPS daemon started"; sleep 3; else echo "MPS daemon failed to start -- continuing without MPS"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; fi
fi
cd src; pids=(); names=(); fail=0
for id in ${RUNS//+/ }; do
  if [ -z "${COND[$id]+x}" ]; then echo "unknown run id $id"; exit 1; fi
  run=../${DIR[$id]}${RUN_TAG}; mkdir -p "$run"
  echo "--- run $id: cond ${COND[$id]} seed ${SEED[$id]} -> $run"
  python -u train_latent.py --data_dir "../$DATA" --save_dir "$run" --cond "${COND[$id]}" --seed "${SEED[$id]}" --resume ${EXTRA//+/ } >> "$run/train.log" 2>&1 &
  pids+=($!); names+=("$id:${COND[$id]}_s${SEED[$id]}")
done
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "done   ${names[$i]}"; else echo "FAILED ${names[$i]} (see train.log)"; fail=1; fi
done
if [ "${POST_EVAL:-0}" = 1 ]; then      # throughput test: time the final evaluation of the (partially trained) runs, all at once
  echo "--- $(date) post-eval stage"; pids=(); names=()
  for id in ${RUNS//+/ }; do
    run=../${DIR[$id]}${RUN_TAG}
    python -u train_latent.py --data_dir "../$DATA" --save_dir "$run" --cond "${COND[$id]}" --seed "${SEED[$id]}" --eval_only >> "$run/eval.log" 2>&1 &
    pids+=($!); names+=("$id:eval")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "done   ${names[$i]}"; else echo "FAILED ${names[$i]} (see eval.log)"; fail=1; fi
  done
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true
[ "${MPS:-0}" = 1 ] && [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && { echo quit | nvidia-cuda-mps-control; tail -5 "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null; }
echo "=== $(date) finished (fail=$fail) ==="
exit $fail
