#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -W group_list=gj26
#PBS -p 1023
#PBS -j oe
# Sixth batch driver: ONE run per node (user preference), project gj26. Exact --resume, so a job cut by walltime is resubmitted unchanged.
#   qsub -N halt-H1 -l walltime=30:00:00 -o runs/pbs/halt-H1.out -v RUN=H1 scripts/pbs_halt.sh
#   qsub -N halt-gtest -q debug-g -l walltime=00:20:00 -o runs/pbs/halt-gtest.out -v MODE=graphtest scripts/pbs_halt.sh
# If configs/halt_use_cuda_graph exists when the job starts, training uses the captured whole-step CUDA graph (same maths, fewer launches).
cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}" || exit 1
module load cuda/12.9
source /work/go39/b20033/code/generalization_venv/bin/activate
export HOME=/work/go39/b20033 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONUNBUFFERED=1
CELLS_FILE=${CELLS_FILE:-configs/cells_halt.txt}; DATA=data/chain_loop; ATOMIC=data/atomic_joint_2026-09-19/train_atomic.json
echo "=== $(date) host=$(hostname) job=${PBS_JOBID:-none} mode=${MODE:-train} run=${RUN:-} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd src
if [ "${MODE:-train}" = bench ]; then
  # speed / numerics of the training step: precision x CUDA graph, 200 updates each, sequential on one GPU; plus a multi-node launch probe
  echo "--- pbsdsh probe over the job's nodes:"; n=$(sort -u "$PBS_NODEFILE" | wc -l)
  for i in $(seq 0 $((n - 1))); do pbsdsh -n $i -- bash -c 'echo "node $(hostname): $(nvidia-smi -L | head -1)"' || echo "pbsdsh -n $i failed"; done
  for arm in ${BENCH_ARMS:-D H}; do for prec in fp32 tf32 bf16; do for g in 0 1; do
    out=../runs/halt_bench_${arm}_${prec}_g${g}; rm -rf "$out"
    python -u train_halt.py --data_dir ../$DATA --atomic ../$ATOMIC --save_dir "$out" --arm $arm --seed 1 --cuda_graph $g --train_precision $prec --updates 200 --stop_after 200 --eval_every 1000000 --monitor_per_cell 2 > "$out.log" 2>&1
    echo "$arm $prec graph=$g: $(grep -E 'stopped at|Error' "$out.log" | tail -1) | losses $(python -c "import json; print([round(json.loads(l)['loss'], 4) for l in open('$out/train_log.jsonl')])" 2>/dev/null)"
  done; done; done
  echo "=== $(date) bench finished ==="; exit 0
fi
if [ "${MODE:-train}" = graphtest ]; then
  # eager vs CUDA-graph: identical seeds / data, 300 updates each, compare the logged losses and the final weights; report ms / update
  for arm in D H P; do
    for g in 0 1; do
      out=../runs/halt_gtest_${arm}_g${g}; rm -rf "$out"
      python -u train_halt.py --data_dir ../$DATA --atomic ../$ATOMIC --save_dir "$out" --arm $arm --seed 1 --cuda_graph $g --updates 300 --stop_after 300 --eval_every 1000000 > "$out.log" 2>&1
      echo "$arm graph=$g: $(grep -E 'stopped at|Error|Traceback' "$out.log" | tail -1)"
    done
    python - <<PY
import json, torch
a = [json.loads(l) for l in open("../runs/halt_gtest_${arm}_g0/train_log.jsonl")]; b = [json.loads(l) for l in open("../runs/halt_gtest_${arm}_g1/train_log.jsonl")]
print("${arm} loss eager vs graph:", [(x["update"], round(x["loss"], 5), round(y["loss"], 5)) for x, y in zip(a, b)])
sa = torch.load("../runs/halt_gtest_${arm}_g0/last.pt", map_location="cpu", weights_only=False)["model"]; sb = torch.load("../runs/halt_gtest_${arm}_g1/last.pt", map_location="cpu", weights_only=False)["model"]
print("${arm} max |weight difference| after 300 updates:", max(float((sa[k] - sb[k]).abs().max()) for k in sa), "| relative:", max(float((sa[k] - sb[k]).abs().max() / (sa[k].abs().max() + 1e-12)) for k in sa))
PY
  done
  echo "=== $(date) graphtest finished ==="; exit 0
fi
if [ -n "${RUNS:-}" ]; then
  # RUNS=H1+H2+...: the i-th run goes to the i-th node of this job via pbsdsh; each node runs the single-run path of this same script
  i=0; pids=()
  for r in ${RUNS//+/ }; do
    pbsdsh -n $i -- bash -c "cd $PBS_O_WORKDIR && RUN=$r PREC=${PREC:-fp32} GRAPH=${GRAPH:-0} PBS_O_WORKDIR=$PBS_O_WORKDIR bash scripts/pbs_halt.sh" > "../runs/pbs/halt-$r.node.out" 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  echo "=== $(date) multi-run finished (fail=$fail) ==="; exit $fail
fi
line=$(grep -E "^${RUN} *\|" "../$CELLS_FILE") || { echo "unknown run id ${RUN}"; exit 1; }
IFS='|' read -r id arm seed dir extra <<< "$line"; arm=$(echo $arm | xargs); seed=$(echo $seed | xargs); dir=$(echo $dir | xargs)
G=${GRAPH:-0}; [ -f ../configs/halt_use_cuda_graph ] && G=1
mkdir -p "../$dir"; echo "--- run $RUN: arm $arm seed $seed -> $dir (cuda_graph=$G, train precision ${PREC:-fp32}) extra:$extra"
python -u train_halt.py --data_dir ../$DATA --atomic ../$ATOMIC --save_dir "../$dir" --arm "$arm" --seed "$seed" --cuda_graph $G --train_precision ${PREC:-fp32} --resume $extra >> "../$dir/train.log" 2>&1
rc=$?; [ $rc = 0 ] && echo "done   $RUN" || echo "FAILED $RUN (see $dir/train.log)"
echo "=== $(date) finished (fail=$rc) ==="; exit $rc
