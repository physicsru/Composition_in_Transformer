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
line=$(grep -E "^${RUN} *\|" "../$CELLS_FILE") || { echo "unknown run id ${RUN}"; exit 1; }
IFS='|' read -r id arm seed dir extra <<< "$line"; arm=$(echo $arm | xargs); seed=$(echo $seed | xargs); dir=$(echo $dir | xargs)
G=0; [ -f ../configs/halt_use_cuda_graph ] && G=1
mkdir -p "../$dir"; echo "--- run $RUN: arm $arm seed $seed -> $dir (cuda_graph=$G) extra:$extra"
python -u train_halt.py --data_dir ../$DATA --atomic ../$ATOMIC --save_dir "../$dir" --arm "$arm" --seed "$seed" --cuda_graph $G --resume $extra >> "../$dir/train.log" 2>&1
rc=$?; [ $rc = 0 ] && echo "done   $RUN" || echo "FAILED $RUN (see $dir/train.log)"
echo "=== $(date) finished (fail=$rc) ==="; exit $rc
