#!/bin/bash
# GPU throughput benchmark for the skills task (debug-g, ~15 min). Submit from toy_model/:
#   qsub -o runs/pbs/bench.out scripts/pbs_bench.sh
# Measures, on data/skills_w2d2 (25k train rows, 10k test rows):
#   phase 1: 6 concurrent processes, batch 256          (the round-1 packing)
#   phase 2: 1 process, batch 256 / 1024 / 4096          (train batch size)
#   phase 3: 1 process, batch 256, eval batch 4096        (evaluation cost)
# nvidia-smi is sampled every 5 s into runs/bench/nvidia_smi.log for the whole job.
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:25:00
#PBS -W group_list=go39
#PBS -N skills-bench
#PBS -j oe
#PBS -p 1023

set -u
cd "${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
module purge
module load cuda/12.9
source /work/go39/b20033/code/generalization_venv/bin/activate
export HOME=/work/go39/b20033
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

OUT=runs/bench; mkdir -p $OUT
CONFIG=../configs/skills_w2d2.yaml
DATA=../data/skills_w2d2
EPOCHS=${EPOCHS:-30}
cd src

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader)"; sleep 5; done ) > ../$OUT/nvidia_smi.log 2>&1 &
SMI=$!

run() {  # name, extra train args...
  local name=$1; shift
  local t0=$(date +%s)
  python -u train.py --config $CONFIG --data_dir $DATA --save_dir ../$OUT/$name --no_wandb --epochs $EPOCHS "$@" > ../$OUT/$name.log 2>&1
  local t1=$(date +%s)
  echo "$name: $((t1 - t0)) s for $EPOCHS epochs = $(( (t1 - t0) * 1000 / EPOCHS )) ms/epoch  [$*]"
}
mark() { echo "$(date +%H:%M:%S) ### $1" >> ../$OUT/nvidia_smi.log; echo "### $1"; }

if [ "${SKIP_PHASE1:-0}" = "1" ]; then echo "phase1 skipped"; else
mark "phase1: 6 concurrent, batch 256"
t0=$(date +%s); pids=()
for i in 1 2 3 4 5 6; do
  python -u train.py --config $CONFIG --data_dir $DATA --save_dir ../$OUT/p1_$i --no_wandb --epochs $EPOCHS > ../$OUT/p1_$i.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"   # not a bare `wait`: that would also wait for the nvidia-smi sampler
echo "phase1: $(( $(date +%s) - t0 )) s wall for $EPOCHS epochs x 6 processes = $(( ($(date +%s) - t0) * 1000 / EPOCHS )) ms/epoch each"
fi

mark "phase2: single process, batch 256"
run single_b256 --batch_size 256
mark "phase2: single process, batch 1024"
run single_b1024 --batch_size 1024
mark "phase2: single process, batch 4096"
run single_b4096 --batch_size 4096
mark "phase3: batch 256, eval batch 4096"
run single_b256_eval4096 --batch_size 256 --eval_batch_size 4096
mark "phase3: batch 1024, eval batch 4096"
run single_b1024_eval4096 --batch_size 1024 --eval_batch_size 4096

kill $SMI 2>/dev/null
echo "=== GPU utilisation per phase (mean of samples) ==="
awk '/###/{phase=substr($0, index($0,"###")+4); next} phase!=""{gsub(/%| MiB| W/,""); split($0,a,", "); u[phase]+=a[1]; m[phase]=a[2]; n[phase]++} END{for(p in u) printf "%-40s util %5.1f%%  mem %s MiB  (%d samples)\n", p, u[p]/n[p], m[p], n[p]}' ../$OUT/nvidia_smi.log
echo "=== done $(date) ==="
