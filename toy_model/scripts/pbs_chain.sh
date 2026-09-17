#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -W group_list=go39
#PBS -p 1023
#PBS -j oe
# Chain task driver (docs/shallow_training_deep_composition_plan_2026-09-16.md). Same conventions as pbs_skills.sh:
#   qsub -N chain-ABC -o runs/pbs/chain.out -v CELLS_FILE=configs/cells_chain.txt,CELLS=chainA+chainB+chainC,SEEDS=1+7+123 scripts/pbs_chain.sh
# cells file: "name | generate_chain.py args | train_chain.py args"; data -> data/chain_<cell>, runs -> runs/chain_<cell>_s<seed>.
# Train args may contain {seed}.
cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}" || exit 1
module load cuda/12.9
source /work/go39/b20033/code/generalization_venv/bin/activate
export HOME=/work/go39/b20033
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

CELLS=${CELLS:-chainA}
SEEDS=${SEEDS:-1+7+123}
CELLS_FILE=${CELLS_FILE:-configs/cells_chain.txt}
RUN_TAG=${RUN_TAG:-}
declare -A GEN_ARGS TRAIN_ARGS
while IFS= read -r line; do
  line=${line%%#*}; [ -z "${line// /}" ] && continue
  IFS='|' read -r name gen tr <<< "$line"
  name=$(echo "$name" | xargs); GEN_ARGS[$name]=$(echo "${gen:-}" | xargs); TRAIN_ARGS[$name]=$(echo "${tr:-}" | xargs)
done < "$CELLS_FILE"

cd src
echo "=== $(date) host=$(hostname) job=${PBS_JOBID:-none} cells=$CELLS seeds=$SEEDS tag='$RUN_TAG' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
mkdir -p ../runs
pids=(); names=()
for cell in ${CELLS//+/ }; do
  if [ -z "${GEN_ARGS[$cell]+x}" ]; then echo "unknown cell: $cell (known in $CELLS_FILE: ${!GEN_ARGS[*]})"; exit 1; fi
  data_dir=../data/chain_${cell}
  if [ ! -f "$data_dir/meta.json" ]; then
    echo "--- generating $data_dir  (${GEN_ARGS[$cell]:-default})"
    if ! python generate_chain.py --output_dir "$data_dir" ${GEN_ARGS[$cell]} > "../runs/gen_chain_${cell}.log" 2>&1; then
      echo "generation failed for $cell:"; cat "../runs/gen_chain_${cell}.log"; exit 1
    fi
    grep -E "AUDIT|w2|d2" "../runs/gen_chain_${cell}.log" | head -3
  fi
  for seed in ${SEEDS//+/ }; do
    run=../runs/chain_${cell}_s${seed}${RUN_TAG}
    mkdir -p "$run"
    echo "--- starting $run  (${TRAIN_ARGS[$cell]:-default train args})"
    python -u train_chain.py --data_dir "$data_dir" --save_dir "$run" --seed "$seed" --resume \
        ${TRAIN_ARGS[$cell]//\{seed\}/$seed} > "$run/train.log" 2>&1 &
    pids+=($!); names+=("$run")
  done
done
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "done   ${names[$i]}"; else echo "FAILED ${names[$i]} (see train.log)"; fail=1; fi
done
echo "=== $(date) finished (fail=$fail) ==="
exit $fail
