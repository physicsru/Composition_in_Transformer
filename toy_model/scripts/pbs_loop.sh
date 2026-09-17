#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=08:00:00
#PBS -W group_list=go39
#PBS -p 1023
#PBS -j oe
# Loop / local-executor batch driver (docs/experiments_loop_next.md). One shared dataset, several trainer scripts:
#   qsub -N loop-L -o runs/pbs/loop_L.out -v CELLS=L,SEEDS=1+7+123 scripts/pbs_loop.sh
#   qsub -N loop-O -o runs/pbs/loop_O.out -v MODE=O scripts/pbs_loop.sh        # arm O: re-evaluate C_k1 (no training) on the extended tests
# cells file (configs/cells_loop.txt): "name | script | train args"; runs -> runs/loop_<cell>_s<seed>; {seed} substituted.
cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}" || exit 1
module load cuda/12.9
source /work/go39/b20033/code/generalization_venv/bin/activate
export HOME=/work/go39/b20033
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

CELLS=${CELLS:-L}
SEEDS=${SEEDS:-1+7+123}
CELLS_FILE=${CELLS_FILE:-configs/cells_loop.txt}
DATA=${DATA:-data/chain_loop}
DATA_ARGS=${DATA_ARGS:---k 1 --test_depths 2,3,4,8,16,32,64,128}
MODE=${MODE:-train}
RUN_TAG=${RUN_TAG:-}
declare -A SCRIPT TRAIN_ARGS
while IFS= read -r line; do
  line=${line%%#*}; [ -z "${line// /}" ] && continue
  IFS='|' read -r name sc tr <<< "$line"
  name=$(echo "$name" | xargs); SCRIPT[$name]=$(echo "${sc:-}" | xargs); TRAIN_ARGS[$name]=$(echo "${tr:-}" | xargs)
done < "$CELLS_FILE"

cd src
echo "=== $(date) host=$(hostname) job=${PBS_JOBID:-none} mode=$MODE cells=$CELLS seeds=$SEEDS tag='$RUN_TAG' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
mkdir -p ../runs
data_dir=../$DATA
if [ ! -f "$data_dir/meta.json" ]; then
  echo "--- generating $data_dir ($DATA_ARGS)"
  if ! python generate_chain.py --output_dir "$data_dir" $DATA_ARGS > "../runs/gen_$(basename "$DATA").log" 2>&1; then
    echo "generation failed:"; cat "../runs/gen_$(basename "$DATA").log"; exit 1
  fi
fi
pids=(); names=()
if [ "$MODE" = "O" ]; then
  # arm O (plan table row O, §12 P0.4): the trained C_k1 models, no new training, free rollouts + external step-wise calling
  # on the extended test matrix (d up to 128) for the final and the best_by_val checkpoints of every seed.
  # six evaluations share the GPU: protocol C free rollouts at d = 128 need --rollout_batch 250 (1000 -> CUDA OOM, job 3385021)
  O_TAGS=${O_TAGS:-final+best}
  for seed in ${SEEDS//+/ }; do
    run=../runs/chain_chainC_k1_s${seed}
    for tag in ${O_TAGS//+/ }; do
      ck=$run/ckpt_250000.pt; [ "$tag" = best ] && ck=$run/best_by_val.pt
      echo "--- O eval $run $tag"
      python -u train_chain.py --data_dir "$data_dir" --save_dir "$run" --protocol C --seed "$seed" --eval_only "$ck" --eval_tag "O_${tag}" \
          --rollout_batch "${ROLLOUT_BATCH:-250}" > "$run/eval_O_${tag}.log" 2>&1 &
      pids+=($!); names+=("$run:$tag")
    done
  done
else
  for cell in ${CELLS//+/ }; do
    if [ -z "${SCRIPT[$cell]+x}" ]; then echo "unknown cell: $cell (known in $CELLS_FILE: ${!SCRIPT[*]})"; exit 1; fi
    for seed in ${SEEDS//+/ }; do
      run=../runs/loop_${cell}_s${seed}${RUN_TAG}
      mkdir -p "$run"
      echo "--- starting $run  (${SCRIPT[$cell]} ${TRAIN_ARGS[$cell]})"
      python -u "${SCRIPT[$cell]}" --data_dir "$data_dir" --save_dir "$run" --seed "$seed" --resume \
          ${TRAIN_ARGS[$cell]//\{seed\}/$seed} $EXTRA_ARGS > "$run/train.log" 2>&1 &
      pids+=($!); names+=("$run")
    done
  done
fi
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "done   ${names[$i]}"; else echo "FAILED ${names[$i]} (see log)"; fail=1; fi
done
echo "=== $(date) finished (fail=$fail) ==="
exit $fail
