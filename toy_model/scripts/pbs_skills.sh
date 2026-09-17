#!/bin/bash
# PBS driver for the skills (W2D2) experiments on Miyabi. Submit from toy_model/:
#
#   cd toy_model
#   qsub -N skills-smoke -q debug-g -l walltime=00:25:00 -o runs/pbs/smoke.out \
#        -v CELLS=w2d2+v1,SEEDS=1,EPOCHS=20,RUN_TAG=_smoke scripts/pbs_skills.sh
#   qsub -N skills-s1 -o runs/pbs/s1.out -v CELLS=w2d2+v1+c1+dose25+div4+d23,SEEDS=1 scripts/pbs_skills.sh
#   qsub -N skills-pilot -o runs/pbs/pilot.out -v CELLS_FILE=configs/cells_pilot.txt,CELLS=A_single_c40k+D_eco_c40k,SEEDS=1,EPOCHS=1000 scripts/pbs_skills.sh
#
# Variables (qsub -v; lists are '+'-separated because -v splits on commas):
#   CELLS      cells to run (default w2d2), names from CELLS_FILE
#   CELLS_FILE cell definitions, relative to toy_model/ (default configs/cells_default.txt)
#   SEEDS    training seeds (default 1+7+123); the data seed stays fixed so every cell shares one world
#   EPOCHS   override epochs (default: from the config)
#   train args may contain the placeholder {seed}, replaced by the run seed (e.g. --init_from ../runs/x_s{seed}/epoch3000.pt)
#   CONFIG   config file, relative to toy_model/src (default ../configs/skills_w2d2.yaml)
#   RUN_TAG  suffix for run directories (default empty)
# Every (cell, seed) run executes concurrently on the node's single GPU; the model is tiny.
# Data is generated once per cell into ../data/skills_<cell>; runs go to ../runs/skills_<cell>_s<seed><RUN_TAG>.
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=08:00:00
#PBS -W group_list=go39
#PBS -N skills
#PBS -j oe
#PBS -p 1023

set -u
cd "${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
module purge
module load cuda/12.9
source /work/go39/b20033/code/generalization_venv/bin/activate
export HOME=/work/go39/b20033            # home-quota workaround (same as the generalization project)
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

CELLS=${CELLS:-w2d2}
SEEDS=${SEEDS:-1+7+123}
EPOCHS=${EPOCHS:-}
CONFIG=${CONFIG:-../configs/skills_w2d2.yaml}
RUN_TAG=${RUN_TAG:-}

# Cell definitions live in a text file: one line per cell, "name | generate_data.py args | train.py args"
# ('#' starts a comment). CELLS selects names from that file.
CELLS_FILE=${CELLS_FILE:-configs/cells_default.txt}
declare -A GEN_ARGS TRAIN_ARGS
while IFS= read -r line; do
  line=${line%%#*}; [ -z "${line// /}" ] && continue
  IFS='|' read -r name gen tr <<< "$line"
  name=$(echo "$name" | xargs); GEN_ARGS[$name]=$(echo "${gen:-}" | xargs); TRAIN_ARGS[$name]=$(echo "${tr:-}" | xargs)
done < "$CELLS_FILE"

cd src
echo "=== $(date) host=$(hostname) job=${PBS_JOBID:-none} cells=$CELLS seeds=$SEEDS epochs=${EPOCHS:-config} tag='$RUN_TAG' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p ../runs
pids=(); names=()
for cell in ${CELLS//+/ }; do
  if [ -z "${GEN_ARGS[$cell]+x}" ]; then echo "unknown cell: $cell (known in $CELLS_FILE: ${!GEN_ARGS[*]})"; exit 1; fi
  data_dir=../data/skills_${cell}
  if [ ! -f "$data_dir/train.json" ]; then
    echo "--- generating $data_dir  (${GEN_ARGS[$cell]:-default})"
    if ! python generate_data.py --config "$CONFIG" ${GEN_ARGS[$cell]} --output_dir "$data_dir" > "../runs/gen_skills_${cell}.log" 2>&1; then
      echo "generation failed for $cell:"; cat "../runs/gen_skills_${cell}.log"; exit 1
    fi
  fi
  for seed in ${SEEDS//+/ }; do
    run=../runs/skills_${cell}_s${seed}${RUN_TAG}
    mkdir -p "$run"
    echo "--- starting $run  (${TRAIN_ARGS[$cell]:-default train args})"
    python -u train.py --config "$CONFIG" --data_dir "$data_dir" --save_dir "$run" --seed "$seed" --no_wandb \
        ${EPOCHS:+--epochs $EPOCHS} ${TRAIN_ARGS[$cell]//\{seed\}/$seed} > "$run/train.log" 2>&1 &
    pids+=($!); names+=("$run")
  done
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "done   ${names[$i]}"; else echo "FAILED ${names[$i]} (see train.log)"; fail=1; fi
done
echo; python ../scripts/summarize_skills.py "${names[@]}"
echo "=== $(date) finished (fail=$fail) ==="
exit $fail
