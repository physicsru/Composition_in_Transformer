#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -W group_list=go39
#PBS -p 1023
#PBS -j oe
# Re-run the final evaluation of protocol D seeds 1 and 7 (their in-job final eval hit CUDA OOM on a shared node) with a
# smaller rollout batch. Evaluates ckpt_250000.pt ("final") and best_by_val.pt ("best") -> final_eval_{evalonly*}.json.
cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}" || exit 1
module load cuda/12.9; source /work/go39/b20033/code/generalization_venv/bin/activate; export HOME=/work/go39/b20033; export CUDA_VISIBLE_DEVICES=0; export OMP_NUM_THREADS=4; export PYTHONUNBUFFERED=1
cd src
for s in 1 7; do
  for tag in final best; do
    ck=../runs/chain_chainD_s$s/ckpt_250000.pt; [ $tag = best ] && ck=../runs/chain_chainD_s$s/best_by_val.pt
    python -u train_chain.py --data_dir ../data/chain_chainD --save_dir ../runs/chain_chainD_s$s --protocol D --max_len 1024 --rollout_batch 250 --eval_only $ck > ../runs/chain_chainD_s$s/eval_$tag.log 2>&1 \
      && mv ../runs/chain_chainD_s$s/final_eval_evalonly.json ../runs/chain_chainD_s$s/final_eval_$tag.json && mv ../runs/chain_chainD_s$s/predictions_evalonly.jsonl ../runs/chain_chainD_s$s/predictions_$tag.jsonl && echo "done s$s $tag" || echo "FAILED s$s $tag"
  done
done
echo "=== $(date) finished ==="
