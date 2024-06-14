#!/bin/bash
export OMP_NUM_THREADS=1

python -u main_smac.py \
  --env_name smac \
  --map_name 10m_vs_11m \
  --nprocesses 1 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --use_gat_encoder \
  --save \
  --seed 79 \
  --gpu 2 \
  | tee smac_10_agents/HyperComm.log
