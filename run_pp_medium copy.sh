#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --env_name predator_prey \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.015 \
  --detach_gap 10 \
  --lrate 0.001 \
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --learn_second_graph \
  --first_gat_normalize \
  --second_gat_normalize \
  --save \
  --seed 88 \
  --gpu 0 \
  --mode cooperative \
  | tee pp_medium_5_agents_coop/HyperComm.log