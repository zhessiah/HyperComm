#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --env_name predator_prey \
  --nagents 25 \
  --dim 20 \
  --max_steps 80 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.001 \
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --gat_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --first_gat_normalize \
  --second_gat_normalize \
  --message_decoder \
  --save \
  --seed 0 \
  --gpu 1 \
  --mode cooperative \
  | tee pp_hard_25_agents_coop/HyperComm.log
