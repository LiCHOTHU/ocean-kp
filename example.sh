#!/bin/bash

(
python launch_experiment.py ./configs/humanoid-multi-dir-500.json --logdir tem-new --gpu 0 --kl_lambda 10 --n_iteration 200 --seed 4 --global_latent 10.0.0.0.0
)&
(
python launch_experiment.py ./configs/humanoid-multi-dir-500.json --logdir tem-new --gpu 1 --kl_lambda 10 --n_iteration 200 --seed 4 \
 --global_latent 0.0.0.2.3 --constraint logitnormal --var 5. --recurrent --vrnn_latent 0.2.3.0.0 --rnn rnn --temp_res 20 --rnn_sample batch_sampling --traj_batch_size 1
)&
(
python launch_experiment.py ./configs/humanoid-multi-dir-500.json --logdir tem-new --gpu 2 --kl_lambda 10 --n_iteration 200 --seed 4 \
 --global_latent 0.0.0.2.3 --constraint logitnormal --var 5. --recurrent --vrnn_latent 0.2.3.0.0 --rnn rnn --temp_res 20 --rnn_sample single_sampling --traj_batch_size 1
)&
(
python launch_experiment.py ./configs/humanoid-multi-dir-500.json --logdir tem-new --gpu 3 --kl_lambda 10 --n_iteration 200 --seed 4 \
 --global_latent 4.0.0.0.0 --constraint logitnormal --var 5. --recurrent --vrnn_latent 4.0.0.0.0 --rnn rnn --temp_res 20 --rnn_sample single_sampling --traj_batch_size 1
)&

python train_hammer.py ./configs/hammer-multi.json --logdir tem-new --gpu 4 --kl_lambda 10 --n_iteration 200 --seed 4 --global_latent 4.0.0.0.0 --constraint logitnormal --var 5. --recurrent --vrnn_latent 4.0.0.0.0 --rnn rnn --temp_res 20 --rnn_sample single_sampling --traj_batch_size 1


python launch_experiment.py ./configs/humanoid-multi-dir-500.json --logdir tem-new --gpu 4 --kl_lambda 10 --n_iteration 200 --seed 4 --global_latent 4.0.0.0.0 --constraint logitnormal --var 5. --recurrent --vrnn_latent 4.0.0.0.0 --rnn rnn --temp_res 20 --rnn_sample single_sampling --traj_batch_size 1