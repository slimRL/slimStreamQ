launch_job/atari/local_dqn.sh --experiment_name test_Breakout  --first_seed 1 --last_seed 1 --features 32 64 64 256 128 \
    --gamma 0.99 --horizon 27_000 --lambda_trace 0.8 --architecture_type cnn --n_epochs 40 \
    --n_training_steps_per_epoch 250_000 --target_update_period 8000 --epsilon_end 0.01 --epsilon_duration 250_000