

loglr=-6
width=256
lr=$(python -c "import math; print(2**${loglr})")
run_name="ganloss_false_pool+percep+clamp+no_mse_weight-cont-ganloss-with-mse-0.2"
echo "Running ${run_name}"

torchrun --nproc_per_node=8 vae_trainer.py \
--learning_rate_vae ${lr} \
--vae_ch ${width} \
--run_name ${run_name} \
--num_epochs 20 \
--max_steps 50000 \
--evaluate_every_n_steps 500 \
--learning_rate_disc 1e-4 \
--batch_size 16 \
--do_clamp \
--do_ganloss \
--load_path "/home/ubuntu/auravasa/ckpt/ganloss_false_pool+percep+clamp+no_mse_weight/vae_epoch_4_step_62001.pt"
# --load_path "/home/ubuntu/auravasa/ckpt/exp_vae_ch_256_lr_0.0078125_weighted_percep+f8areapool_l2_0.0/vae_epoch_1_step_27001.pt"