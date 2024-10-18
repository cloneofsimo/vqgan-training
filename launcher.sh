

loglr=-7
width=64
lr=$(python -c "import math; print(2**${loglr})")
run_name="stage_4_msepool-cont-512-1.0-1.0-batch-gradnorm_make_deterministic"
echo "Running ${run_name}"

torchrun --nproc_per_node=8 vae_trainer.py \
--learning_rate_vae ${lr} \
--vae_ch ${width} \
--run_name ${run_name} \
--num_epochs 20 \
--max_steps 100000 \
--evaluate_every_n_steps 500 \
--learning_rate_disc 1e-5 \
--batch_size 12 \
--do_clamp \
--do_ganloss \
--project_name "HrDecoderAE" \
--decoder_also_perform_hr True
#--load_path "/home/ubuntu/auravasa/ckpt/stage_3_msepool-cont-512-1.0-1.0-batch-gradnorm/vae_epoch_1_step_23501.pt"
#--load_path "/home/ubuntu/auravasa/ckpt/stage2_msepool-cont-512-1.0-1.0-batch-gradnorm/vae_epoch_0_step_28501.pt"
# --load_path "/home/ubuntu/auravasa/ckpt/exp_vae_ch_256_lr_0.0078125_weighted_percep+f8areapool_l2_0.0/vae_epoch_1_step_27001.pt"  