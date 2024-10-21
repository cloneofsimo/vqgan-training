

loglr=-7
width=128
lr=$(python -c "import math; print(2**${loglr})")
run_name="stage1_test"
echo "Running ${run_name}"

torchrun --nproc_per_node=8 tvae_trainer.py \
--learning_rate_vae ${lr} \
--vae_ch ${width} \
--run_name ${run_name} \
--num_epochs 20 \
--max_steps 100000 \
--evaluate_every_n_steps 1000 \
--learning_rate_disc 3e-5 \
--batch_size 2 \
--do_clamp \
--do_ganloss \
--project_name "HrDecoderTAE" \
--decoder_also_perform_hr False \
--do_compile True \
--vae_z_channels 16 \
--vae_ch_mult 1,2,4,4 \
--use_lecam True \
--disc_type "hinge" \
#--load_path "/home/ubuntu/auravasa/ckpt/stage_3_hdr_z64_f16_add_flip_lr_disc_1e-4/vae_epoch_0_step_42001.pt"