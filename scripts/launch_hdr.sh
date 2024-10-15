

loglr=-6
width=128
lr=$(python -c "import math; print(2**${loglr})")
run_name="stage_1_perceppool_128_hdr_nogan"
echo "Running ${run_name}"

torchrun --nproc_per_node=8 vae_trainer.py \
--learning_rate_vae ${lr} \
--vae_ch ${width} \
--run_name ${run_name} \
--num_epochs 20 \
--max_steps 100000 \
--evaluate_every_n_steps 500 \
--learning_rate_disc 1e-5 \
--batch_size 8 \
--do_clamp \
--project_name "HrDecoderAE" \
--decoder_also_perform_hr True \
--do_compile True \
--crop_invariance False 