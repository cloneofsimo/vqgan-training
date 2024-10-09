loglrs=(-8 -7 -6 -5 -4 -3 -2)
MODEL_WIDTHS=(64 128 256)
for loglr in "${loglrs[@]}"; do
    for width in "${MODEL_WIDTHS[@]}"; do
        lr=$(python -c "import math; print(2**${loglr})")
        run_name="exp_vae_ch_${width}_lr_${lr}"

        echo "Running ${run_name}"
        
        torchrun --nproc_per_node=8 vae_trainer.py \
        --learning_rate_vae ${lr} \
        --vae_ch ${width} \
        --run_name ${run_name} \
        --num_epochs 1
        
    done
done












    