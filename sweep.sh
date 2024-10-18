## Sweep 1. is attention useful?

loglrs=(-8 -7 -6 -5 -4 -3 -2)
MODEL_WIDTHS=(32 64 128)

for loglr in "${loglrs[@]}"; do
    for attn in "True" "False"; do
        for width in "${MODEL_WIDTHS[@]}"; do
            lr=$(python -c "import math; print(2**${loglr})")
            run_name="exp_vae_ch_${width}_lr_${lr}_attn_${attn}"

            echo "Running ${run_name}"
            
            torchrun --nproc_per_node=8 vae_trainer.py \
            --learning_rate_vae ${lr} \
            --vae_ch ${width} \
            --run_name ${run_name} \
            --num_epochs 20 \
            --max_steps 2000 \
            --evaluate_every_n_steps 250 \
            --batch_size 32 \
            --do_clamp \
            --do_attn ${attn} \
            --project_name "vae_sweep_attn_lr_width"

        done
    done
done

## Sweep 2. Can we initialize better?

loglrs=(-8 -7 -6 -5 -4 -3 -2)
MODEL_WIDTHS=(64)

for loglr in "${loglrs[@]}"; do
    for attn in "True" "False"; do
        for width in "${MODEL_WIDTHS[@]}"; do
            lr=$(python -c "import math; print(2**${loglr})")
            run_name="exp_vae_ch_${width}_lr_${lr}_attn_${attn}"

            echo "Running ${run_name}"
            
            torchrun --nproc_per_node=8 vae_trainer.py \
            --learning_rate_vae ${lr} \
            --vae_ch ${width} \
            --run_name ${run_name} \
            --num_epochs 20 \
            --max_steps 2000 \
            --evaluate_every_n_steps 250 \
            --batch_size 32 \
            --do_clamp \
            --do_attn ${attn} \
            --project_name "vae_sweep_attn_lr_width"

        done
    done
done









    