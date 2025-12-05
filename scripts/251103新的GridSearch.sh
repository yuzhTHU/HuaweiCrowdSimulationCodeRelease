python train.py \
    --name new-1 \
    --datasets All \
    --test_ratio 0.2 \
    --lstm_layer_num 1 \
    --denoise_step 2 \
    --lr 2e-4 \
    --dropout 0.5 \
    --model_dim 64 \
    --step_offset 10 \
    --batch_size 64 \
    --sample_num 20

python train.py \
    --name new-1-lr5e-4 \
    --lr 5e-4 \
    --batch_size 64 \
    --datasets All \
    --test_ratio 0.2 \
    --lstm_layer_num 1 \
    --denoise_step 2 \
    --dropout 0.5 \
    --model_dim 64 \
    --step_offset 10 \
    --sample_num 20

python train.py \
    --name new-1-BS128 \
    --lr 2e-4 \
    --batch_size 128 \
    --datasets All \
    --test_ratio 0.2 \
    --lstm_layer_num 1 \
    --denoise_step 2 \
    --dropout 0.5 \
    --model_dim 64 \
    --step_offset 10 \
    --sample_num 20

python train.py \
    --name new-1-CFG \
    --lr 2e-4 \
    --batch_size 128 \
    --datasets All \
    --test_ratio 0.2 \
    --lstm_layer_num 1 \
    --denoise_step 2 \
    --dropout 0.5 \
    --model_dim 64 \
    --step_offset 10 \
    --sample_num 20 \
    --p_drop_map 0.2 \
    --p_drop_destination 0.3 \
    --p_drop_speed 0.3
