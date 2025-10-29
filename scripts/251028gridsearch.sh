python train.py \
    --test_before_train \
    --name Baseline \
    --datasets All \
    --test_ratio 0.2 

python train.py \
    --name BS64 \
    --datasets All \
    --test_ratio 0.2 \
    --batch_size 64 --required_memory_MB 12000

python train.py \
    --name BS128 \
    --datasets All \
    --test_ratio 0.2 \
    --batch_size 128 --required_memory_MB 23000

python train.py \
    --name lr5e-3 \
    --datasets All \
    --test_ratio 0.2 \
    --lr 5e-3 

python train.py \
    --name lr2e-4 \
    --datasets All \
    --test_ratio 0.2 \
    --lr 2e-4 

python train.py \
    --name sample10 \
    --datasets All \
    --test_ratio 0.2 \
    --sample_num 10 

python train.py \
    --name sample5 \
    --datasets All \
    --test_ratio 0.2 \
    --sample_num 5 

python train.py \
    --name sample1 \
    --datasets All \
    --test_ratio 0.2 \
    --sample_num 1

python train.py \
    --name denoise5 \
    --datasets All \
    --test_ratio 0.2 \
    --denoise_step 5 

python train.py \
    --name denoise2 \
    --datasets All \
    --test_ratio 0.2 \
    --denoise_step 2 

python train.py \
    --name offset10 \
    --datasets All \
    --test_ratio 0.2 \
    --step_offset 10 

python train.py \
    --name scale10 \
    --datasets All \
    --test_ratio 0.2 \
    --scale_Accelerate 10.0

python train.py \
    --name scale0.1 \
    --datasets All \
    --test_ratio 0.2 \
    --scale_Accelerate 0.1

python train.py \
    --name no_antithetic \
    --datasets All \
    --test_ratio 0.2 \
    --no_antithetic_sampling
    
python train.py \
    --name no_antithetic \
    --datasets All \
    --test_ratio 0.2 \
    --no_antithetic_sampling

python train.py \
    --name LSTM2 \
    --datasets All \
    --test_ratio 0.2 \
    --lstm_layer_num 2

python train.py \
    --name LSTM1 \
    --datasets All \
    --test_ratio 0.2 \
    --lstm_layer_num 1

python train.py \
    --name drop0.5 \
    --datasets All \
    --test_ratio 0.2 \
    --dropout 0.5

python train.py \
    --name drop0.1 \
    --datasets All \
    --test_ratio 0.2 \
    --dropout 0.1

python train.py \
    --name dim64 \
    --datasets All \
    --test_ratio 0.2 \
    --model_dim 64

python train.py \
    --name dim256 \
    --datasets All \
    --test_ratio 0.2 \
    --model_dim 256 \
    --required_memory_MB 11000

python train.py \
    --name predict-acc \
    --datasets All \
    --test_ratio 0.2 \
    --no-predict_noise \
    --loss_type accelerate \