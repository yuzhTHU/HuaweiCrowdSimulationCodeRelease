python train.py \
    --name ETH/UCY_0.2test \
    --datasets ETH/UCY \
    --test_ratio 0.2 \
    --test_before_train \
    --device auto \
    --batch_size 256 \
    --denoise_step 5
python train.py \
    --name GC_0.2test \
    --datasets GC \
    --test_ratio 0.2 \
    --test_before_train \
    --device auto \
    --batch_size 256 \
    --denoise_step 5
python train.py \
    --name SDD_0.2Stest \
    --datasets SDD \
    --test_ratio 0.2 \
    --split_by_scenario \
    --test_before_train \
    --device auto \
    --batch_size 256 \
    --denoise_step 5
python train.py \
    --name WayMo_0.2Stest \
    --datasets WayMo \
    --test_ratio 0.2 \
    --split_by_scenario \
    --test_before_train \
    --device auto \
    --batch_size 32 \
    --dot_per_meter 1 \
    --denoise_step 5
