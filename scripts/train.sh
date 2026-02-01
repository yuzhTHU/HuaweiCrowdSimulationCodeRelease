# Ours
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "eth_single_K=10" --train_datasets hotel zara01 zara02 univ --test_datasets eth --sample_num 10
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "hotel_single_K=10" --train_datasets eth zara01 zara02 univ --test_datasets hotel --sample_num 10
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "zara01_single_K=10" --train_datasets eth hotel zara02 univ --test_datasets zara01 --sample_num 10
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "zara02_single_K=10" --train_datasets eth hotel zara01 univ --test_datasets zara02 --sample_num 10
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "univ_single_K=10" --train_datasets eth hotel zara01 zara02 --test_datasets univ --sample_num 10
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "eth_unified_K=10" --train_datasets hotel zara01 zara02 univ GC_train SDD_train WayMo_train --test_datasets eth --sample_num 10 --required_memory_MB 10000
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "hotel_unified_K=10" --train_datasets eth zara01 zara02 univ GC_train SDD_train WayMo_train --test_datasets hotel --sample_num 10 --required_memory_MB 10000
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "zara01_unified_K=10" --train_datasets eth hotel zara02 univ GC_train SDD_train WayMo_train --test_datasets zara01 --sample_num 10 --required_memory_MB 10000
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "zara02_unified_K=10" --train_datasets eth hotel zara01 univ GC_train SDD_train WayMo_train --test_datasets zara02 --sample_num 10 --required_memory_MB 10000
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "univ_unified_K=10" --train_datasets eth hotel zara01 zara02 GC_train SDD_train WayMo_train --test_datasets univ --sample_num 10 --required_memory_MB 10000
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "GC_single_K=10" --train_datasets GC_train --test_datasets GC_test --sample_num 10
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "SDD_single_K=10" --train_datasets SDD_train --test_datasets SDD_test --sample_num 10
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "WayMo_single_K=10" --train_datasets WayMo_train --test_datasets WayMo_test --sample_num 10
conda activate ./venv && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train.py --exp_name "GC_SDD_WayMo_unified_K=10" --train_datasets eth hotel zara01 zara02 univ GC_train SDD_train WayMo_train --test_datasets GC_test SDD_test WayMo_test --sample_num 10 --required_memory_MB 10000


conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "eth_single_K=10" --train_datasets hotel zara01 zara02 univ --test_datasets eth --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "hotel_single_K=10" --train_datasets eth zara01 zara02 univ --test_datasets hotel --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "zara01_single_K=10" --train_datasets eth hotel zara02 univ --test_datasets zara01 --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "zara02_single_K=10" --train_datasets eth hotel zara01 univ --test_datasets zara02 --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "univ_single_K=10" --train_datasets eth hotel zara01 zara02 --test_datasets univ --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "GC_single_K=10" --train_datasets GC_train --test_datasets GC_test --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "SDD_single_K=10" --train_datasets SDD_train --test_datasets SDD_test --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "WayMo_single_K=10" --train_datasets WayMo_train --test_datasets WayMo_test --sample_num 10

# conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "eth_unified_K=10" --train_datasets hotel zara01 zara02 univ GC_train SDD_train WayMo_train --test_datasets eth --sample_num 10
# conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "hotel_unified_K=10" --train_datasets eth zara01 zara02 univ GC_train SDD_train WayMo_train --test_datasets hotel --sample_num 10
# conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "zara01_unified_K=10" --train_datasets eth hotel zara02 univ GC_train SDD_train WayMo_train --test_datasets zara01 --sample_num 10
# conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "zara02_unified_K=10" --train_datasets eth hotel zara01 univ GC_train SDD_train WayMo_train --test_datasets zara02 --sample_num 10
# conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "univ_unified_K=10" --train_datasets eth hotel zara01 zara02 GC_train SDD_train WayMo_train --test_datasets univ --sample_num 10
# conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_social_stgcnn.py --minimize_gpu --exp_name "GC_SDD_WayMo_unified_K=10" --train_datasets eth hotel zara01 zara02 univ GC_train SDD_train WayMo_train --test_datasets GC_test SDD_test WayMo_test --sample_num 10


conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_spdiff.py --minimize_gpu --exp_name "eth_single_K=10" --train_datasets hotel zara01 zara02 univ --test_datasets eth --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_spdiff.py --minimize_gpu --exp_name "hotel_single_K=10" --train_datasets eth zara01 zara02 univ --test_datasets hotel --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_spdiff.py --minimize_gpu --exp_name "zara01_single_K=10" --train_datasets eth hotel zara02 univ --test_datasets zara01 --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_spdiff.py --minimize_gpu --exp_name "zara02_single_K=10" --train_datasets eth hotel zara01 univ --test_datasets zara02 --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_spdiff.py --minimize_gpu --exp_name "univ_single_K=10" --train_datasets eth hotel zara01 zara02 --test_datasets univ --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_spdiff.py --minimize_gpu --exp_name "GC_single_K=10" --train_datasets GC_train --test_datasets GC_test --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_spdiff.py --minimize_gpu --exp_name "SDD_single_K=10" --train_datasets SDD_train --test_datasets SDD_test --sample_num 10
conda activate ./venv && export PYTHONPATH=. && python ./baselines/train_spdiff.py --minimize_gpu --exp_name "WayMo_single_K=10" --train_datasets WayMo_train --test_datasets WayMo_test --sample_num 10
