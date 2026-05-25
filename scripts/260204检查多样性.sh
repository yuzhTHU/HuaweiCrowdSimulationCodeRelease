#!/bin/bash
trap 'kill 0' SIGINT  # 捕获 Ctrl+C 并杀掉所有子进程

current_jobs=0
max_jobs=18

# conda activate ./venv
# export CUDA_VISIBLE_DEVICES=0,1,2,3

for k in {1..20}; do
    #python test.py --exp_name "eth_single_K=${k}" --train_datasets hotel zara01 zara02 univ --test_datasets eth --sample_num "${k}" --batch_size 64 --reload_checkpoint ./logs/train/eth_single_K=10/best.pth &
    #((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    #sleep 1

    # python test.py --exp_name "hotel_single_K=${k}" --train_datasets eth zara01 zara02 univ --test_datasets hotel --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/hotel_single_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    # python test.py --exp_name "zara01_single_K=${k}" --train_datasets eth hotel zara02 univ --test_datasets zara01 --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/zara01_single_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    # python test.py --exp_name "zara02_single_K=${k}" --train_datasets eth hotel zara01 univ --test_datasets zara02 --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/zara02_single_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    # python test.py --exp_name "univ_single_K=${k}" --train_datasets eth hotel zara01 zara02 --test_datasets univ --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/univ_single_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    # python test.py --exp_name "eth_unified_K=${k}" --train_datasets hotel zara01 zara02 univ GC_train SDD_train WayMo_train --test_datasets eth --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/eth_unified_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    # python test.py --exp_name "hotel_unified_K=${k}" --train_datasets eth zara01 zara02 univ GC_train SDD_train WayMo_train --test_datasets hotel --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/hotel_unified_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    # python test.py --exp_name "zara01_unified_K=${k}" --train_datasets eth hotel zara02 univ GC_train SDD_train WayMo_train --test_datasets zara01 --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/zara01_unified_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    # python test.py --exp_name "zara02_unified_K=${k}" --train_datasets eth hotel zara01 univ GC_train SDD_train WayMo_train --test_datasets zara02 --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/zara02_unified_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    # python test.py --exp_name "univ_unified_K=${k}" --train_datasets eth hotel zara01 zara02 GC_train SDD_train WayMo_train --test_datasets univ --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/univ_unified_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    # python test.py --exp_name "GC_single_K=${k}" --train_datasets GC_train --test_datasets GC_test --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/GC_single_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    python test.py --exp_name "SDD_single_K=${k}" --train_datasets SDD_train --test_datasets SDD_test --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/SDD_single_K=10/best.pth" &
    ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    sleep 1
    
    # python test.py --exp_name "WayMo_single_K=${k}" --train_datasets WayMo_train --test_datasets WayMo_test --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/WayMo_single_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
    
    # python test.py --exp_name "GC_SDD_WayMo_unified_K=${k}" --train_datasets eth hotel zara01 zara02 univ GC_train SDD_train WayMo_train --test_datasets GC_test SDD_test WayMo_test --sample_num "${k}" --batch_size 64 --reload_checkpoint "./logs/train/GC_SDD_WayMo_unified_K=10/best.pth" &
    # ((++current_jobs >= max_jobs)) && { wait -n; ((current_jobs--)); } # 如果达到最大并发数，等待任一进程结束
done

wait
