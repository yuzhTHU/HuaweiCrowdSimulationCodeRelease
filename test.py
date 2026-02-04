import os
import re
import sys
import json
import torch
import shlex
import random
import logging
import numpy as np
import torch.utils.data as D
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from src.dataset import ETHDataset, UCYDataset, SDDDataset, GCDataset, WayMoDataset, ORCADataset
from src.model import Model, RelativeModel, NewModel
from src.diffusion import DDPM, DDIM
from src.utils.logger import init_logger
from src.utils.seed import seed_all
from src.utils.timer import NamedTimer
from src.utils.auto_gpu import AutoGPU
from src.utils.fix_parser import add_negation_flags, add_minus_flags
from src.utils.tag2ansi import tag2ansi
from src.utils.use_npu import USE_NPU, npu_attention_fallback_context
from src.tasks import test_once

_logger = logging.getLogger("src.test")

def main(args):
    ## Load Dataset
    if args.test_datasets:
        train_dataset = []
        test_dataset = []
        for (datasets, dataset_names) in zip((train_dataset, test_dataset), (args.train_datasets, args.test_datasets)):
            for dataset in dataset_names:
                if dataset == 'eth':
                    datasets.append(ETHDataset.load_data(args, "./data/ETH/seq_eth/obsmat.txt"))
                elif dataset == 'hotel':
                    datasets.append(ETHDataset.load_data(args, "./data/ETH/seq_hotel/obsmat.txt"))
                elif dataset == 'zara01':
                    datasets.append(UCYDataset.load_data(args, './data/UCY/data/data_zara/crowds_zara01.vsp'))
                elif dataset == 'zara02':
                    datasets.append(UCYDataset.load_data(args, './data/UCY/data/data_zara/crowds_zara02.vsp'))
                elif dataset == 'univ':
                    datasets.append(UCYDataset.load_data(args, './data/UCY/data/data_university_students/students003.vsp'))
                elif dataset == 'GC_train':
                    _gc_dataset = GCDataset.load_data(args, "./data/GC/Annotation") if '_gc_dataset' not in locals() else _gc_dataset
                    d = deepcopy(_gc_dataset)
                    test_ratio = 0.2
                    train_num = int(len(_gc_dataset) * (1-test_ratio))
                    d.name = d.name + f"_{100-test_ratio*100:.0f}train"
                    d.samples = d.samples[:train_num]
                    datasets.append(d)
                elif dataset == 'GC_test':
                    _gc_dataset = GCDataset.load_data(args, "./data/GC/Annotation") if '_gc_dataset' not in locals() else _gc_dataset
                    d = deepcopy(_gc_dataset)
                    test_ratio = 0.2
                    train_num = int(len(_gc_dataset) * (1-test_ratio))
                    d.name = d.name + f"_{test_ratio*100:.0f}test"
                    d.samples = d.samples[train_num:]
                    datasets.append(d)
                elif dataset == 'SDD_train':
                    for path in ['bookstore/video0','bookstore/video1','bookstore/video2','bookstore/video3','coupa/video0','coupa/video1','coupa/video2','deathCircle/video0','deathCircle/video1','deathCircle/video2','deathCircle/video3','gates/video0','gates/video1','gates/video2','gates/video3','gates/video4','gates/video5','gates/video6','gates/video7','hyang/video0','hyang/video1','hyang/video2','hyang/video3','hyang/video4','hyang/video5','hyang/video6','hyang/video7','hyang/video8','hyang/video9','hyang/video10','hyang/video11','hyang/video12','hyang/video13','little/video0','little/video1','little/video2','nexus/video0','nexus/video1','nexus/video2','nexus/video3','nexus/video4','nexus/video5','nexus/video6','nexus/video7','nexus/video8','quad/video0','quad/video1','quad/video2']:
                        datasets.append(SDDDataset.load_data(args, f"./data/SDD/annotations/{path}/annotations.txt"))
                elif dataset == 'SDD_test':
                    for path in ['bookstore/video4','bookstore/video5','bookstore/video6','coupa/video3','deathCircle/video4','gates/video8','hyang/video14','little/video3','nexus/video9','nexus/video10','nexus/video11','quad/video3']:
                        datasets.append(SDDDataset.load_data(args, f"./data/SDD/annotations/{path}/annotations.txt"))
                elif dataset == 'WayMo_train':
                    _waymo_datasets = WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/", total=500) if '_waymo_datasets' not in locals() else _waymo_datasets
                    for d in _waymo_datasets:
                        d = deepcopy(d)
                        test_ratio = 0.2
                        train_num = int(len(d) * (1-test_ratio))
                        d.name = d.name + f"_{100-test_ratio*100:.0f}train"
                        d.samples = d.samples[:train_num]
                        datasets.append(d)
                elif dataset == 'WayMo_test':
                    _waymo_datasets = WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/", total=500) if '_waymo_datasets' not in locals() else _waymo_datasets
                    for d in _waymo_datasets:
                        d = deepcopy(d)
                        test_ratio = 0.2
                        train_num = int(len(d) * (1-test_ratio))
                        d.name = d.name + f"_{test_ratio*100:.0f}test"
                        d.samples = d.samples[train_num:]
                        datasets.append(d)
                else:
                    raise ValueError(f"Unknown dataset {dataset}!")

        # 将每个场景的特定比例样本划分到测试集
        dataset_list = train_dataset
        train_dataset = []
        eval_dataset = []
        for d in dataset_list:
            d1 = d
            d2 = deepcopy(d)
            train_num = int(len(d) * (1-args.eval_ratio))
            d1.name = d1.name + f"_{100-args.eval_ratio*100:.0f}train"
            d2.name = d2.name + f"_{args.eval_ratio*100:.0f}eval"
            d1.samples = d1.samples[:train_num]
            d2.samples = d2.samples[train_num:]
            train_dataset.append(d1)
            eval_dataset.append(d2)
    else:
        # 加载数据集
        dataset_list = []
        if 'All' in args.datasets:
            args.datasets.remove('All')
            args.datasets += ['ETH', 'UCY', 'GC', 'SDD', 'WayMo']
        if "ETH" in args.datasets:
            dataset_list += ETHDataset.load_data_batch(args, "./data/ETH/")
            args.datasets.remove("ETH")
        if "UCY" in args.datasets:
            dataset_list += UCYDataset.load_data_batch(args, "./data/UCY/data/")
            args.datasets.remove("UCY")
        if "GC" in args.datasets:
            dataset_list += [GCDataset.load_data(args, "./data/GC/Annotation")]
            args.datasets.remove("GC")
        if "SDD" in args.datasets:
            dataset_list += SDDDataset.load_data_batch(args, "./data/SDD/annotations/")
            args.datasets.remove("SDD")
        if 'WayMo' in args.datasets:
            dataset_list += WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/", total=500)
            args.datasets.remove("WayMo")
        if 'ORCA' in args.datasets:
            dataset_list += ORCADataset.load_data_batch(args, "./data/ORCA/")
            args.datasets.remove("ORCA")
        if 'debug' in args.datasets:
            # dataset_list += [UCYDataset.load_data(args, './data/UCY/data/data_university_students/students003.vsp')]
            # dataset_list += [SDDDataset.load_data(args, "./data/SDD/annotations/hyang/video0/annotations.txt")]
            dataset_list += [WayMoDataset.load_data(args, './data/WayMo/Processed/00000_1_2aa43fad083efbf3/data.csv.gz')]
            # dataset_list[0].samples = dataset_list[0].samples[int(len(dataset_list[0].samples) * 0.8):]
            args.datasets.remove('debug')
        if len(args.datasets) > 0:
            raise ValueError(f"Unknown datase: {args.datasets}!")
        # 检查地图
        for dataset in dataset_list:
            map_data = dataset.map_data
            delta_x = map_data.xmax - map_data.xmin
            delta_y = map_data.ymax - map_data.ymin
            w, h = map_data.map.shape
            if not (0.8 < (ratio := (delta_x / w) / (delta_y / h)) < 1.2):
                _logger.warning(
                    f"Map aspect ratio of {dataset.name} mismatch: "
                    f"data ratio={ratio:.4f} (xrange={delta_x:.4f}, yrange={delta_y:.4f}, "
                    f"map shape={map_data.map.shape}), may cause distortion."
                )
                exit(1)
        # 划分训练集和测试集
        if args.test_name is not None:
            # 将名称中包含指定字符串的场景划分到测试集
            train_dataset = []
            test_dataset = []
            for d in dataset_list:
                if any(test_name in d.name for test_name in args.test_name):
                    test_dataset.append(d)
                else:
                    train_dataset.append(d)
        elif args.test_ratio is not None and args.split_by_scenario:
            # 将特定比例的场景划分到测试集
            random.shuffle(dataset_list)
            test_size = max(1, int(len(dataset_list) * args.test_ratio))
            train_dataset = dataset_list[:-test_size]
            test_dataset = dataset_list[-test_size:]
        elif args.test_ratio is not None and not args.split_by_scenario:
            # 将每个场景的特定比例样本划分到测试集
            train_dataset = []
            test_dataset = []
            for d in dataset_list:
                d1 = d
                d2 = deepcopy(d)
                train_num = int(len(d) * (1-args.test_ratio))
                d1.name = d1.name + f"_{100-args.test_ratio*100:.0f}train"
                d2.name = d2.name + f"_{args.test_ratio*100:.0f}test"
                d1.samples = d1.samples[:train_num]
                d2.samples = d2.samples[train_num:]
                train_dataset.append(d1)
                test_dataset.append(d2)
        else:
            # 训练集和测试集相同
            train_dataset = dataset_list
            test_dataset = dataset_list
        eval_dataset = test_dataset
    # 创建数据加载器
    train_loaders = []
    eval_loaders = []
    test_loaders = []
    for dataset in train_dataset:
        if len(dataset) == 0:
            _logger.warning(f"Dataset {dataset.name} has no training samples!")
            continue
        train_loaders.append(D.DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    for dataset in eval_dataset:
        if len(dataset) == 0:
            _logger.warning(f"Dataset {dataset.name} has no evaluation samples!")
            continue
        eval_loaders.append(D.DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size // args.sample_num,  # 在实际测试时 batch_size 会乘上 sample_num，可能会很大导致 OOM
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    for dataset in test_dataset:
        if len(dataset) == 0:
            _logger.warning(f"Dataset {dataset.name} has no testing samples!")
            continue
        test_loaders.append(D.DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size // args.sample_num,  # 在实际测试时 batch_size 会乘上 sample_num，可能会很大导致 OOM
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    _logger.note(
        "Datasets:\n"
        # f"Train on {[d.name for d in train_dataset]} datasets ({sum([len(d) for d in train_dataset]):,} samples in total)\n"
        # f"Eval on {[d.name for d in eval_dataset]} datasets ({sum([len(d) for d in eval_dataset]):,} samples in total)\n"
        f"Test on {[d.name for d in test_dataset]} datasets ({sum([len(d) for d in test_dataset]):,} samples in total)"
    )

    ## Load Model
    if args.use_new_model:
        model = NewModel(args).to(args.device)
    elif args.use_relative_model:
        model = RelativeModel(args).to(args.device)
    else:
        model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    if args.sampling_method == "DDIM":
        diffusion = DDIM(args)
    elif args.sampling_method == "DDPM":
        diffusion = DDPM(args, flexibility=0.0)
    else:
        raise ValueError(f"Unknown sampling_method {args.sampling_method}!")
    _logger.note(
        "Model Parameters:\n"
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        f"Total: {sum(p.numel() for p in model.parameters()):,}"
    )

    ## Reload Checkpoint
    if args.force_new_experiment:
        checkpoint_path = None
    elif args.reload_checkpoint is not None:
        # 如果指定了 checkpoint 路径，则从该路径加载
        checkpoint_path = Path(args.reload_checkpoint)
    elif (Path(args.save_path) / "checkpoint.pth").exists():
        # 如果当前保存路径下存在 checkpoint，则从该路径加载
        checkpoint_path = Path(args.save_path) / "checkpoint.pth"
    else:
        checkpoint_path = None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        start_epoch = checkpoint["epoch"] + 1
        if 'args' in checkpoint:
            saved_args = checkpoint['args']
            for key in sorted(set(saved_args.keys()) | set(vars(args).keys())):
                if key in [
                    'device', 'save_dir', 'save_path', 'command', 'name', 'exp_name',
                    'seed', 'reload_checkpoint', 'test_before_train', 'test_per_epoch',
                    'p_drop_destination', 'p_drop_map', 'p_drop_speed',
                ]:
                    continue
                val1 = saved_args.get(key, None)
                val2 = getattr(args, key, None)
                if val1 != val2:
                    _logger.warning(
                        f"Argument '{key}' differs from the saved checkpoint: "
                        f"saved_args={val1} vs. current_args={val2}"
                    )
        model.load_state_dict(checkpoint["model"])
        _logger.note(tag2ansi(f"Checkpoint loaded from [underline green]{checkpoint_path}[reset], resume from epoch [underline green]{start_epoch}[reset]."))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            _logger.warning("Optimizer state not found in checkpoint, optimizer re-initialized.")
    else:
        start_epoch = 0
        raise ValueError("No checkpoint found to load!")

    ## Test
    torch.set_grad_enabled(False)
    model.eval()
    with npu_attention_fallback_context(model, enable=USE_NPU):
        test_records = test_once(args, test_loaders, model, criterion, diffusion, start_epoch)
    with open(f"{args.save_path}/records.jsonl", "a") as f:
        if test_records is not None:
            f.write(json.dumps(test_records) + "\n")

    ## Log Test Result
    w = np.array(test_records['sample_nums'], dtype=float)
    w /= w.sum()
    test_records['accuracy'] = 1 - np.sum(w * test_records['ade']) / np.sum(w * test_records['trajlen'])
    test_records['unweighted_accuracy'] = 1 - np.mean(test_records['ade']) / np.mean(test_records['trajlen'])
    _logger.note(tag2ansi(
        f"[bold underline orange]Test Accuracy={test_records['accuracy']:.2%}[reset] (unweighted={test_records['unweighted_accuracy']:.2%}), "
        f"at [#66CCFF]epoch {test_records['epoch']}[reset]. "
        f"[#66CCFF]ADE={np.sum(w * test_records['ade']):.4f}, "
        f"[#66CCFF]FDE={np.sum(w * test_records['fde']):.4f}, "
        f"[#66CCFF]X_ERROR (normal)={np.nansum(w * test_records['norm_err']) / np.sum(w * np.isfinite(test_records['norm_err'])):.4f}, "
        f"[#66CCFF]Y_ERROR (tangential)={np.nansum(w * test_records['tan_err']) / np.sum(w * np.isfinite(test_records['tan_err'])):.4f}, "
        f"[#66CCFF]Collision-Ped={np.sum(w * test_records['collision_ped']) - (base := np.sum(w * test_records['collision_ped_base'])):.2%} (+{base:.2%}), "
        f"[#66CCFF]Collision-Veh={np.sum(w * test_records['collision_veh']) - (base := np.sum(w * test_records['collision_veh_base'])):.2%} (+{base:.2%}), "
        f"[#66CCFF]Collision-Map={np.sum(w * test_records['collision_map']) - (base := np.sum(w * test_records['collision_map_base'])):.2%} (+{base:.2%}), "
        f"[#66CCFF]APD={np.sum(w * test_records['apd']):.4f}, "
        f"[#66CCFF]AvgLen={np.sum(w * test_records['trajlen']):.4f}, "
        f"[#66CCFF]Loss={np.sum(w * test_records['loss']):.4f}, "
        f"[#66CCFF]PedNum={np.sum(w * test_records['ped_num']):.1f}, "
        f"[#66CCFF]VehNum={np.sum(w * test_records['veh_num']):.1f}, "
        f"[#66CCFF]RolloutTime={np.mean(test_records['rollout_time'])*1000:.2f}ms "
        f"([bold underline orange]FPS={1/np.mean(test_records['rollout_time']):.2f} Hz[reset])"
    ))
    if len(set(test_records['dataset_class'])) > 1:
        for klass in sorted(list(set(test_records['dataset_class']))):
            idxs = [i for i, k in enumerate(test_records['dataset_class']) if k == klass]
            ade = np.array([test_records['ade'][i] for i in idxs])
            fde = np.array([test_records['fde'][i] for i in idxs])
            trajlen = np.array([test_records['trajlen'][i] for i in idxs])
            ped_num = np.array([test_records['ped_num'][i] for i in idxs])
            veh_num = np.array([test_records['veh_num'][i] for i in idxs])
            norm_err = np.array([test_records['norm_err'][i] for i in idxs])
            tan_err = np.array([test_records['tan_err'][i] for i in idxs])
            collision_ped = np.array([test_records['collision_ped'][i] for i in idxs])
            collision_veh = np.array([test_records['collision_veh'][i] for i in idxs])
            collision_map = np.array([test_records['collision_map'][i] for i in idxs])
            collision_ped_base = np.array([test_records['collision_ped_base'][i] for i in idxs])
            collision_veh_base = np.array([test_records['collision_veh_base'][i] for i in idxs])
            collision_map_base = np.array([test_records['collision_map_base'][i] for i in idxs])
            apd = np.array([test_records['apd'][i] for i in idxs])
            rollout_time = np.array([test_records['rollout_time'][i] for i in idxs])
            w = np.array([test_records['sample_nums'][i] for i in idxs], dtype=float)
            w /= w.sum()
            acc = 1 - np.sum(w * ade) / np.sum(w * trajlen)
            _logger.info(tag2ansi(
                f"[#66CCFF]Overall on {klass} datasets: "
                f"[bold underline orange]Accuracy={acc:.2%}[reset], "
                f"[#66CCFF]ADE={np.sum(w * ade):.4f}, "
                f"[#66CCFF]FDE={np.sum(w * fde):.4f}, "
                f"[#66CCFF]X_ERROR (normal)={np.nansum(w * norm_err) / np.sum(w * np.isfinite(norm_err)):.4f}, "
                f"[#66CCFF]Y_ERROR (tangential)={np.nansum(w * tan_err) / np.sum(w * np.isfinite(tan_err)):.4f}, "
                f"[#66CCFF]Collision-Ped={np.sum(w * collision_ped) - (base := np.sum(w * collision_ped)):.2%} (+{base:.2%}), "
                f"[#66CCFF]Collision-Veh={np.sum(w * collision_veh) - (base := np.sum(w * collision_veh)):.2%} (+{base:.2%}), "
                f"[#66CCFF]Collision-Map={np.sum(w * collision_map) - (base := np.sum(w * collision_map)):.2%} (+{base:.2%}), "
                f"[#66CCFF]APD={np.sum(w * apd):.4f}, "
                f"[#66CCFF]AvgLen={np.sum(w * trajlen):.4f}, "
                f"[#66CCFF]PedNum={np.sum(w * ped_num):.4f}, "
                f"[#66CCFF]VehNum={np.sum(w * veh_num):.4f}, "
                f"[#66CCFF]RolloutTime={np.mean(rollout_time)*1000:.2f}ms "
                f"([bold underline orange]FPS={1/np.mean(rollout_time):.2f} Hz[reset])"
            ))

    _logger.note(f"Testing finished. Re-run: {args.command}")


if __name__ == "__main__":
    parser = ArgumentParser()
    # 基础配置
    parser.add_argument("--name", type=str, default="test", help="实验任务名称，用于生成实验ID")
    parser.add_argument("--exp_name", type=str, default=None, help="手动指定实验名称（若指定则覆盖自动生成的名称）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备，可选 'cpu', 'cuda:0' 或 'auto'（自动选择显存充足的 GPU）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，固定以复现实验结果")
    parser.add_argument("--save_dir", type=str, default="./logs/test", help="日志和模型权重的保存根目录")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式（输出更多日志，不保存部分文件）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的工作线程数（0 表示主线程）")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=128, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率 (Learning Rate)")
    parser.add_argument("--epochs", type=int, default=10000, help="最大训练轮数")
    parser.add_argument('--patience', type=int, default=20, help="Early Stopping 的耐心值（多少个 epoch 验证集指标不提升则停止）")
    parser.add_argument('--loss_type', type=str, default='noise', choices=['position', 'accelerate', 'noise'], help="损失函数计算的目标类型")
    parser.add_argument('--reload_checkpoint', type=str, default=None, help="用于测试的 checkpoint（.pth 文件）")
    parser.add_argument('--force_new_experiment', action='store_true', help="是否强制不使用 checkpoint 继续训练，即使存在 checkpoint 文件")
    parser.add_argument('--required_memory_MB', type=int, default=5000, help="自动选择 GPU 时要求的最小剩余显存 (MB)")

    # 扩散模型参数 (Diffusion)
    parser.add_argument('--sampling_method', type=str, default="DDIM", choices=['DDPM', 'DDIM'], help="采样/生成方法")
    parser.add_argument("--T", type=int, default=100, help="训练时的最大扩散步数 (Timesteps)")
    parser.add_argument('--sample_num', type=int, default=20, help="测试推理时，为每个轨迹生成的样本数量（用于评估多样性和准确性）")
    parser.add_argument('--denoise_step', type=int, default=2, help="DDIM 采样时的去噪步数（加速采样）")
    parser.add_argument('--step_offset', type=int, default=10, help="采样的起始时间步偏移量（从 T-offset 开始采样）")
    parser.add_argument('--antithetic_sampling', action='store_true', default=True, help="是否使用对偶采样以减少方差")
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'], help="噪声调度表类型")
    parser.add_argument('--predict_noise', action='store_true', default=True, help="模型是否预测噪声（True预测epsilon, False预测x0）")
    parser.add_argument('--rollout_lambda', type=float, default=1.0, help="多帧 Rollout 损失的时间衰减系数")
    parser.add_argument('--multi_frame_rollout', type=int, default=1, help="训练时单次迭代预测未来的帧数（Rollout 步数）")
    parser.add_argument('--scale_accelerate', type=float, default=1.0, help="加速度数据的缩放因子（用于稳定训练）")

    # 数据增强与Dropout
    parser.add_argument('--p_drop_map', type=float, default=None, help="以一定概率丢弃地图信息（实现无地图引导的生成）")
    parser.add_argument('--p_drop_destination', type=float, default=None, help="以一定概率丢弃目的地条件（实现无目标引导的生成）")
    parser.add_argument('--p_drop_speed', type=float, default=None, help="以一定概率丢弃初始速度条件")
    parser.add_argument('--dropout', type=float, default=0.5, help="模型中的 Dropout 比率")

    # 数据集配置
    parser.add_argument('--train_datasets', type=str, default=[], nargs='*', choices=['eth', 'hotel', 'zara01', 'zara02', 'univ', 'GC_train', 'SDD_train', 'WayMo_train'], help="使用的训练数据集列表 (KDD)")
    parser.add_argument('--test_datasets', type=str, default=[], nargs='*', choices=['eth', 'hotel', 'zara01', 'zara02', 'univ', 'GC_test', 'SDD_test', 'WayMo_test'], help="使用的训练数据集列表 (KDD)")
    parser.add_argument('--eval_ratio', type=float, default=0.2, help="训练集划分为训练/验证集的比例 (KDD)")
    parser.add_argument('--datasets', type=str, default=["ETH"], nargs='*', choices=['ETH', 'UCY', 'GC', 'SDD', 'WayMo', 'ORCA', 'All', 'debug'], help="使用的训练数据集列表")
    parser.add_argument("--hist_step", type=int, default=8, help="输入的历史轨迹长度（帧数）")
    parser.add_argument("--pred_step", type=int, default=1, help="单步预测的未来轨迹长度（帧数，通常配合 Rollout 使用）")
    parser.add_argument("--skip_step", type=int, default=1, help="数据采样的滑动窗口步长")
    parser.add_argument("--roll_step", type=int, default=12, help="测试/验证时需要预测的总未来帧数")
    parser.add_argument("--fps", type=int, default=2.5, help="数据重采样后的目标帧率 (Hz)")
    parser.add_argument("--dot_per_meter", type=int, default=1, help="栅格化地图的分辨率（每米对应的像素点数）")
    parser.add_argument('--test_name', type=str, default=None, nargs='+', help="指定作为测试集的场景名称（substring匹配）")
    parser.add_argument('--test_ratio', type=float, default=None, help="自动划分测试集的比例 (0.0 ~ 1.0)")
    parser.add_argument('--split_by_scenario', action='store_true', help="是否按场景划分训练/测试集（否则按轨迹样本划分）")
    parser.add_argument('--cache_dataset', action='store_true', default=True, help="是否缓存预处理后的数据集以加速加载")
    parser.add_argument('--test_before_train', action='store_true', help="是否在训练开始前先运行一次测试")
    parser.add_argument('--test_per_epoch', type=int, default=10, help="每隔多少个 epoch 运行一次测试")
    parser.add_argument('--collision_threshold', type=float, default=0.6, help="碰撞检测的距离阈值（单位：米）")

    # 模型结构参数
    parser.add_argument('--model_dim', type=int, default=64, help="模型的隐藏层维度 (Hidden Dimension)")
    parser.add_argument('--map_feature_dim', type=int, default=64, help="地图特征提取网络的输出维度")
    parser.add_argument('--head_num', type=int, default=4, help="Transformer 注意力头的数量")
    parser.add_argument('--attention_layer_num', type=int, default=1, help="Transformer 层的堆叠数量")
    parser.add_argument('--lstm_layer_num', type=int, default=1, help="LSTM 层的堆叠数量")
    parser.add_argument('--latent_token_num', type=int, default=16, help="地图特征的 Latent Token 数量")
    parser.add_argument('--use_relative_model', action='store_true', default=True, help="是否使用相对坐标模型结构")
    parser.add_argument('--use_spatial_anchor', action='store_true', default=True, help="是否使用空间锚点增强位置编码")
    parser.add_argument('--use_new_model', action='store_true', default=False, help="是否使用改进版的新模型结构")
    parser.add_argument('--use_nan_embedding', action='store_true', default=True, help="是否使用可学习的空值嵌入")

    # 条件引导参数
    parser.add_argument('--cfg_des', type=float, default=None, help="通过 Classifier-Free Guidance 引导控制目的地条件的影响强度")
    parser.add_argument('--cfg_map', type=float, default=None, help="通过 Classifier-Free Guidance 引导控制地图条件的影响强度")
    parser.add_argument('--cg_dir', type=float, default=None, help="通过 Classifier Guidance 引导控制向目的地前进的影响强度")
    parser.add_argument('--cg_dis', type=float, default=None, help="通过 Classifier Guidance 引导控制与目的地距离的影响强度")
    parser.add_argument('--cg_sfm_des', type=float, default=None, help="通过 Classifier Guidance 引导控制社会力目标引导条件的影响强度")
    parser.add_argument('--cg_sfm_obs', type=float, default=None, help="通过 Classifier Guidance 引导控制社会力地图排斥条件的影响强度")
    parser.add_argument('--cg_sfm_soc', type=float, default=None, help="通过 Classifier Guidance 引导控制社会力社交排斥条件的影响强度")
    parser.add_argument('--sfm_t_des', type=float, default=0.5, help="社会力中目标引导力的弛豫时间")
    parser.add_argument('--sfm_a_ped', type=float, default=25, help="社会力中行人排斥力的强度系数")
    parser.add_argument('--sfm_a_veh', type=float, default=30, help="社会力中车辆排斥力的强度系数")
    parser.add_argument('--sfm_a_map', type=float, default=30, help="社会力中地图排斥力的强度系数")
    parser.add_argument('--sfm_b_ped', type=float, default=0.08, help="社会力中行人排斥力的衰减系数")
    parser.add_argument('--sfm_b_veh', type=float, default=0.10, help="社会力中车辆排斥力的衰减系数")
    parser.add_argument('--sfm_b_map', type=float, default=0.10, help="社会力中地图排斥力的衰减系数")
    parser.add_argument('--sfm_r_map', type=int, default=10, help="社会力中地图排斥力距离阈值 (in pixel)")
    parser.add_argument('--sfm_a_damp', type=float, default=0.5, help="社会力中速度阻尼系数（用于计算引导力时的速度衰减）")
    parser.add_argument('--use_sfm', action='store_true', default=False, help="使用社会力模型代替神经网络计算引导力")

    parser = add_minus_flags(parser) ## --key_name -> --key-name
    parser = add_negation_flags(parser) ## --action-as-true -> --no-action-as-true
    args, unknown = parser.parse_known_args()

    ## Build Save Path
    if args.exp_name is None:
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        curr = now.strftime("%H%M%S")
        host = gethostname()
        exp_name = f'{date}_{args.name}_{curr}_{host}'
        exp_name = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub('_', exp_name.strip())
        exp_name = exp_name or 'unnamed'
        exp_name = exp_name[:255] # Max filename length on most filesystems
        args.exp_name = exp_name
    save_path = Path(args.save_dir) / args.exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    ## Init Logger
    init_logger(
        "src",
        exp_name=args.exp_name,
        log_file=save_path / "info.log",
        info_level="debug" if args.debug else "info",
    )

    ## Warm Unknown Args
    if unknown:
        _logger.warning(f"Unknown args: {unknown}")

    ## Set Seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed_all(args.seed)
    ## Set Command
    args.command = ' '.join(map(shlex.quote, [sys.executable, *sys.argv]))
    ## Select GPU
    if args.device == "auto":
        args.device = AutoGPU().choice_gpu(memory_MB=args.required_memory_MB, interval=15) if not USE_NPU else 'npu'

    ## Save Args
    args_path = save_path / "args.json"
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists(): i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    _logger.note(f"Args: {args}")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    ## Start Testing
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
