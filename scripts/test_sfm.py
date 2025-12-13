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
from src.utils.auto_gpu import AutoGPU
from src.utils.fix_parser import add_negation_flags, add_minus_flags
from src.utils.tag2ansi import tag2ansi
from src.utils.get_force_map import get_force_map
from src.utils.extract_patches import extract_patches_torch
import torch.nn.functional as F
import re
import sys
import time
import json
import torch
import shlex
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as D
from copy import deepcopy
from tqdm import tqdm
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
from src.utils.plot import get_fig
from src.utils.auto_gpu import AutoGPU
from src.utils.fix_parser import add_negation_flags, add_minus_flags
from src.utils.tag2ansi import tag2ansi


_logger = logging.getLogger("src.test")


def test_once_sfm(args, test_loaders, model, criterion, diffusion, epoch):

    args.t_des_force=0.5
    args.r=10
    args.a_map_force=3.0
    args.d_map_force=0.6
    args.a_ped_force=2.0
    args.d_ped_force=0.3
    args.a_veh_force=5.0
    args.d_veh_force=0.5
    args.vel_damping=0.5

    test_timer = NamedTimer(unit='it', mode='pace')
    records_list = []
    for loader in test_loaders:
        map_data = loader.dataset.map_data
        map = torch.from_numpy(map_data.map).to(args.device).float()
        test_timer.add('prepare data')
        model.set_map_embedding(
            map=map,
            xmin=map_data.xmin,
            xmax=map_data.xmax,
            ymin=map_data.ymin,
            ymax=map_data.ymax,
        )
        test_timer.add('embed map')
        records = dict(loss=[], ade=[], fde=[], trajlen=[], ped_num=[], veh_num=[], rollout_time=[])
        for batch_idx, batch in enumerate(tqdm(loader, disable=False, leave=False, dynamic_ncols=True)):
            pos = batch['pos'].to(args.device)  # (batch_size, #pedestrian, 2)
            vel = batch['vel'].to(args.device)  # (batch_size, #pedestrian, 2)
            hst = batch['hst'].to(args.device)  # (batch_size, #pedestrian, hist_step, 2)
            des = batch['des'].to(args.device)  # (batch_size, #pedestrian, 2)
            spd = batch['spd'].to(args.device)  # (batch_size, #pedestrian, 1)
            veh = batch['veh'].to(args.device)  # (batch_size, #vehicle, hist_step + 1, 2)
            future_acc = batch['future_acc'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            future_pos = batch['future_pos'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            future_veh = batch['future_veh'].to(args.device)  # (batch_size, #vehicle, pred_step*roll_step, 2)
            ped_length = batch['ped_length'].to(args.device)  # (batch_size,)
            veh_length = batch['veh_length'].to(args.device)  # (batch_size,)

            S = args.sample_num  # 采样次数
            N = args.denoise_step  # 采样步数
            assert args.T % N == 0, f"试图使用 {N} 步采样，然而训练步数 {args.T} mod {N} 不等于 0!"
            assert 1 <= args.step_offset <= args.T // N, f"step_offset 应该取值于 {{1, ..., {args.T // N}}}!"
            pos_now = pos.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
            vel_now = vel.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
            hst_now = hst.repeat(S, 1, 1, 1)  # (S*B, #pedestrian, hist_step, 2)
            des_now = des.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
            spd_now = spd.repeat(S, 1, 1)  # (S*B, #pedestrian, 1)
            veh_now = veh.repeat(S, 1, 1, 1)  # (S*B, #vehicle, hist_step + 1, 2)
            ped_length_repeat = ped_length.repeat(S)  # (S*B,)
            veh_length_repeat = veh_length.repeat(S)  # (S*B,)
            test_timer.add('prepare data', n=0)

            for_plot = []
            acc_pred = []
            start_time = time.time()
            for step in range(args.roll_step):
                model.set_veh_embedding(veh=veh_now)
                model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
                model.set_sur_info()
                test_timer.add('embed data')

                # shape = list(future_acc.shape)
                # shape[0] *= S
                # shape[2] = args.pred_step
                # xt = torch.randn(shape, device=args.device)  # 从噪声开始
                # for_plot.append([diffusion.noise_to_x0(xt=xt, denoise_t=args.T, noise=0) / args.scale_accelerate])
                # stride = args.T // N
                # steps = reversed(range(args.step_offset, args.T+1, stride))
                # for t in tqdm(steps, disable=True, leave=False, dynamic_ncols=True):
                #     noisy_acc = xt
                #     denoise_t = torch.full((xt.shape[0],), t, device=args.device, dtype=torch.long)
                #     output = model(
                #         noisy_acc=noisy_acc, 
                #         denoise_t=denoise_t,
                #         ped_length=ped_length_repeat, 
                #         veh_length=veh_length_repeat,
                #     )  # (S*B, #pedestrian, pred_step, 2)
                #     if args.predict_noise:
                #         for_plot[-1].append(diffusion.noise_to_x0(xt=xt, denoise_t=t, noise=output) / args.scale_accelerate)
                #         xt = diffusion.denoise(xt, t, noise=output, stride=min(stride, t))
                #     else:
                #         for_plot[-1].append(output / args.scale_accelerate)
                #         xt = diffusion.denoise(xt, t, x0=output, stride=min(stride, t))
                # acc_new = xt / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)

                vel_now2 = vel_now.unsqueeze(-2)
                pos_now2 = pos_now.unsqueeze(-2)
                # 目的地导向力
                desire_vel = F.normalize(des_now.unsqueeze(-2) - pos_now2, dim=-1) * spd_now.unsqueeze(-2)  # (S*B, #pedestrian, pred_step, 2)
                des_force = (desire_vel - vel_now2).nan_to_num(0.0) / args.t_des_force  # (S*B, #pedestrian, pred_step, 2)
                # 场景障碍物排斥力
                F_map = get_force_map(r=args.r, A=args.a_map_force, B=args.d_map_force, device=args.device)  # (2r+1, 2r+1, 2)
                idx = pos_now2[..., 0].sub(model.xmin).div(model.xmax - model.xmin).mul(model.map.shape[0]).round().long().clamp(0, model.map.shape[0] - 1)  # (S*B, #pedestrian, pred_step)
                jdx = pos_now2[..., 1].sub(model.ymin).div(model.ymax - model.ymin).mul(model.map.shape[1]).round().long().clamp(0, model.map.shape[1] - 1)  # (S*B, #pedestrian, pred_step)
                patches = extract_patches_torch(model.map, idx.reshape(-1), jdx.reshape(-1), r=10).reshape(*idx.shape, 2*args.r+1, 2*args.r+1) # (S*B, #pedestrian, pred_step, 2r+1, 2r+1)
                map_force = (patches[..., None] * F_map).nan_to_num(0.0).flatten(-3, -2).sum(-2) # (S*B, #pedestrian, pred_step, 2)
                # 其他行人排斥力
                p = pos_now2[:, None, :, :, :] - pos_now2[:, :, None, :, :] # (S*B, #focal-pedestrian, #other-pedestrian, pred_step, 2)
                d = torch.norm(p, dim=-1, keepdim=True)
                n = -p / d.clamp(min=1e-6)
                F_ped = args.a_ped_force * torch.exp(-d / args.d_ped_force) * n
                ped_force = F_ped.nan_to_num(0.0).sum(dim=1) # (S*B, #pedestrian, pred_step, 2)
                # 其它车辆排斥力 (车辆用最后一帧位置)
                p = veh_now[:, :, None, -1:, :] - pos_now2[:, None, :, :, :] # (S*B, #vehicle, #pedestrian, pred_step, 2)
                d = torch.norm(p, dim=-1, keepdim=True)
                n = -p / d.clamp(min=1e-6)
                F_veh = args.a_veh_force * torch.exp(-d / args.d_veh_force) * n
                veh_force = F_veh.nan_to_num(0.0).sum(dim=1) # (S*B, #pedestrian, pred_step, 2)
                # 阻尼力
                vel_force = -args.vel_damping * vel_now2  # (S*B, #pedestrian, pred_step, 2)
                # 合力
                acc_new = des_force + map_force + ped_force + veh_force + vel_force


                acc_pred.append(acc_new)
                test_timer.add('denoise')

                vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
                pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
                veh_new = future_veh[:, :, step*args.pred_step:(step+1)*args.pred_step, :].repeat(S, 1, 1, 1)  # (S*B, #vehicle, pred_step, 2)

                hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (S*B, #pedestrian, hist_step, 2)
                veh_now = torch.cat([veh_now, veh_new], dim=-2)[:, :, -args.hist_step-1:, :]  # (S*B, #vehicle, hist_step + 1, 2)
                pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
                vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)
                test_timer.add('rollout')

            rollout_time = (time.time() - start_time) / args.roll_step

            acc_pred = torch.concat(acc_pred, dim=-2)  # (S*B, #pedestrian, roll_step*pred_step, 2)
            batch_size, ped_num, _, _ = future_acc.shape
            acc_pred = acc_pred.view(S, batch_size, ped_num, args.roll_step*args.pred_step, 2)  # (S, B, #pedestrian, roll_step*pred_step, 2)
            # 获取有效的行人掩模
            mask = torch.arange(ped_num, device=args.device).expand(batch_size, ped_num) < ped_length.unsqueeze(-1)  # (B, #pedestrian)
            # 计算 pos_true 和 vel_true
            acc_true = future_acc # (B, #pedestrian, roll_step*pred_step, 2)
            vel_true = vel.unsqueeze(-2) + acc_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, roll_step*pred_step, 2)
            pos_true = pos.unsqueeze(-2) + vel_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, roll_step*pred_step, 2)
            max_err = np.nanmax((pos_true - future_pos).abs().cpu().numpy(), axis=(-2, -1))
            _logger.debug(
                f"pos_true 和 future 最大差距 > 1: {(max_err > 1).mean():.2%}, "
                f"pos_true 和 future 最大差距 > 1e-6: {(max_err > 1e-6).mean():.2%}"
            )
            # pos_true = future_pos # (B, #pedestrian, roll_step*pred_step, 2)
            # 计算 loss
            loss = criterion(acc_pred, acc_true.expand(acc_pred.shape)) # float
            records['loss'].extend([loss.item()] * future_acc.shape[0]) # List[float]
            # 计算 distance error
            vel_pred = vel.unsqueeze(-2) + acc_pred.cumsum(dim=-2) / args.fps  # (S, B, #pedestrian, pred_step, 2)
            pos_pred = pos.unsqueeze(-2) + vel_pred.cumsum(dim=-2) / args.fps  # (S, B, #pedestrian, pred_step, 2)
            dis_err = (pos_pred - pos_true).norm(dim=-1) # (S, B, #pedestrian, pred_step)
            test_timer.add('evaluate')
            # 可视化
            # if batch_idx == 0:
            #     pid = 0
            #     save_path = f"{args.save_path}/visualize/epoch{epoch}_{loader.dataset.name}_idx{batch_idx}_pid{pid}.png"
            #     # visualize(args, pos, vel, hst, for_plot, mask, pos_true, pos_pred, save_path, pid)
            #     test_timer.add('visualize')
            # 移除 padding 的行人
            dis_err = dis_err[:, mask, :] # (S, valid{B*#pedestrian}, pred_step)
            # 选择 ade 最佳的 sample
            sample_idx = dis_err.mean(dim=-1).argmin(dim=0)  # (valid{B*#pedestrian},)
            valid_idx = torch.arange(dis_err.shape[1], device=args.device)  # (valid{B*#pedestrian},)
            dis_err = dis_err[sample_idx, valid_idx, :]  # (valid{B*#pedestrian}, pred_step)
            # 计算 ade, fde
            ade = dis_err.mean(dim=-1) # (valid{B*#pedestrian})
            fde = dis_err[..., -1] # (valid{B*#pedestrian})
            records['ade'].extend(ade.cpu().tolist()) # List[float]
            records['fde'].extend(fde.cpu().tolist()) # List[float]
            # 计算轨迹长度
            trajlen = pos_true.diff(dim=-2).norm(dim=-1).sum(dim=-1)[mask] # (valid{B*#pedestrian})
            records['trajlen'].extend(trajlen.cpu().tolist()) # List[float]
            # 统计行人和车辆数量
            records['ped_num'].extend(ped_length.cpu().tolist()) # List[int]
            records['veh_num'].extend(veh_length.cpu().tolist()) # List[int]
            # 统计 Rollout 用时
            records['rollout_time'].append(rollout_time)  # List[float]
            test_timer.add('evaluate', n=0)
        records_list.append(records)
        _logger.info(tag2ansi(
            f"[#66CCFF][Epoch {epoch}/{args.epochs}] Eval on {loader.dataset.name}: "
            f"[bold underline orange]Accuracy={1 - np.mean(records['ade']) / np.mean(records['trajlen']):.2%}[reset], "
            f"[#66CCFF]Loss={np.mean(records['loss']):.4f}, "
            f"[#66CCFF]ADE={np.mean(records['ade']):.4f}, "
            f"[#66CCFF]FDE={np.mean(records['fde']):.4f}, "
            f"[#66CCFF]AvgLen={np.mean(records['trajlen']):.4f}, "
            f"[#66CCFF]PedNum={np.mean(records['ped_num']):.1f}, "
            f"[#66CCFF]VehNum={np.mean(records['veh_num']):.1f}, "
            f"[#66CCFF]RolloutTime={np.mean(records['rollout_time'])*1000:.2f}ms "
            f"([bold underline orange]FPS={1/np.mean(records['rollout_time']):.2f} Hz[reset])"
        ))
    all_records = {
        'epoch': epoch,
        'dataset_class': [type(loader.dataset).__name__.removesuffix('Dataset') for loader in test_loaders],
        'dataset_names': [loader.dataset.name for loader in test_loaders],
        'sample_nums': [len(loader.dataset) for loader in test_loaders],
    }
    for records in records_list:
        for k, v in records.items():
            if k not in all_records:
                all_records[k] = []
            all_records[k].append(np.mean(v))
    w = np.array(all_records['sample_nums'], dtype=float)
    w /= w.sum()
    all_records['accuracy'] = 1 - np.sum(w * all_records['ade']) / np.sum(w * all_records['trajlen'])
    all_records['unweighted_accuracy'] = 1 - np.mean(all_records['ade']) / np.mean(all_records['trajlen'])
    _logger.note(tag2ansi(
        f"[#66CCFF][Epoch {epoch}/{args.epochs}] Overall: "
        f"[bold underline orange]Accuracy={all_records['accuracy']:.2%}[reset] (unweighted={all_records['unweighted_accuracy']:.2%}), "
        f"[#66CCFF]Loss={np.sum(w * all_records['loss']):.4f}, "
        f"[#66CCFF]ADE={np.sum(w * all_records['ade']):.4f}, "
        f"[#66CCFF]FDE={np.sum(w * all_records['fde']):.4f}, "
        f"[#66CCFF]AvgLen={np.sum(w * all_records['trajlen']):.4f}, "
        f"[#66CCFF]PedNum={np.sum(w * all_records['ped_num']):.4f}, "
        f"[#66CCFF]VehNum={np.sum(w * all_records['veh_num']):.4f}, "
        f"[#66CCFF]RolloutTime={np.mean(all_records['rollout_time'])*1000:.2}ms "
        f"([bold underline orange]FPS={1/np.mean(all_records['rollout_time']):.2f} Hz[reset]), "
        f"[#66CCFF]Time={test_timer}"
    ))
    if len(set(all_records['dataset_class'])) > 1:
        for klass in sorted(list(set(all_records['dataset_class']))):
            idxs = [i for i, k in enumerate(all_records['dataset_class']) if k == klass]
            ade = np.array([all_records['ade'][i] for i in idxs])
            fde = np.array([all_records['fde'][i] for i in idxs])
            trajlen = np.array([all_records['trajlen'][i] for i in idxs])
            ped_num = np.array([all_records['ped_num'][i] for i in idxs])
            veh_num = np.array([all_records['veh_num'][i] for i in idxs])
            rollout_time = np.array([all_records['rollout_time'][i] for i in idxs])
            w = np.array([all_records['sample_nums'][i] for i in idxs], dtype=float)
            w /= w.sum()
            acc = 1 - np.sum(w * ade) / np.sum(w * trajlen)
            _logger.note(tag2ansi(
                f"[#66CCFF][Epoch {epoch}/{args.epochs}] Overall on {klass} datasets: "
                f"[bold underline orange]Accuracy={acc:.2%}[reset], "
                f"[#66CCFF]ADE={np.sum(w * ade):.4f}, "
                f"[#66CCFF]FDE={np.sum(w * fde):.4f}, "
                f"[#66CCFF]AvgLen={np.sum(w * trajlen):.4f}, "
                f"[#66CCFF]PedNum={np.sum(w * ped_num):.4f}, "
                f"[#66CCFF]VehNum={np.sum(w * veh_num):.4f}, "
                f"[#66CCFF]RolloutTime={np.mean(rollout_time)*1000:.2f}ms "
                f"([bold underline orange]FPS={1/np.mean(rollout_time):.2f} Hz[reset])"
            ))
    return all_records


def main(args):
    ## Load Dataset
    # 加载数据集
    dataset_list = []
    if 'All' in args.datasets:
        args.datasets.remove('All')
        args.datasets += ['ETH', 'UCY', 'GC', 'SDD', 'WayMo', 'ORCA']
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
        dataset_list += WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/", total=100)
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
        test_dataset = []
        for d in dataset_list:
            if any(test_name in d.name for test_name in args.test_name):
                test_dataset.append(d)
    elif args.test_ratio is not None and args.split_by_scenario:
        # 将特定比例的场景划分到测试集
        random.shuffle(dataset_list)
        test_size = max(1, int(len(dataset_list) * args.test_ratio))
        test_dataset = dataset_list[-test_size:]
    elif args.test_ratio is not None and not args.split_by_scenario:
        # 将每个场景的特定比例样本划分到测试集
        test_dataset = []
        for d in dataset_list:
            d2 = deepcopy(d)
            train_num = int(len(d) * (1-args.test_ratio))
            d2.name = d2.name + f"_{args.test_ratio*100:.0f}test"
            d2.samples = d2.samples[train_num:]
            test_dataset.append(d2)
    else:
        # 训练集和测试集相同
        test_dataset = dataset_list
    # 创建数据加载器
    test_loaders = []
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
    if args.reload_checkpoint is not None:
        # 如果指定了 checkpoint 路径，则从该路径加载
        checkpoint_path = Path(args.reload_checkpoint)
    elif (Path(args.save_path) / "checkpoint.pth").exists():
        # 如果当前保存路径下存在 checkpoint，则从该路径加载
        checkpoint_path = Path(args.save_path) / "checkpoint.pth"
    else:
        raise ValueError("No checkpoint found to load!")
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
                    'p_drop_destination', 'p_drop_map', 'p_drop_speed',
                    'seed', 'reload_checkpoint', 'test_before_train', 'test_per_epoch',
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

    ## Test
    torch.set_grad_enabled(False)
    model.eval()
    test_records = test_once_sfm(args, test_loaders, model, criterion, diffusion, start_epoch)

    # 保存日志
    with open(f"{args.save_path}/records.jsonl", "a") as f:
        if test_records is not None:
            f.write(json.dumps(test_records) + "\n")

    ## Log Result
    _logger.note(tag2ansi(
        f"[bold underline orange]Accuracy={test_records['accuracy']:.2%}[reset] "
        f"at [#66CCFF]epoch {test_records['epoch']}[reset]. "
        f"[#66CCFF]ADE={np.mean(test_records['ade']):.4f}, "
        f"[#66CCFF]FDE={np.mean(test_records['fde']):.4f}, "
        f"[#66CCFF]AvgLen={np.mean(test_records['trajlen']):.4f}, "
        f"[#66CCFF]Loss={np.mean(test_records['loss']):.4f}, "
        f"[#66CCFF]PedNum={np.mean(test_records['ped_num']):.1f}, "
        f"[#66CCFF]VehNum={np.mean(test_records['veh_num']):.1f}"
    ))
    if len(set(test_records['dataset_class'])) > 1:
        for klass in sorted(list(set(test_records['dataset_class']))):
            idxs = [i for i, k in enumerate(test_records['dataset_class']) if k == klass]
            ade = np.array([test_records['ade'][i] for i in idxs])
            fde = np.array([test_records['fde'][i] for i in idxs])
            trajlen = np.array([test_records['trajlen'][i] for i in idxs])
            ped_num = np.array([test_records['ped_num'][i] for i in idxs])
            veh_num = np.array([test_records['veh_num'][i] for i in idxs])
            rollout_time = np.array([test_records['rollout_time'][i] for i in idxs])
            w = np.array([test_records['sample_nums'][i] for i in idxs], dtype=float)
            w /= w.sum()
            acc = 1 - np.sum(w * ade) / np.sum(w * trajlen)
            _logger.info(tag2ansi(
                f"[#66CCFF]Overall on {klass} datasets: "
                f"[bold underline orange]Accuracy={acc:.2%}[reset], "
                f"[#66CCFF]ADE={np.sum(w * ade):.4f}, "
                f"[#66CCFF]FDE={np.sum(w * fde):.4f}, "
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
    parser.add_argument('--reload_checkpoint', type=str, required=True, help="用于测试的 checkpoint（.pth 文件）")
    parser.add_argument('--required_memory_MB', type=int, default=6000, help="自动选择 GPU 时要求的最小剩余显存 (MB)")

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
        args.device = AutoGPU().choice_gpu(memory_MB=args.required_memory_MB, interval=15)

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
