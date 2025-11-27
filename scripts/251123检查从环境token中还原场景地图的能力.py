import os
if os.getcwd().endswith("scripts"):
    os.chdir("..")
assert os.path.exists("src"), "Please run this script from the project root directory!"

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
from src.dataset import ETHDataset, UCYDataset, SDDDataset, GCDataset, WayMoDataset
from src.model.model import Model, RelativeModel, FourierPositionalEncoding
from src.diffusion import DDPM, DDIM
from src.utils.logger import init_logger
from src.utils.seed import seed_all
from src.utils.timer import NamedTimer
from src.utils.plot import get_fig
from src.utils.auto_gpu import AutoGPU
from src.utils.negation_flags import add_negation_flags
from src.utils.tag2ansi import tag2ansi

_logger = logging.getLogger("src.main")

import torch.nn as nn
class Model2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fourier = FourierPositionalEncoding(out_dim=args.model_dim, num_bands=args.model_dim)
        self.attention = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=args.model_dim,
                nhead=args.head_num,
                dim_feedforward=4*args.model_dim,
                dropout=args.dropout,
                activation='relu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=args.attention_layer_num,
        )
        self.out = nn.Sequential(
            nn.Linear(args.model_dim, args.model_dim),
            nn.ReLU(),
            nn.Linear(args.model_dim, args.model_dim),
            nn.ReLU(),
            nn.Linear(args.model_dim, args.model_dim),
            nn.ReLU(),
            nn.Linear(args.model_dim, 6),
        )

    def forward(self, ltn_embedding, w, h, xmin, xmax, ymin, ymax):
        x = torch.linspace(xmin, xmax, w, device=ltn_embedding.device)
        y = torch.linspace(ymin, ymax, h, device=ltn_embedding.device)
        z = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)  # (w, h, 2)
        z = self.fourier(z)  # (w, h, model_dim)
        z = self.attention(z.flatten(0, 1).unsqueeze(0), ltn_embedding.unsqueeze(0)).reshape(z.shape)  # (w, h, model_dim)
        z = self.out(z)  # (w, h, 3)
        return z

class Criterion(nn.Module):
    def quantize(self, x):
        is_zero = (x == 0)
        is_dot1 = (0.05 < x) & (x < 0.15)
        is_dot2 = (0.15 < x) & (x < 0.25)
        is_one = (x == 1)
        is_nan = torch.isnan(x)
        is_others = (~is_zero) & (~is_nan) & (~is_one)
        quantize_x = (
            0 * is_zero.long()
            + 1 * is_dot1.long()
            + 2 * is_dot2.long()
            + 3 * is_one.long()
            + 4 * is_others.long()
            + 5 * is_nan.long()
        )
        return quantize_x

    def forward(self, pred, true):
        """ true: (B, ...), pred: (B, ..., 4) """
        true_class = self.quantize(true)
        # --- 动态计算权重 ---
        # 统计当前 batch (map) 中各像素的数量
        # flat_target = true_class.view(-1)
        # counts = torch.bincount(flat_target, minlength=4).float()
        # 或者使用预设的经验权重 (推荐，更稳定)
        # 0(空地): 1.0
        # 1(障碍物): 20.0 (强迫恢复障碍)
        # 2(草坪): 20.0 (强迫恢复草坪)
        # 3(NaN): 0.0
        weights = torch.tensor([1.0, 5.0, 10.0, 20.0, 20.0, 0.0], device=pred.device)
        return nn.functional.cross_entropy(
            pred.unsqueeze(0).permute(0, 3, 1, 2), 
            true_class.unsqueeze(0),
            weight=weights,
        )

def main(args):
    ## Load Dataset
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
        dataset_list += WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/", total=100)
        args.datasets.remove("WayMo")
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
    if False:
        N = len(dataset_list)
        C = min(5, N)
        R = math.ceil(N / C)
        fi, fig, axes = get_fig(R, C, AW=6, AH=6, dpi=300, lw=1, fontsize=14)
        for i, dataset in enumerate(dataset_list):
            ax = axes[i]
            ax.imshow(dataset.map_data.map.T, origin='lower', extent=(
                dataset.map_data.xmin,
                dataset.map_data.xmax,
                dataset.map_data.ymin,
                dataset.map_data.ymax,
            ), cmap='gray_r')
            for pid, group in dataset.df_data.groupby('id'):
                color = 'blue' if group['type'].iloc[0] == 'pedestrian' else 'red'
                ax.plot(group['x'], group['y'], color=color, linewidth=1, alpha=0.5)
                ax.scatter(group['x'].iloc[0], group['y'].iloc[0], color=color, marker='o', s=10)
                ax.scatter(group['x'].iloc[-1], group['y'].iloc[-1], color=color, marker='x', s=10)
            ax.set_title(f"{dataset.name} ({len(dataset)} samples)", fontsize=14)
        fig.tight_layout()
        fig_path = Path(args.save_path) / "dataset_overview.png"
        fig.savefig(fig_path)
        _logger.note(f"Dataset overview figure saved to {fig_path}")
        plt.close(fig)
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
        raise ValueError("--split_by_scenario must be set when --test_ratio is used!")
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
    # 创建数据加载器
    train_loaders = []
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
    for dataset in test_dataset:
        if len(dataset) == 0:
            _logger.warning(f"Dataset {dataset.name} has no testing samples!")
            continue
        test_loaders.append(D.DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    _logger.note(
        "Datasets:\n"
        f"Train on {[d.name for d in train_dataset]} datasets ({sum([len(d) for d in train_dataset]):,} samples in total)\n"
        f"Test on {[d.name for d in test_dataset]} datasets ({sum([len(d) for d in test_dataset]):,} samples in total)"
    )

    ## Load Model
    if args.use_relative_model:
        model = RelativeModel(args).to(args.device)
    else:
        model = Model(args).to(args.device)
    model2 = Model2(args).to(args.device)
    params_to_train = [*model2.parameters()]
    if args.finetune_model:
        params_to_train += [*model.parameters()]
        _logger.info("Finetune the model.map_embedder as well.")
    else:
        _logger.info("Freeze the model.map_embedder parameters.")
    optimizer = torch.optim.Adam(params_to_train, lr=args.lr)
    criterion = Criterion()
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
                    'save_dir', 'save_path', 'reload_checkpoint', 'command',
                    'device', 'required_memory_MB', 'test_name', 'test_ratio', 'split_by_scenario',
                    'num_workers', 'test_per_epoch', 'test_before_train'
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
        if False and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            _logger.warning("Optimizer state not found in checkpoint, optimizer re-initialized.")
    else:
        start_epoch = 0

    ## Train
    timer = NamedTimer()
    for epoch in range(start_epoch, args.epochs+1):
        # 训练一个 epoch
        if epoch > 0:
            torch.set_grad_enabled(True)
            model.train()
            train_records = train_once(args, train_loaders, model, model2, optimizer, criterion, diffusion, epoch)
            timer.add('train')
        else:
            train_records = None

        # 测试一个 epoch
        if (epoch > 0 and not epoch % args.test_per_epoch) or (epoch == 0 and args.test_before_train):
            torch.set_grad_enabled(False)
            model.eval()
            if train_loaders != test_loaders:
                test_once(args, train_loaders, model, model2, criterion, diffusion, epoch)
            test_records = test_once(args, test_loaders, model, model2, criterion, diffusion, epoch)
            timer.add('test')
        else:
            test_records = None

        # 保存日志
        with open(f"{args.save_path}/records.jsonl", "a") as f:
            if train_records is not None:
                f.write(json.dumps(train_records) + "\n")
            if test_records is not None:
                f.write(json.dumps(test_records) + "\n")

        # 保存加载点
        if '_last_checkpoint_time' not in locals() or (datetime.now() - _last_checkpoint_time).seconds  > 300:
            # 只在间隔超过 5 min 时保存
            _last_checkpoint_time = datetime.now()
            save_path = f"{args.save_path}/checkpoint.pth"
            torch.save({
                "epoch": epoch,
                "args": vars(args),
                "model": model.state_dict(),
                "model2": model2.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_path)
            _logger.info(tag2ansi(f"Checkpoint saved to [underline green]{save_path}[reset]."))
            timer.add('save_checkpoint')
        
        # 定期保存
        if set(str(epoch)[1:]) == {'0'}:
            # 只在 epoch=10,20,...,100,...,1000,... 时保存
            save_path = Path(args.save_path) / "checkpoints" / f"epoch{epoch}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "args": vars(args),
                "model": model.state_dict(),
                "model2": model2.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_path)
            _logger.note(tag2ansi(f"Model saved to [underline green]{save_path}[reset]"))
            timer.add('save_periodly')

        # 保存最佳模型
        if test_records is not None:
            if (
                'best_records' not in locals() or 
                np.mean(test_records['loss']) < np.mean(best_records['loss'])
            ):
                patience = args.patience
                best_records = test_records
                save_path = f"{args.save_path}/best.pth"
                torch.save({
                    "epoch": epoch,
                    "args": vars(args),
                    "model": model.state_dict(),
                    "model2": model2.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)
                _logger.note(tag2ansi(f"Best model saved to [underline green]{save_path}[reset]"))
            else:
                patience -= 1
                _logger.info(tag2ansi(
                    f"Patience left: [brightred]{patience}/{args.patience}[reset] ("
                    f"[bold underline orange]best Loss={np.mean(best_records['loss']):.2%}[reset] "
                    f"at epoch [#66CCFF]{best_records['epoch']}[reset]. "
                ))
            timer.add('save_best')

        # 打印用时
        allocated = torch.cuda.memory_allocated(args.device) / 1024 / 1024 / 1024
        reserved = torch.cuda.memory_reserved(args.device) / 1024 / 1024 / 1024
        peak = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024 / 1024
        _logger.info(tag2ansi(
            f"[pink][Epoch {epoch}/{args.epochs}] finished. "
            f"Time Usage={timer}, "
            f"CUDA ({args.device}) usage: allocated={allocated:.1f}GiB, peak={peak:.1f}GiB, reserved={reserved:.1f}GiB"
            # f"adjust reserved memory from {reserved_raw/1024:.1f}GiB to {reserved_new/1024:.1f}GiB"
            "[reset]"
        ))

        # 释放额外的显存
        if train_records is not None:
            peak = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024
            reserved_raw = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            torch.cuda.empty_cache() # 释放 reserved 但是未被 allocated 的 block
            reserved_new = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            if reserved_new < peak: # 释放了过多的显存，之后可能会 OOM
                allocated = torch.cuda.memory_allocated(args.device) / 1024 / 1024
                if (keep_MB := int(np.ceil(peak - allocated))) > 0: # 把需要的显存再占回来
                    fuck_cuda = AutoGPU.fuck_gpu(device=args.device, memory_MB=keep_MB, block_MB=None)
                    del fuck_cuda
                reserved_new = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            _logger.info(tag2ansi(
                f"[brown]Adjust reserved memory from {reserved_raw/1024:.1f}GiB to {reserved_new/1024:.1f}GiB. [reset]"
            ))
        
        # 提前终止
        if 'patience' in locals() and patience <= 0:
            _logger.warning(tag2ansi(
                f"Early stopping at epoch [lightred]{epoch}/{args.epochs}[reset], "
                f"[bold underline orange]best Loss={np.mean(best_records['loss']):.2%}[reset] "
                f"at [#66CCFF]epoch {best_records['epoch']}[reset]. "
            ))
            break

    _logger.note(f"Training finished. Re-run: {args.command}")


def train_once(args, train_loaders, model, model2, optimizer, criterion, diffusion, epoch):
    train_timer = NamedTimer(unit='it', mode='pace')
    records_list = []
    optimizer.zero_grad()
    for loader in train_loaders:
        map_data = loader.dataset.map_data
        map_true = torch.from_numpy(map_data.map).to(args.device).float()
        records = dict(loss=[])
        train_timer.add('prepare data')
        model.set_map_embedding(
            map=map_true,
            xmin=map_data.xmin,
            xmax=map_data.xmax,
            ymin=map_data.ymin,
            ymax=map_data.ymax,
        )
        map_pred = model2(model.ltn_embedding, map_true.shape[0], map_true.shape[1], map_data.xmin, map_data.xmax, map_data.ymin, map_data.ymax)
        train_timer.add('forward')
        # mask = torch.isnan(map_true)
        # map_true[mask] = 0.0
        # loss = criterion(map_pred[~mask], map_true[~mask])
        loss = criterion(map_pred, map_true)
        loss.backward()
        train_timer.add('backward')
    optimizer.step()
    optimizer.zero_grad()
    records['loss'].append(loss.item())
    records_list.append(records)
    _logger.debug(
        f"[Epoch {epoch}/{args.epochs}] Train on {loader.dataset.name}: "
        f"Loss={np.mean(records['loss']):.4f}"
    )
    all_records = {
        'epoch': epoch,
        'dataset_names': [loader.dataset.name for loader in train_loaders],
        'sample_nums': [len(loader.dataset) for loader in train_loaders],
    }
    for records in records_list:
        for k, v in records.items():
            if k not in all_records:
                all_records[k] = []
            if isinstance(v[0], (int, float)):
                mean_v = np.mean(v)
            else: 
                mean_v = np.mean(v, axis=0).tolist()
            all_records[k].append(mean_v)
    _logger.info(tag2ansi(
        f"[#66CCFF][Epoch {epoch}/{args.epochs}] "
        f"[#66CCFF]Loss={np.mean(all_records['loss']):.4f} "
        f"[#66CCFF]Time={train_timer}"
    ))
    return all_records


def test_once(args, test_loaders, model, model2, criterion, diffusion, epoch):
    test_timer = NamedTimer(unit='it', mode='pace')
    records_list = []
    for loader in test_loaders:
        records = {'loss': []}
        map_data = loader.dataset.map_data
        map_true = torch.from_numpy(map_data.map).to(args.device).float()
        test_timer.add('prepare data')
        model.set_map_embedding(
            map=map_true,
            xmin=map_data.xmin,
            xmax=map_data.xmax,
            ymin=map_data.ymin,
            ymax=map_data.ymax,
        )
        map_pred = model2(model.ltn_embedding, map_true.shape[0], map_true.shape[1], map_data.xmin, map_data.xmax, map_data.ymin, map_data.ymax)
        test_timer.add('forward')
        # mask = torch.isnan(map_true)
        # map_true[mask] = 0.0
        # loss = criterion(map_pred[~mask], map_true[~mask])
        loss = criterion(map_pred, map_true)
        records['loss'].append(loss.item())
        records_list.append(records)
        _logger.info(tag2ansi(
            f"[#66CCFF][Epoch {epoch}/{args.epochs}] Eval on {loader.dataset.name}: "
            f"[#66CCFF]Loss={np.mean(records['loss']):.4f}, "
        ))
        # 可视化
        if True:
            save_path = Path(f"{args.save_path}/visualize/epoch{epoch}/{loader.dataset.name}.png")
            fi, fig, axes = get_fig(3, 3, AW=12, AH=12, dpi=100, fontsize=14, lw=1)
            ax_loader = iter(axes)

            ax = next(ax_loader)
            cmap = plt.get_cmap('gray_r').copy()
            cmap.set_bad(color='#596275')
            ax.imshow(map_true.cpu().numpy().T, cmap=cmap, origin='lower', vmin=0, vmax=1, extent=[map_data.xmin, map_data.xmax, map_data.ymin, map_data.ymax])
            ax.set_title("Ground Truth Map")

            ax = next(ax_loader)
            ax.imshow(criterion.quantize(map_true).cpu().numpy().T, origin='lower', vmin=0, vmax=5, extent=[map_data.xmin, map_data.xmax, map_data.ymin, map_data.ymax])
            ax.set_title("Quantized Ground Truth Map")

            ax = next(ax_loader)
            ax.imshow(map_pred.argmax(dim=-1).detach().cpu().numpy().T, origin='lower', vmin=0, vmax=5, extent=[map_data.xmin, map_data.xmax, map_data.ymin, map_data.ymax])
            ax.set_title("Predicted Map w/o Normalization")

            for i in range(6):
                ax = next(ax_loader)
                ax.imshow(map_pred.softmax(dim=-1)[..., i].detach().cpu().numpy().T, origin='lower', vmin=0, vmax=1, extent=[map_data.xmin, map_data.xmax, map_data.ymin, map_data.ymax])
                ax.set_title(f"Predicted Map Prob. of Class {i}")

            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)
            _logger.info(tag2ansi(f"Visualization saved to [green]{save_path}[reset]"))
            test_timer.add('visualize')
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
    _logger.note(tag2ansi(
        f"[#66CCFF][Epoch {epoch}/{args.epochs}] Overall: "
        f"[#66CCFF]Loss={np.sum(all_records['loss']):.4f}, "
        f"[#66CCFF]Time={test_timer}"
    ))
    return all_records


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="检查从环境token中还原场景地图的能力")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--sampling_method', type=str, default="DDIM", choices=['DDPM', 'DDIM'])
    parser.add_argument("--T", type=int, default=100, help="训练时的扩散步数")
    parser.add_argument('--sample_num', type=int, default=20, help="测试时每个轨迹采样 {sample_num} 次")
    parser.add_argument('--denoise_step', type=int, default=2, help="采样时进行 {denoise_step} 次去噪")
    parser.add_argument('--step_offset', type=int, default=10, help="最后一步去噪从 x_{step_offset} 到 x_0")
    parser.add_argument('--antithetic_sampling', action='store_true', default=True)
    parser.add_argument('--loss_type', type=str, default='noise', choices=['position', 'accelerate', 'noise'])
    parser.add_argument('--rollout_lambda', type=float, default=1.0, help="rollout loss 衰减系数，设置 <1 以赋予未来更高权重")
    parser.add_argument('--multi_frame_rollout', type=int, default=1, help="每次训练时 rollout 的帧数")
    parser.add_argument('--scale_accelerate', type=float, default=1.0, help="加速度的缩放比例")
    parser.add_argument('--p_drop_map', type=float, default=None)
    parser.add_argument('--p_drop_destination', type=float, default=None)
    parser.add_argument('--p_drop_speed', type=float, default=None)
    parser.add_argument("--hist_step", type=int, default=8)
    parser.add_argument("--pred_step", type=int, default=1)
    parser.add_argument("--skip_step", type=int, default=1)
    parser.add_argument("--roll_step", type=int, default=12)
    parser.add_argument("--fps", type=int, default=2.5)
    parser.add_argument("--dot_per_meter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./logs/检查从环境token中还原场景地图的能力")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--map_feature_dim', type=int, default=64)
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--attention_layer_num', type=int, default=1)
    parser.add_argument('--lstm_layer_num', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--latent_token_num', type=int, default=16)
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--datasets', type=str, default=["debug"], choices=['ETH', 'UCY', 'GC', 'SDD', 'WayMo', 'All', 'debug'], nargs='*')
    parser.add_argument('--test_name', type=str, default=None, nargs='+')
    parser.add_argument('--test_ratio', type=float, default=None)
    parser.add_argument('--split_by_scenario', action='store_true')
    parser.add_argument('--cache_dataset', action='store_true', default=True)
    parser.add_argument('--test_before_train', action='store_true')
    parser.add_argument('--test_per_epoch', type=int, default=10)
    parser.add_argument('--reload_checkpoint', type=str, default=None, help='/path/to/checkpoint.pth')
    parser.add_argument('--predict_noise', action='store_true', default=True)
    parser.add_argument('--required_memory_MB', type=int, default=6000)
    parser.add_argument('--finetune_model', action='store_true', default=False)
    parser.add_argument('--use_relative_model', action='store_true', default=True)
    parser.add_argument('--use_spatial_anchor', action='store_true', default=False)
    parser = add_negation_flags(parser)
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

    ## Start Training
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
