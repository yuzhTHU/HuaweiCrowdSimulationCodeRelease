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
from fvcore.nn import FlopCountAnalysis
from src.dataset.base_dataset import RasterizedMap
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


# 配置日志
_logger = logging.getLogger("src.test_flops")

import types


def main(args):
    print(f"\n{'='*20} 开始 FLOPs 分析 (Monkey Patch Ver.) {'='*20}")
    print(f"Device: {args.device}")

    # 1. 初始化模型
    if args.use_new_model:
        model = NewModel(args).to(args.device)
    elif args.use_relative_model:
        model = RelativeModel(args).to(args.device)
    else:
        model = Model(args).to(args.device)
    model.eval()
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

    # 定义一个 "All-In-One" 的 Forward 函数，方便 fvcore 统计 FLOPs
    def unified_forward_patch(self, pos, vel, hst, des, spd, veh, future_acc, future_veh, map, xmin, xmax, ymin, ymax, ped_length, veh_length, is_inference=False):
        if is_inference:
            S = args.sample_num
            pos_now = pos.repeat(S, 1, 1)
            vel_now = vel.repeat(S, 1, 1)
            hst_now = hst.repeat(S, 1, 1, 1)
            des_now = des.repeat(S, 1, 1)
            spd_now = spd.repeat(S, 1, 1)
            veh_now = veh.repeat(S, 1, 1, 1)
            ped_length_repeat = ped_length.repeat(S)
            veh_length_repeat = veh_length.repeat(S)
            
            shape = list(future_acc.shape)
            shape[0] *= S
            shape[2] = args.pred_step
            noisy_acc = torch.randn(shape, device=args.device)
            denoise_t = torch.full((noisy_acc.shape[0],), 10, device=args.device, dtype=torch.long)
            
            current_ped_len = ped_length_repeat
            current_veh_len = veh_length_repeat
        else:
            pos_now, vel_now, hst_now = pos, vel, hst
            des_now, spd_now, veh_now = des, spd, veh
            
            acc_true = future_acc[:, :, 0:args.pred_step, :] 
            noisy_acc, _, denoise_t = diffusion.add_noise(acc_true * args.scale_accelerate)
            
            current_ped_len = ped_length
            current_veh_len = veh_length

        self.set_map_embedding(map=map, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        self.set_veh_embedding(veh=veh_now)
        self.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
        self.set_sur_info()
        
        return self._original_forward(
            noisy_acc=noisy_acc, 
            denoise_t=denoise_t,
            ped_length=current_ped_len, 
            veh_length=current_veh_len,
        )

    # -------------------------------------------------------------------
    # !!! 核心魔法：Monkey Patch !!!
    # 1. 备份原始的 forward 方法，改名为 _original_forward
    # 2. 将 unified_forward_patch 绑定为新的 forward
    # -------------------------------------------------------------------
    model._original_forward = model.forward
    model.forward = types.MethodType(unified_forward_patch, model)
    print("已成功 Patch 模型 Forward 方法，准备 Trace...")

    # 2. 构造 Dummy Data
    B = 1; P = 64; V = 64; H = 100; W = 100
    map_data = torch.randn(W, H).to(args.device)
    pos = torch.randn(B, P, 2).to(args.device)
    vel = torch.randn(B, P, 2).to(args.device)
    hst = torch.randn(B, P, args.hist_step, 2).to(args.device)
    des = torch.randn(B, P, 2).to(args.device)
    spd = torch.randn(B, P, 1).to(args.device)
    veh = torch.randn(B, max(1, V), args.hist_step + 1, 2).to(args.device)
    ped_length = torch.tensor([P] * B).long().to(args.device)
    veh_length = torch.tensor([V] * B).long().to(args.device)
    future_acc = torch.randn(B, P, args.pred_step, 2).to(args.device)
    future_veh = torch.randn(B, max(1, V), args.pred_step, 2).to(args.device)

    # -------------------------------------------------------------------------
    # 3. 计算 Train FLOPs
    # -------------------------------------------------------------------------
    print("--> 正在计算 Train FLOPs...")
    
    # 注意参数列表：要和 unified_forward_patch 对应
    # (pos, vel, hst, des, spd, veh, future_acc, future_veh, map, xmin, xmax, ymin, ymax, ped_length, veh_length, is_inference)
    train_inputs = (
        pos, vel, hst, des, spd, veh, future_acc, future_veh, 
        map_data, 0, W, 0, H, # Map & Limits
        ped_length, veh_length, 
        False # is_inference=False
    )
    
    # 直接传入 model，不再需要 TrainWrapper！
    train_analysis = FlopCountAnalysis(model, train_inputs)
    train_analysis.unsupported_ops_warnings(False)
    train_flops = train_analysis.total()

    # -------------------------------------------------------------------------
    # 4. 计算 Test FLOPs
    # -------------------------------------------------------------------------
    print("--> 正在计算 Test FLOPs (Single Step)...")
    
    test_inputs = (
        pos, vel, hst, des, spd, veh, future_acc, future_veh, 
        map_data, 0, W, 0, H, 
        ped_length, veh_length, 
        True # is_inference=True
    )

    test_analysis = FlopCountAnalysis(model, test_inputs)
    test_analysis.unsupported_ops_warnings(False)
    test_flops = test_analysis.total()

    # -------------------------------------------------------------------------
    # 5. 结果打印
    # -------------------------------------------------------------------------
    _logger.note(
        f"Train Single Forward: {train_flops / 1e9:.4f} GFLOPs\n"
        f"Train Total (Fwd + Bwd): {3 * train_flops / 1e9:.4f} GFLOPs (per iteration)\n"
    )
    _logger.note(
        f"Test Single Forward: {test_flops / 1e9:.4f} GFLOPs\n"
        f"Test Total Inference (Denoise for {args.denoise_step} steps): {test_flops * args.denoise_step / 1e9:.4f} GFLOPs\n"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    # 基础配置
    parser.add_argument("--name", type=str, default="test_flops", help="实验任务名称，用于生成实验ID")
    parser.add_argument("--exp_name", type=str, default=None, help="手动指定实验名称（若指定则覆盖自动生成的名称）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备，可选 'cpu', 'cuda:0' 或 'auto'（自动选择显存充足的 GPU）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，固定以复现实验结果")
    parser.add_argument("--save_dir", type=str, default="./logs/test_flops", help="日志和模型权重的保存根目录")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式（输出更多日志，不保存部分文件）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的工作线程数（0 表示主线程）")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=128, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率 (Learning Rate)")
    parser.add_argument("--epochs", type=int, default=10000, help="最大训练轮数")
    parser.add_argument('--patience', type=int, default=20, help="Early Stopping 的耐心值（多少个 epoch 验证集指标不提升则停止）")
    parser.add_argument('--loss_type', type=str, default='noise', choices=['position', 'accelerate', 'noise'], help="损失函数计算的目标类型")
    parser.add_argument('--reload_checkpoint', type=str, default=None, help="断点续训的 checkpoint 路径（.pth 文件）")
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
    parser.add_argument('--test_before_train', action='store_true', help="是否在训练开始前先运行一次测试")
    parser.add_argument('--test_per_epoch', type=int, default=10, help="每隔多少个 epoch 运行一次测试")

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

    ## Start Training
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
