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


class TrainWrapper(torch.nn.Module):
    def __init__(self, args, model, diffusion, criterion):
        super().__init__()
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.criterion = criterion

    def forward(self, pos, vel, hst, des, spd, veh, future_acc, future_veh, map, xmin, xmax, ymin, ymax, ped_length, veh_length):
        args = self.args
        model = self.model
        diffusion = self.diffusion
        criterion = self.criterion

        # train_once
        pos_now = pos
        vel_now = vel
        hst_now = hst
        des_now = des
        spd_now = spd
        veh_now = veh
        rollout_loss = []
        for step in range(args.multi_frame_rollout):
            # DDPM forward
            # pos_true = future_pos[:, :, args.pred_step*step:args.pred_step*(step+1), :] # (B, #pedestrian, pred_step, 2)
            # vel_true = pos_true.diff(dim=-2, prepend=pos_now.unsqueeze(-2)) * args.fps  # (B, #pedestrian, roll_step*pred_step, 2)
            # acc_true = vel_true.diff(dim=-2, prepend=vel_now.unsqueeze(-2)) * args.fps  # (B, #pedestrian, roll_step*pred_step, 2)
            acc_true = future_acc[:, :, args.pred_step*step:args.pred_step*(step+1), :] # (B, #pedestrian, pred_step, 2)
            noisy_acc, noise_true, denoise_t = diffusion.add_noise(acc_true * args.scale_accelerate)

            if args.p_drop_map and random.random() < args.p_drop_map:
                map = torch.full_like(map, torch.nan, device=map.device)
            if args.p_drop_destination and random.random() < args.p_drop_destination:
                des_now = torch.full_like(des_now, torch.nan, device=des_now.device)
            if args.p_drop_speed and random.random() < args.p_drop_speed:
                spd_now = torch.full_like(spd_now, torch.nan, device=spd_now.device)

            # DDPM backward
            model.set_map_embedding(
                map=map,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )
            model.set_veh_embedding(veh=veh_now)
            model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
            model.set_sur_info()
            output = model(
                noisy_acc=noisy_acc, 
                denoise_t=denoise_t,
                ped_length=ped_length, 
                veh_length=veh_length,
            )  # (B, #pedestrian, pred_step, 2)

            # Compute Loss
            if args.predict_noise:
                noise_pred = output
                acc_pred = diffusion.noise_to_x0(xt=noisy_acc, denoise_t=denoise_t, noise=noise_pred) / args.scale_accelerate
            else:
                acc_pred = output / args.scale_accelerate
            
            if args.loss_type == 'accelerate':
                loss = criterion(acc_pred, acc_true)
            elif args.loss_type == 'position':
                # acc_true 是从 pos_true 算出来的，因此不用再返回去计算 pos_true 了
                vel_true = vel_now.unsqueeze(-2) + acc_true.cumsum(dim=-2) / args.fps
                pos_true = pos_now.unsqueeze(-2) + vel_true.cumsum(dim=-2) / args.fps
                vel_pred = vel_now.unsqueeze(-2) + acc_pred.cumsum(dim=-2) / args.fps
                pos_pred = pos_now.unsqueeze(-2) + vel_pred.cumsum(dim=-2) / args.fps
                loss = criterion(pos_pred, pos_true)
            elif args.loss_type == 'noise':
                if not args.predict_noise:
                    raise ValueError("When using noise prediction loss, the model must predict noise!")
                loss = criterion(noise_pred, noise_true)
            else:
                raise ValueError(f"Unknown loss type {args.loss_type}!")
            rollout_loss.append(loss.detach().cpu().tolist())

            acc_new = acc_pred.detach()  # (B, #pedestrian, pred_step, 2)
            vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (B, #pedestrian, pred_step, 2)
            pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (B, #pedestrian, pred_step, 2)
            veh_new = future_veh[:, :, step*args.pred_step:(step+1)*args.pred_step, :]  # (B, #vehicle, pred_step, 2)

            hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (B, #pedestrian, hist_step, 2)
            veh_now = torch.cat([veh_now, veh_new], dim=-2)[:, :, -args.hist_step-1:, :]  # (B, #vehicle, hist_step + 1, 2)
            pos_now = pos_new[:, :, -1, :] # (B, #pedestrian, 2)
            vel_now = vel_new[:, :, -1, :] # (B, #pedestrian, 2)


class TestWrapper(torch.nn.Module):
    def __init__(self, args, model, diffusion, criterion):
        super().__init__()
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.criterion = criterion

    def forward(self, pos, vel, hst, des, spd, veh, future_acc, future_veh, map, xmin, xmax, ymin, ymax, ped_length, veh_length):
        args = self.args
        model = self.model
        diffusion = self.diffusion

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

        model.set_veh_embedding(veh=veh_now)
        model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
        model.set_sur_info()

        shape = list(future_acc.shape)
        shape[0] *= S
        shape[2] = args.pred_step
        xt = torch.randn(shape, device=args.device)  # 从噪声开始
        stride = args.T // N
        steps = reversed(range(args.step_offset, args.T+1, stride))
        for t in tqdm(steps, disable=True, leave=False, dynamic_ncols=True):
            noisy_acc = xt
            denoise_t = torch.full((xt.shape[0],), t, device=args.device, dtype=torch.long)
            output = model(
                noisy_acc=noisy_acc, 
                denoise_t=denoise_t,
                ped_length=ped_length_repeat, 
                veh_length=veh_length_repeat,
            )  # (S*B, #pedestrian, pred_step, 2)
            if args.predict_noise:
                xt = diffusion.denoise(xt, t, noise=output, stride=min(stride, t))
            else:
                xt = diffusion.denoise(xt, t, x0=output, stride=min(stride, t))
        acc_new = xt / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)

        vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
        pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
        veh_new = future_veh[:, :, 0*args.pred_step:(0+1)*args.pred_step, :].repeat(S, 1, 1, 1)  # (S*B, #vehicle, pred_step, 2)

        hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (S*B, #pedestrian, hist_step, 2)
        veh_now = torch.cat([veh_now, veh_new], dim=-2)[:, :, -args.hist_step-1:, :]  # (S*B, #vehicle, hist_step + 1, 2)
        pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
        vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)


def main(args):
    _logger.info(f"Running FLOPs analysis on device: {args.device}")

    # 1. 初始化模型
    # 这里默认使用 RelativeModel，你可以修改为 NewModel 或 Model
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

    # 2. 构造 Dummy Data
    B = 1  # Batch Size
    P = 64 # 行人数量
    V = 64 # 车辆数量
    H = 100 # Map Height
    W = 100 # Map Width
    
    map = torch.randn(W, H).to(args.device)
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

    # 3. 计算 Train 的前向推理 FLOPs
    train_wrapper = TrainWrapper(args, model, diffusion, criterion)
    train_inputs = (pos, vel, hst, des, spd, veh, future_acc, future_veh, map, 0, W, 0, H, ped_length, veh_length)
    train_analysis = FlopCountAnalysis(train_wrapper, train_inputs)
    train_analysis.unsupported_ops_warnings(False) # 忽略不支持算子的警告
    train_flops = train_analysis.total()
    _logger.note(
        f"FLOPs Analysis Results:\n"
        f"  Train Forward FLOPs: {train_flops:,} FLOPs\n"
    )

    # 4. 计算 Test 的前向推理 FLOPs
    test_wrapper = TestWrapper(args, model, diffusion, criterion)
    test_inputs = (pos, vel, hst, des, spd, veh, future_acc, future_veh, map, 0, W, 0, H, ped_length, veh_length)
    test_analysis = FlopCountAnalysis(test_wrapper, test_inputs)
    test_analysis.unsupported_ops_warnings(False) # 忽略不支持算子的警告
    test_flops = test_analysis.total()
    _logger.note(
        f"FLOPs Analysis Results:\n"
        f"  Test Forward FLOPs: {test_flops:,} FLOPs\n"
    )

    # # 单位转换
    # G = 1e9
    
    # _logger.info("-" * 60)
    # _logger.info(f"Model: {type(model).__name__}")
    # _logger.info(f"Input Shape: Batch={B}, Pedestrians={P}, Vehicles={V}")
    # _logger.info("-" * 60)
    
    # _logger.info(f"1. [Embedding] (Map+Veh+Ped+Graph): \t{f_embed / G:.6f} GFLOPs")
    # _logger.info(f"2. [Denoise] (Single Forward):      \t{f_denoise / G:.6f} GFLOPs")
    # _logger.info("-" * 60)

    # # --- 场景 A: 训练 (Train) ---
    # # 训练通常包含：
    # # 1. 计算一次 Embedding
    # # 2. 计算一次 Forward (用于算 Loss)
    # # 3. 计算一次 Backward (通常估算为 2 * Forward)
    # # 注意：Map Embedding 在训练中通常每个 Batch 都要算一次 (因为有 dropout 或不同场景)
    
    # train_forward = f_embed + f_denoise
    # train_backward = 2 * train_forward # 经验估算
    # total_train = train_forward + train_backward
    
    # _logger.info("[Scenario 1: Training one sample]")
    # _logger.info(f"  (Assume 1 Rollout Step, Backward ≈ 2*Forward)")
    # _logger.info(f"  Total Training FLOPS: \033[92m{total_train / G:.6f} GFLOPs\033[0m")
    # _logger.info(f"  Breakdown: Forward ({train_forward/G:.4f} G) + Backward ({train_backward/G:.4f} G)")

    # # --- 场景 B: 测试 (Test / Inference) ---
    # # 测试通常包含：
    # # 1. Embeddings (在每个 Rollout Step 开始时算一次)
    # # 2. Denoise Loop (循环 denoise_step 次)
    # # 注意：在 test_once 中，map embedding 是在循环外算的，可以忽略不计；
    # # 但是 Ped/Veh embedding 是在每个 Rollout Step 都要算的。
    
    # # 这里我们计算“预测一步 (One Rollout Step)”所需的算力
    # # 包含：一次 Ped/Veh Embed + (Denoise Steps * U-Net)
    
    # # 近似认为 Test 的 Embedding 开销 = f_embed (稍微多算了一点 Map Encoder，但通常 Map Encoder 开销很小)
    # test_one_rollout_step = f_embed + (f_denoise * args.denoise_step)
    
    # _logger.info("-" * 60)
    # _logger.info("[Scenario 2: Inference (Test) one rollout step]")
    # _logger.info(f"  (Denoise Steps = {args.denoise_step})")
    # _logger.info(f"  Total Inference FLOPS: \033[93m{test_one_rollout_step / G:.6f} GFLOPs\033[0m")
    # _logger.info(f"  Breakdown: Embedding ({f_embed/G:.4f} G) + {args.denoise_step} x Denoise ({args.denoise_step * f_denoise/G:.4f} G)")
    # _logger.info("-" * 60)


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
