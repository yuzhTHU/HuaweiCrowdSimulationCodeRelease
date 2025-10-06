import json
import torch
import random
import logging
import numpy as np
import torch.utils.data as D
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from src.dataset import ETHDataset, UCYDataset, SDDDataset
from src.model.model import Model
from src.diffusion import DDPM, DDIM
from src.utils.logger import init_logger
from src.utils.seed import seed_all

_logger = logging.getLogger("src.train")


def main(args):
    # Load Dataset
    dataset_list = [
        # SDDDataset.load_data(args, "./data/SDD/annotations/bookstore/video1/annotations.txt"),
        # SDDDataset.load_data(args, "./data/SDD/annotations/bookstore/video2/annotations.txt"),
        *UCYDataset.load_data_batch(args, "./data/UCY/data/"),
        *ETHDataset.load_data_batch(args, "./data/ETH/"),
    ]
    train_dataset = [d for d in dataset_list if 'zara01' not in d.name]
    test_dataset = [d for d in dataset_list if 'zara01' in d.name]
    # train_dataset = dataset_list
    # test_dataset = dataset_list
    train_loaders = [
        D.DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        )
        for dataset in train_dataset
    ]
    test_loaders = [
        D.DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        )
        for dataset in test_dataset
    ]
    _logger.note(
        "Datasets:"
        f"Train on {[d.name for d in train_dataset]} datasets ({sum([len(d) for d in train_dataset])} samples in total)\n"
        f"Test on {[d.name for d in test_dataset]} datasets ({sum([len(d) for d in test_dataset])} samples in total)"
    )

    # Load Model
    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    ddim = DDIM(args)
    _logger.note(
        "Trainable Parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        f" ({sum(p.numel() for p in model.parameters()):,} in total)"
        f" on {args.device}"
    )

    # Train
    from src.utils.timer import NamedTimer
    train_timer = NamedTimer(unit='it', mode='pace')
    test_timer = NamedTimer(unit='it', mode='pace')
    for epoch in range(args.epochs+1):
        train_timer.add('drop')
        if epoch > 0:
            torch.set_grad_enabled(True)
            model.train()
            for loader in train_loaders:
                map_data = loader.dataset.map_data
                map = torch.from_numpy(map_data.map).to(args.device).float()
                total_loss = 0.0
                for batch in tqdm(loader, total=len(loader), disable=False, leave=False):
                    pos = batch['pos'].to(args.device)  # (batch_size, #pedestrian, 2)
                    vel = batch['vel'].to(args.device)  # (batch_size, #pedestrian, 2)
                    hst = batch['hst'].to(args.device)  # (batch_size, #pedestrian, hist_step, 2)
                    des = batch['des'].to(args.device)  # (batch_size, #pedestrian, 2)
                    spd = batch['spd'].to(args.device)  # (batch_size, #pedestrian)
                    veh = batch['veh'].to(args.device)  # (batch_size, #vehicle, hist_step + 1, 2)
                    acc = batch['acc'].to(args.device)  # (batch_size, #pedestrian, pred_step, 2)
                    ped_length = batch['ped_length'].to(args.device)  # (batch_size,)
                    veh_length = batch['veh_length'].to(args.device)  # (batch_size,)
                    train_timer.add('prepare data')

                    # DDPM forward
                    acc_true = acc # (B, #pedestrian, pred_step, 2)
                    noisy_acc, noise_true, denoise_t = ddim.add_noise(acc_true)
                    train_timer.add('DDPM forward')

                    # DDPM backward
                    model.set_map_embedding(
                        map=map,
                        xmin=map_data.xmin,
                        xmax=map_data.xmax,
                        ymin=map_data.ymin,
                        ymax=map_data.ymax,
                    )
                    model.set_veh_embedding(veh=veh)
                    model.set_ped_embedding(pos=pos, vel=vel, hst=hst, des=des, spd=spd)
                    model.set_sur_info()
                    acc_pred = model(
                        noisy_acc=noisy_acc, denoise_t=denoise_t,
                        ped_length=ped_length, veh_length=veh_length
                    )  # (B, #pedestrian, pred_step, 2)
                    train_timer.add('DDPM backword')

                    # Compute Loss & Backpropagate
                    loss = criterion(acc_pred, acc_true)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * acc_true.shape[0]
                    train_timer.add('backpropagate')
                _logger.info(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"Loss={total_loss/len(loader.dataset):.4f} on {loader.dataset.name} dataset "
                )
            _logger.info(
                f"[Epoch {epoch}/{args.epochs}] "
                f"{train_timer}"
            )
        
        if not (epoch % args.test_per_epoch) or (epoch == 0 and args.test_before_train):
            test_timer.add('drop')
            torch.set_grad_enabled(False)
            model.eval()
            for loader in test_loaders:
                _logger.info(f"[Epoch {epoch}/{args.epochs}] Start evaluating on {loader.dataset.name} dataset")
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
                loss_list = []
                ade_list = []
                fde_list = []
                trajlen_list = []
                for batch in tqdm(loader, total=len(loader), disable=False, leave=False):
                    pos = batch['pos'].to(args.device)  # (batch_size, #pedestrian, 2)
                    vel = batch['vel'].to(args.device)  # (batch_size, #pedestrian, 2)
                    hst = batch['hst'].to(args.device)  # (batch_size, #pedestrian, hist_step, 2)
                    des = batch['des'].to(args.device)  # (batch_size, #pedestrian, 2)
                    spd = batch['spd'].to(args.device)  # (batch_size, #pedestrian)
                    veh = batch['veh'].to(args.device)  # (batch_size, #vehicle, hist_step + 1, 2)
                    acc = batch['acc'].to(args.device)  # (batch_size, #pedestrian, pred_step, 2)
                    future = batch['future'].to(args.device)  # (batch_size, #pedestrian, pred_step, 2)
                    ped_length = batch['ped_length'].to(args.device)  # (batch_size,)
                    veh_length = batch['veh_length'].to(args.device)  # (batch_size,)
                    test_timer.add('prepare data', n=0)

                    model.set_veh_embedding(veh=veh)
                    model.set_ped_embedding(pos=pos, vel=vel, hst=hst, des=des, spd=spd)
                    model.set_sur_info()
                    test_timer.add('embed data')

                    S = 20  # 采样次数
                    N = 10  # 采样步数
                    acc_pred = []
                    for_plot = []
                    for _ in range(S):
                        xt = torch.randn(acc.shape, device=args.device)  # 从噪声开始
                        for_plot.append([xt])
                        assert args.T % N == 0, f"试图使用 {N} 步采样，然而训练步数 {args.T} % {N} 不等于 0!"
                        stride = args.T // N
                        for t in tqdm(range(args.T, 0, -stride), disable=True, leave=False):
                            noisy_acc = xt
                            denoise_t = torch.full((xt.shape[0],), t, device=args.device, dtype=torch.long)
                            x0_pred = model(
                                noisy_acc=noisy_acc, denoise_t=denoise_t,
                                ped_length=ped_length, veh_length=veh_length,
                            )  # (B, #pedestrian, 2)
                            xt = ddim.denoise(xt, t, x0_pred, stride=stride)
                            for_plot[-1].append(xt)
                        acc_pred.append(xt)
                    test_timer.add('denoise')
                    # 获取有效的行人掩模
                    batch_size, ped_num, _ = pos.shape
                    mask = torch.arange(ped_num, device=args.device).expand(batch_size, ped_num) < ped_length.unsqueeze(-1)  # (B, #pedestrian)
                    # 计算 pos_true 和 vel_true
                    acc_true = acc # (B, #pedestrian, pred_step, 2)
                    vel_true = vel.unsqueeze(-2) + acc_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, pred_step, 2)
                    pos_true = pos.unsqueeze(-2) + vel_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, pred_step, 2)
                    _logger.debug(f"pos_true 和 future 差距：{np.nanmax((pos_true - future).cpu().abs().numpy()):.4f}")
                    # 计算 loss
                    acc_pred = torch.stack(acc_pred, dim=0)  # (S, B, #pedestrian, pred_step, 2)
                    loss = criterion(acc_pred[0], acc_true) # float
                    loss_list.extend([loss.item()] * acc.shape[0]) # List[float]
                    # 计算 distance error
                    vel_pred = vel.unsqueeze(-2) + acc_pred.cumsum(dim=-2) / args.fps  # (S, B, #pedestrian, pred_step, 2)
                    pos_pred = pos.unsqueeze(-2) + vel_pred.cumsum(dim=-2) / args.fps  # (S, B, #pedestrian, pred_step, 2)
                    dis_err = (pos_pred - pos_true).norm(dim=-1) # (S, B, #pedestrian, pred_step)
                    test_timer.add('evaluate')
                    # 可视化
                    if False:
                        import matplotlib.pyplot as plt
                        from src.utils.plot import get_fig
                        for_plot = torch.stack([torch.stack(fp, dim=0) for fp in for_plot], dim=0)  # (S, N+1, B, #pedestrian, pred_step, 2)
                        for_plot = vel.unsqueeze(-2) + for_plot.cumsum(dim=-2) / args.fps  # (S, N+1, B, #pedestrian, pred_step, 2)
                        for_plot = pos.unsqueeze(-2) + for_plot.cumsum(dim=-2) / args.fps  # (S, N+1, B, #pedestrian, pred_step, 2)
                        fi, fig, axes = get_fig(3, 4, AW=6, AH=6, dpi=300)
                        pid = 0
                        for idx, n in enumerate(range(N+1)):
                            ax = axes[idx]
                            ax.plot(*hst[mask, :, :][pid].T.cpu().numpy(), color='blue') # 历史轨迹
                            ax.scatter(*pos[mask, :][pid].cpu().numpy(), color='blue') # 当前位置
                            ax.plot(*pos_true[mask, :, :][pid].T.cpu().numpy(), 'r:') # 未来轨迹
                            for line in for_plot[:, n, mask, :, :][:, pid]: # 逐步的扩散结果
                                ax.plot(*line.T.cpu().numpy())
                        ax = axes[11]
                        ax.plot(*hst[mask, :, :][pid].T.cpu().numpy(), color='blue') # 历史轨迹
                        ax.scatter(*pos[mask, :][pid].cpu().numpy(), color='blue') # 当前位置
                        ax.plot(*pos_true[mask, :, :][pid].T.cpu().numpy(), 'r:') # 未来轨迹
                        for line in pos_pred[:, mask, :, :][:, pid]: # 最终的采样结果
                            ax.plot(*line.T.cpu().numpy())
                        fig.savefig(f"{args.save_path}/eval_epoch{epoch}_{loader.dataset.name}_pid{pid}.png")
                        plt.close(fig)
                        del fi, fig, axes
                        test_timer.add('visualize')
                    # 移除 padding 的行人                    
                    dis_err = dis_err[:, mask, :] # (S, valid{B*#pedestrian}, pred_step)
                    # 选择 ade 最佳的 sample
                    sample_idx = dis_err.mean(dim=-1).argmin(dim=0)  # (valid{B*#pedestrian},)
                    batch_idx = torch.arange(dis_err.shape[1], device=args.device)  # (valid{B*#pedestrian},)
                    dis_err = dis_err[sample_idx, batch_idx, :]  # (valid{B*#pedestrian}, pred_step)
                    # 计算 ade, fde
                    ade = dis_err.mean(dim=-1) # (valid{B*#pedestrian})
                    fde = dis_err[..., -1] # (valid{B*#pedestrian})
                    ade_list.extend(ade.cpu().tolist()) # List[float]
                    fde_list.extend(fde.cpu().tolist()) # List[float]
                    # 计算轨迹长度
                    trajlen = pos_true.diff(dim=-2).norm(dim=-1).sum(dim=-1)[mask] # (valid{B*#pedestrian})
                    trajlen_list.extend(trajlen.cpu().tolist()) # List[float]
                    test_timer.add('evaluate', n=0)
                _logger.info(
                    f"[Epoch {epoch}/{args.epochs}] Eval "
                    f"Loss={np.mean(loss_list):.4f} "
                    f"ADE={np.mean(ade_list):.4f} "
                    f"FDE={np.mean(fde_list):.4f} "
                    f"AvgLen={np.mean(trajlen_list):.4f} "
                    f"({test_timer})"
                )

    _logger.note("Training finished.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="DDPM")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--hist_step", type=int, default=8)
    parser.add_argument("--pred_step", type=int, default=12)
    parser.add_argument("--skip_step", type=int, default=1)
    parser.add_argument("--fps", type=int, default=2.5)
    parser.add_argument("--dot_per_meter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./logs/train")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--map_feature_dim', type=int, default=64)
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--latent_token_num', type=int, default=16)
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--no_cache_dataset', dest='cache_dataset', action='store_false', default=True)
    parser.add_argument('--test_before_train', action='store_true')
    parser.add_argument('--test_per_epoch', type=int, default=10)
    args, unknown = parser.parse_known_args()

    # Build Save Path
    now = datetime.now()
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H%M%S")
    host = gethostname()
    args.name = f'{date}_{args.name}_{time}_{host}'
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        args.name = args.name.replace(char, '_')
    save_path = Path(args.save_dir) / args.name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    # Set Seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed_all(args.seed)

    # Init Logger
    init_logger(
        "src",
        exp_name=args.name,
        log_file=save_path / "info.log",
        info_level="debug" if args.debug else "info",
    )

    # Save Args
    _logger.note(f"Args: {args}")
    with open(save_path / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    # Start Training
    setproctitle(f"{args.name}@ZihanYu")
    main(args)
