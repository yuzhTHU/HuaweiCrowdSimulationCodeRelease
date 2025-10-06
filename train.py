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
    ]
    train_dataset = [d for d in dataset_list if d.name != 'hotel']
    test_dataset = [d for d in dataset_list if d.name == 'hotel']
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
        f"Train: {[d.name for d in train_dataset]} datasets, ({sum([len(d) for d in train_dataset])} samples)"
        f"Test: {[d.name for d in test_dataset]} datasets, ({sum([len(d) for d in test_dataset])} samples)"
    )

    # Load Model
    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    ddim = DDIM(args)

    # Train
    for epoch in range(args.epochs):
        torch.set_grad_enabled(True)
        model.train()
        for loader in train_loaders:
            # _logger.info(f"[Epoch {epoch+1}/{args.epochs}] Start training on {loader.dataset.name} dataset")
            map_data = loader.dataset.map_data
            map = torch.from_numpy(map_data.map).to(args.device).float()
            total_loss = 0.0
            for batch in tqdm(loader, total=len(loader), disable=True):
                pos = batch['pos'].to(args.device)  # (batch_size, #pedestrian, 2)
                vel = batch['vel'].to(args.device)  # (batch_size, #pedestrian, 2)
                hst = batch['hst'].to(args.device)  # (batch_size, #pedestrian, hist_step, 2)
                des = batch['des'].to(args.device)  # (batch_size, #pedestrian, 2)
                spd = batch['spd'].to(args.device)  # (batch_size, #pedestrian)
                veh = batch['veh'].to(args.device)  # (batch_size, #vehicle, hist_step + 1, 2)
                acc = batch['acc'].to(args.device)  # (batch_size, #pedestrian, pred_step, 2)
                ped_length = batch['ped_length'].to(args.device)  # (batch_size,)
                veh_length = batch['veh_length'].to(args.device)  # (batch_size,)

                # DDPM forward
                noisy_acc, noise_true, denoise_t = ddim.add_noise(acc)

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
                noise_pred = model(
                    noisy_acc=noisy_acc, denoise_t=denoise_t,
                    ped_length=ped_length, veh_length=veh_length
                )  # (B, #pedestrian, pred_step, 2)

                # Compute Loss & Backpropagate
                loss = criterion(noise_pred, noise_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * acc.shape[0]
            _logger.info(f"[Epoch {epoch+1}/{args.epochs}] Loss={total_loss/len(loader.dataset):.4f}")
        
        if not (epoch + 1) % 10:
            torch.set_grad_enabled(False)
            model.eval()
            for loader in test_loaders:
                _logger.info(f"[Epoch {epoch+1}/{args.epochs}] Start evaluating on {loader.dataset.name} dataset")
                map_data = loader.dataset.map_data
                map = torch.from_numpy(map_data.map).to(args.device).float()
                model.set_map_embedding(
                    map=map,
                    xmin=map_data.xmin,
                    xmax=map_data.xmax,
                    ymin=map_data.ymin,
                    ymax=map_data.ymax,
                )
                loss_list = []
                ade_list = []
                fde_list = []
                trajlen_list = []
                for batch in tqdm(loader, total=len(loader), disable=False):
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

                    model.set_veh_embedding(veh=veh)
                    model.set_ped_embedding(pos=pos, vel=vel, hst=hst, des=des, spd=spd)
                    model.set_sur_info()

                    x_t = torch.randn(acc.shape, device=args.device)  # 从噪声开始
                    assert args.T % 50 == 0, f"试图使用 {50} 步采样，然而训练步数 {args.T} % {50} 不等于 0!"
                    stride = args.T // 50
                    for t in tqdm(reversed(range(0, args.T, stride)), disable=True):
                        noisy_acc = x_t
                        denoise_t = torch.full((x_t.shape[0],), t, device=args.device, dtype=torch.long)
                        noise_pred = model(
                            noisy_acc=noisy_acc, denoise_t=denoise_t,
                            ped_length=ped_length, veh_length=veh_length,
                        )  # (B, #pedestrian, 2)
                        x_t = ddim.denoise(x_t, t, noise_pred, stride=stride)
                    
                    acc_pred = x_t
                    acc_true = acc # (B, #pedestrian, pred_step, 2)
                    vel_true = vel.unsqueeze(-2) + acc_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, pred_step, 2)
                    pos_true = pos.unsqueeze(-2) + vel_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, pred_step, 2)

                    _logger.debug(f"{np.nanmax((pos_true - future).cpu().abs().numpy()):.4f}")

                    loss = criterion(acc_pred, acc_true) # float
                    loss_list.extend([loss.item()] * acc.shape[0]) # List[float]

                    vel_pred = vel.unsqueeze(-2) + acc_pred.cumsum(dim=-2) / args.fps
                    pos_pred = pos.unsqueeze(-2) + vel_pred.cumsum(dim=-2) / args.fps
                    dis_err = (pos_pred - pos_true).norm(dim=-1) # (B, #pedestrian, pred_step)
                    ade = dis_err.mean(dim=-1) # (B, #pedestrian)
                    fde = dis_err[..., -1] # (B, #pedestrian)
                    ade_list.extend(ade.cpu().tolist()) # List[List[float]]
                    fde_list.extend(fde.cpu().tolist()) # List[List[float]]
                    trajlen = pos_true.diff(dim=-2).norm(dim=-1).sum(dim=-1) # (B, #pedestrian)
                    trajlen_list.extend(trajlen.cpu().tolist()) # List[List[float]]
                _logger.info(
                    f"[Epoch {epoch+1}/{args.epochs}] Eval "
                    f"Loss={np.mean(loss_list):.4f} "
                    f"ADE={np.mean(sum(ade_list, [])):.4f} "
                    f"FDE={np.mean(sum(fde_list, [])):.4f} "
                    f"AvgLen={np.mean(sum(trajlen_list, [])):.4f} "
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
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument("--num_workers", type=int, default=0)
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
