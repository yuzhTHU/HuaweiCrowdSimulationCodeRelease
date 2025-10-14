import json
import torch
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
from src.model.model import Model
from src.diffusion import DDPM, DDIM
from src.utils.logger import init_logger
from src.utils.seed import seed_all
from src.utils.timer import NamedTimer
from src.utils.plot import get_fig
from src.utils.auto_gpu import AutoGPU

_logger = logging.getLogger("src.train")


def main(args):
    ## Load Dataset
    # 加载数据集
    if args.debug:
        dataset_list = [UCYDataset.load_data(args, "./data/UCY/data/data_zara/crowds_zara01.vsp")]
    elif args.datasets == "ETH/UCY":
        dataset_list = [
            *UCYDataset.load_data_batch(args, "./data/UCY/data/"),
            *ETHDataset.load_data_batch(args, "./data/ETH/"),
        ]
    elif args.datasets == "GC":
        dataset_list = [GCDataset.load_data(args, "./data/GC/Annotation")]
    elif args.datasets == "SDD":
        dataset_list = SDDDataset.load_data_batch(args, "./data/SDD/annotations/")
    elif args.datasets == 'WayMo':
        dataset_list = WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/")
    elif args.datasets == "zara01":
        dataset_list = [UCYDataset.load_data(args, "./data/UCY/data/data_zara/crowds_zara01.vsp")]
        dataset_list[0].samples = dataset_list[0].samples[:1]
    else:
        raise ValueError(f"Unknown dataset {args.datasets}!")
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
    # 创建数据加载器
    train_loaders = []
    test_loaders = []
    for dataset in train_dataset:
        train_loaders.append(D.DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    for dataset in test_dataset:
        test_loaders.append(D.DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size // args.sample_num,  # 在实际测试时 batch_size 会乘上 sample_num，可能会很大导致 OOM
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    _logger.note(
        "Datasets:\n"
        f"Train on {[d.name for d in train_dataset]} datasets ({sum([len(d) for d in train_dataset]):,} samples in total)\n"
        f"Test on {[d.name for d in test_dataset]} datasets ({sum([len(d) for d in test_dataset]):,} samples in total)"
    )

    ## Load Model
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
        checkpoint_path = None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        _logger.note(f"Checkpoint loaded from {checkpoint_path}, resume from epoch {start_epoch}.")
        if "optimizer" in checkpoint:
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
            train_records = train_once(args, train_loaders, model, optimizer, criterion, diffusion, epoch)
            timer.add('train')
        else:
            train_records = None

        # 测试一个 epoch
        if (epoch > 0 and not epoch % args.test_per_epoch) or (epoch == 0 and args.test_before_train):
            torch.set_grad_enabled(False)
            model.eval()
            test_records = test_once(args, test_loaders, model, criterion, diffusion, epoch)
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
        if epoch % args.save_per_epoch == 0: # and timer.time > 300:
            # 只在运行超过 5 min 时保存
            save_path = f"{args.save_path}/checkpoint.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_path)
            _logger.note(f"Checkpoint saved to {save_path}")
            timer.add('save_checkpoint')
        
        # 定期保存
        if set(str(epoch)[1:]) == {'0'}:
            # 只在 epoch=10,20,...,100,...,1000,... 时保存
            save_path = Path(args.save_path) / "checkpoints" / f"epoch{epoch}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
            }, save_path)
            _logger.note(f"Model saved to {save_path}")
            timer.add('save_periodly')

        # 保存最佳模型
        if test_records is not None:
            if (
                'best_records' not in locals() or 
                np.mean(test_records['ade']) < np.mean(best_records['ade'])
            ):
                patience = args.patience
                best_records = test_records
                save_path = f"{args.save_path}/best.pth"
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    # "optimizer": optimizer.state_dict(),
                }, save_path)
                _logger.note(f"Best model saved to {save_path}")
            else:
                patience -= 1
                _logger.info(f"Patience left: {patience}/{args.patience}")
            timer.add('save_best')

        # 打印用时
        _logger.info(f"[Epoch {epoch}/{args.epochs}] finished. Time Usage={timer}")
        
        # 提前终止
        if 'patience' in locals() and patience <= 0:
            _logger.warning(
                f"Early stopping at epoch {epoch}, "
                f"best ADE={np.mean(best_records['ade']):.4f} at epoch {best_records['epoch']}, "
                f"FDE={np.mean(best_records['fde']):.4f}, "
                f"AvgLen={np.mean(best_records['trajlen']):.4f}, "
                f"Loss={np.mean(best_records['loss']):.4f}, "
                f"PedNum={np.mean(best_records['ped_num']):.1f}, "
                f"VehNum={np.mean(best_records['veh_num']):.1f}. "
            )
            break

    _logger.note("Training finished.")


def train_once(args, train_loaders, model, optimizer, criterion, diffusion, epoch):
    train_timer = NamedTimer(unit='it', mode='pace')
    records_list = []
    for loader in train_loaders:
        map_data = loader.dataset.map_data
        map = torch.from_numpy(map_data.map).to(args.device).float()
        records = dict(loss=[])
        for batch in tqdm(loader, total=len(loader), disable=False, leave=False, dynamic_ncols=True):
            pos = batch['pos'].to(args.device)  # (batch_size, #pedestrian, 2)
            vel = batch['vel'].to(args.device)  # (batch_size, #pedestrian, 2)
            hst = batch['hst'].to(args.device)  # (batch_size, #pedestrian, hist_step, 2)
            des = batch['des'].to(args.device)  # (batch_size, #pedestrian, 2)
            spd = batch['spd'].to(args.device)  # (batch_size, #pedestrian)
            veh = batch['veh'].to(args.device)  # (batch_size, #vehicle, hist_step + 1, 2)
            future_acc = batch['future_acc'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            ped_length = batch['ped_length'].to(args.device)  # (batch_size,)
            veh_length = batch['veh_length'].to(args.device)  # (batch_size,)
            train_timer.add('prepare data')

            # DDPM forward
            acc_true = future_acc[:, :, :args.pred_step, :] # (B, #pedestrian, pred_step, 2)
            noisy_acc, noise_true, denoise_t = diffusion.add_noise(acc_true)
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
            output = model(
                noisy_acc=noisy_acc, denoise_t=denoise_t,
                ped_length=ped_length, veh_length=veh_length,
            )  # (B, #pedestrian, pred_step, 2)
            train_timer.add('DDPM backword')

            # Compute Loss & Backpropagate
            if args.predict_noise:
                loss = criterion(output, noise_true)
            else:
                loss = criterion(output, acc_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            records['loss'].extend([loss.item()] * acc_true.shape[0])
            train_timer.add('backpropagate')
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
            all_records[k].extend(v)
    _logger.info(f"[Epoch {epoch}/{args.epochs}] Loss={np.mean(all_records['loss']):.4f}, Time={train_timer}")
    return all_records


def test_once(args, test_loaders, model, criterion, diffusion, epoch):
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
        records = dict(loss=[], ade=[], fde=[], trajlen=[], ped_num=[], veh_num=[])
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
            for step in range(args.roll_step):
                model.set_veh_embedding(veh=veh_now)
                model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
                model.set_sur_info()
                test_timer.add('embed data')

                shape = list(future_acc.shape)
                shape[0] *= S
                shape[2] = args.pred_step
                xt = torch.randn(shape, device=args.device)  # 从噪声开始
                for_plot.append([xt])
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
                    for_plot[-1].append(xt)
                acc_new = xt  # (S*B, #pedestrian, pred_step, 2)
                acc_pred.append(acc_new)
                test_timer.add('denoise')

                vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
                pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
                veh_new = future_veh[:, :, step*args.pred_step:(step+1)*args.pred_step, :].repeat(S, 1, 1, 1)  # (S*B, #vehicle, pred_step, 2)
                pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
                vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)
                hst_now = torch.cat([hst_now[:, :, args.pred_step:, :], pos_new], dim=-2) # (S*B, #pedestrian, hist_step, 2)
                veh_now = torch.cat([veh_now[:, :, args.pred_step:, :], veh_new], dim=-2)  # (S*B, #vehicle, hist_step + 1, 2)
                test_timer.add('rollout')

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
            if batch_idx == 0:
                pid = 0
                save_path = f"{args.save_path}/visualize/epoch{epoch}_{loader.dataset.name}_idx{batch_idx}_pid{pid}.png"
                visualize(args, pos, vel, hst, for_plot, mask, pos_true, pos_pred, save_path, pid)
                test_timer.add('visualize')
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
            test_timer.add('evaluate', n=0)
        records_list.append(records)
        _logger.info(
            f"[Epoch {epoch}/{args.epochs}] Eval on {loader.dataset.name}: "
            f"Loss={np.mean(records['loss']):.4f}, "
            f"ADE={np.mean(records['ade']):.4f}, "
            f"FDE={np.mean(records['fde']):.4f}, "
            f"AvgLen={np.mean(records['trajlen']):.4f}, "
            f"PedNum={np.mean(records['ped_num']):.1f}, "
            f"VehNum={np.mean(records['veh_num']):.1f}, "
        )
    all_records = {
        'epoch': epoch,
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
    _logger.note(
        f"[Epoch {epoch}/{args.epochs}] Overall: "
        f"Loss={np.sum(w * all_records['loss']):.4f}, "
        f"ADE={np.sum(w * all_records['ade']):.4f}, "
        f"FDE={np.sum(w * all_records['fde']):.4f}, "
        f"AvgLen={np.sum(w * all_records['trajlen']):.4f}, "
        f"PedNum={np.sum(w * all_records['ped_num']):.4f}, "
        f"VehNum={np.sum(w * all_records['veh_num']):.4f}, "
        f"Time={test_timer}"
    )
    return all_records


def visualize(args, pos, vel, hst, for_plot, mask, pos_true, pos_pred, save_path, pid):
    S = args.sample_num
    N = args.denoise_step
    batch_size, ped_num = pos.shape[:2]
    for_plot_acc = torch.concat([torch.stack(i, dim=0) for i in for_plot], dim=-2)  # (N+1, S*B, #pedestrian, roll_step*pred_step, 2)
    for_plot_acc = for_plot_acc.view(N+1, S, batch_size, ped_num, args.roll_step*args.pred_step, 2)  # (N+1, S, B, #pedestrian, roll_step*pred_step, 2)
    for_plot_vel = vel.unsqueeze(-2) + for_plot_acc.cumsum(dim=-2) / args.fps  # (N+1, S, B, #pedestrian, roll_step*pred_step, 2)
    for_plot_pos = pos.unsqueeze(-2) + for_plot_vel.cumsum(dim=-2) / args.fps  # (N+1, S, B, #pedestrian, roll_step*pred_step, 2)
    fi, fig, axes = get_fig(3, 4, AW=6, AH=6, dpi=300)
    if N + 1 <= 11:
        loader = range(N+1)
    else:
        loader = np.linspace(0, N+1, 12, dtype=int)[:-1].tolist()
    for idx, n in enumerate(loader):
        ax = axes[idx]
        ax.plot(*hst[mask, :, :][pid].cpu().numpy().T, color='blue', lw=1.0) # 历史轨迹
        ax.scatter(*pos[mask, :][pid].cpu().numpy(), color='blue') # 当前位置
        ax.plot(*pos_true[mask, :, :][pid].cpu().numpy().T, 'r.:', markevery=args.pred_step, lw=1.0) # 未来轨迹
        ax.title.set_text(f"Step {N-n} / {N}")
        for line in for_plot_pos[n, :, mask, :, :][:, pid]: # 逐步的扩散结果
            ax.plot(*line.cpu().numpy().T)
    ax = axes[-1]
    ax.plot(*hst[mask, :, :][pid].cpu().numpy().T, color='blue', lw=1.0) # 历史轨迹
    ax.scatter(*pos[mask, :][pid].cpu().numpy(), color='blue') # 当前位置
    ax.plot(*pos_true[mask, :, :][pid].cpu().numpy().T, 'r.:', markevery=args.pred_step, lw=1.0) # 未来轨迹
    for line in pos_pred[:, mask, :, :][:, pid]: # 最终的采样结果
        ax.plot(*line.cpu().numpy().T)
    for ax in axes:
        ax.axis('equal')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    del fi, fig, axes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--sampling_method', type=str, default="DDIM", choices=['DDPM', 'DDIM'])
    parser.add_argument("--T", type=int, default=100, help="训练时的扩散步数")
    parser.add_argument('--sample_num', type=int, default=20, help="测试时每个轨迹采样 {sample_num} 次")
    parser.add_argument('--denoise_step', type=int, default=10, help="采样时进行 {denoise_step} 次去噪")
    parser.add_argument('--step_offset', type=int, default=1, help="最后一步去噪从 x_{step_offset} 到 x_0")
    parser.add_argument('--no_antithetic_sampling', action='store_false', dest='antithetic_sampling', default=True)
    parser.add_argument("--hist_step", type=int, default=8)
    parser.add_argument("--pred_step", type=int, default=1)
    parser.add_argument("--skip_step", type=int, default=1)
    parser.add_argument("--roll_step", type=int, default=12)
    parser.add_argument("--fps", type=int, default=2.5)
    parser.add_argument("--dot_per_meter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./logs/train")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--map_feature_dim', type=int, default=64)
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--latent_token_num', type=int, default=16)
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--datasets', type=str, default="ETH/UCY", choices=['ETH/UCY', 'GC', 'SDD', 'WayMo', 'zara01', 'debug'])
    parser.add_argument('--test_name', type=str, default=None, nargs='+')
    parser.add_argument('--test_ratio', type=float, default=None)
    parser.add_argument('--split_by_scenario', action='store_true')
    parser.add_argument('--no_cache_dataset', dest='cache_dataset', action='store_false', default=True)
    parser.add_argument('--test_before_train', action='store_true')
    parser.add_argument('--test_per_epoch', type=int, default=10)
    parser.add_argument('--save_per_epoch', type=int, default=50)
    parser.add_argument('--reload_checkpoint', type=str, default=None, help='/path/to/checkpoint.pth')
    parser.add_argument('--no_predict_noise', action='store_false', dest='predict_noise', default=True)
    args, unknown = parser.parse_known_args()

    ## Build Save Path
    if args.exp_name is None:
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        time = now.strftime("%H%M%S")
        host = gethostname()
        args.exp_name = f'{date}_{args.name}_{time}_{host}'
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            args.name = args.name.replace(char, '_')
    save_path = Path(args.save_dir) / args.exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    ## Set Seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed_all(args.seed)

    ## Init Logger
    init_logger(
        "src",
        exp_name=args.exp_name,
        log_file=save_path / "info.log",
        info_level="debug" if args.debug else "info",
    )

    ## Select GPU
    if args.device == "auto":
        args.device = AutoGPU().choice_gpu(memory_MB=6000, interval=15)

    ## Save Command
    args.command = ' '.join(sys.argv)

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
