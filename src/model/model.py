import math
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from src.utils.timer import NamedTimer

_logger = logging.getLogger(__name__)


class NanEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)  # 正常数值的映射
        self.nan_embed = nn.Parameter(torch.randn(embed_dim))  # 用于 nan 的可学习向量

    def forward(self, x):
        nan_mask = torch.isnan(x).all(dim=-1)
        x = torch.nan_to_num(x, nan=0.0)
        out = self.embed(x)  # (N, L, D)
        out[nan_mask, :] = self.nan_embed
        return out


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.LongTensor):
        # sinusoidal position encoding
        half_dim = self.embed_dim // 2
        freq = torch.exp(
            -torch.arange(half_dim, device=t.device).float()
            * (math.log(10000.0) / (half_dim - 1))
        )
        emb = t[:, None].float() * freq[None]  # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch, embed_dim)
        return self.mlp(emb)  # (batch, embed_dim)


class MeanPoolingLSTM(nn.Module):
    def __init__(self, input_dim, embed_dim, layer_num):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=embed_dim,
            num_layers=layer_num,
            batch_first=True,
        )

    def forward(self, x):
        """x: (batch_size, #pedestrain/#vehicle, seq_len, input_dim)"""
        shape = x.shape
        x = x.view(-1, *shape[-2:])  # (batch_size * N, seq_len, input_dim)
        out, _ = self.lstm(x)  # (batch_size, seq_len, embed_dim)
        out = out.mean(dim=-2)  # (batch_size, embed_dim)
        out = out.view(*shape[:-2], shape[-1])  # (batch_size, N, embed_dim)
        return out


class Permuted(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Residual(nn.Module):
    def __init__(self, *layers, input_dim=None, output_dim=None):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.need_proj = (
            input_dim is not None and output_dim is not None and input_dim != output_dim
        )
        if self.need_proj:
            self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.need_proj:
            return self.proj(x) + self.net(x)
        else:
            return x + self.net(x)


class FourierPositionalEncoding(nn.Module):
    def __init__(self, out_dim: int = 256, num_bands: int = 64, max_freq: float = 10.0):
        """
        将标量 x ∈ R 编码到高维空间
        Args:
            out_dim: 最终编码维度
            num_bands: Fourier Feature 的频率数量
            max_freq: 最大频率
        """
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.freqs = torch.linspace(1.0, max_freq, num_bands)
        self.proj = nn.Linear(num_bands * 4, out_dim)

    def forward(self, x: torch.Tensor):
        """x: (..., 2)"""
        freqs = self.freqs.to(x.device) * math.pi * 2
        x_proj = x[..., (0,)] * freqs  # (..., num_bands)
        y_proj = x[..., (1,)] * freqs  # (..., num_bands)
        fourier = torch.cat([
            torch.sin(x_proj), 
            torch.cos(x_proj),
            torch.sin(y_proj),
            torch.cos(y_proj),
        ], dim=-1)  # (..., num_bands*4)
        pe = self.proj(fourier)
        return pe


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.denoise_t_embedder = SinusoidalEmbedding(args.model_dim)
        self.noisy_acc_embedder = nn.Sequential(
            nn.Linear(2, args.model_dim),
            MeanPoolingLSTM(args.model_dim, args.model_dim, args.lstm_layer_num),
            nn.LayerNorm(args.model_dim),
        )
        self.positional_encoding = FourierPositionalEncoding(out_dim=args.model_dim, num_bands=args.model_dim, max_freq=10.0)

        self.pos_embedder = nn.Sequential(
            nn.Linear(2, args.model_dim),
            nn.LayerNorm(args.model_dim),
        )
        self.vel_embedder = nn.Sequential(
            nn.Linear(2, args.model_dim),
            nn.LayerNorm(args.model_dim),
        )
        self.hst_embedder = nn.Sequential(
            NanEmbedding(2, args.model_dim),
            MeanPoolingLSTM(args.model_dim, args.model_dim, args.lstm_layer_num),
            nn.LayerNorm(args.model_dim),
        )
        self.des_embedder = nn.Sequential(
            NanEmbedding(2, args.model_dim),
            nn.LayerNorm(args.model_dim),
        )
        self.spd_embedder = nn.Sequential(
            NanEmbedding(1, args.model_dim),
            nn.LayerNorm(args.model_dim),
        )
        self.ped_encoder = nn.Sequential(
            nn.LayerNorm(args.model_dim),
            nn.Linear(args.model_dim, 4*args.model_dim),
            nn.ReLU(),
            nn.Linear(4*args.model_dim, args.model_dim),
        )

        self.veh_embedder = nn.Sequential(
            NanEmbedding(2, args.model_dim),
            MeanPoolingLSTM(args.model_dim, args.model_dim, args.lstm_layer_num),
            nn.LayerNorm(args.model_dim),
        )

        self.map_embedder = nn.Sequential(
            NanEmbedding(1, args.map_feature_dim//4),
            Permuted(2, 0, 1),  # (H, W, C) -> (C, H, W)
            nn.Conv2d(args.map_feature_dim//4, args.map_feature_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.map_feature_dim//2, args.map_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.map_feature_dim, args.model_dim, kernel_size=1),  # 1x1 卷积，相当于每个 (h,w) 位置的 Linear(C->Df)
            Permuted(1, 2, 0),  # (C, H, W) -> (H, W, C)
            nn.LayerNorm(args.model_dim),
        )

        self.ped_attention = nn.TransformerDecoder(
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
        self.veh_attention = nn.TransformerDecoder(
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
        self.map_attention = nn.TransformerDecoder(
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
        self.latent_attntn = nn.TransformerDecoder(
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
        self.latent_tokens = nn.Parameter(
            torch.randn(args.latent_token_num, args.model_dim)
        )

        self.fusion_fc = Residual(
            nn.LayerNorm(args.model_dim),
            nn.Linear(args.model_dim, 4*args.model_dim),
            nn.ReLU(),
            nn.Linear(4*args.model_dim, args.model_dim),
        )
        self.output_fc = Residual(
            nn.LayerNorm(2*args.model_dim),
            nn.Linear(2*args.model_dim, args.model_dim),
            nn.ReLU(),
            nn.Linear(args.model_dim, args.model_dim),
            nn.ReLU(),
            nn.Linear(args.model_dim, args.pred_step*2),
            input_dim=2*args.model_dim, 
            output_dim=args.pred_step*2,
        )

    def set_ped_embedding(
        self,
        pos: torch.FloatTensor, 
        vel: torch.FloatTensor,
        hst: torch.FloatTensor,
        des: torch.FloatTensor,
        spd: torch.FloatTensor,
    ):
        pos_embedding = self.pos_embedder(pos) # (batch_size, #pedestrian, model_dim)
        vel_embedding = self.vel_embedder(vel) # (batch_size, #pedestrian, model_dim)
        hst_embedding = self.hst_embedder(hst) # (batch_size, #pedestrian, model_dim)
        des_embedding = self.des_embedder(des) # (batch_size, #pedestrian, model_dim)
        spd_embedding = self.spd_embedder(spd) # (batch_size, #pedestrian, model_dim)
        ped_embedding = pos_embedding + vel_embedding + hst_embedding + des_embedding + spd_embedding
        self.ped_embedding = ped_embedding
        self.pos = pos

        pe = self.positional_encoding(pos) # (batch_size, #pedestrian, model_dim)
        self.pe = pe

    def set_veh_embedding(
        self,
        veh: torch.FloatTensor,
    ):
        shape = list(veh.shape)
        if shape[1] == 0:
            shape[1] = 1
            veh = torch.full(shape, float('nan'), device=veh.device)
        veh_embedding = self.veh_embedder(veh) # (batch_size, #vehicle, model_dim)
        self.veh_embedding = veh_embedding

    def set_map_embedding(
        self,
        map: torch.FloatTensor,
        xmin: torch.FloatTensor, 
        xmax: torch.FloatTensor, 
        ymin: torch.FloatTensor, 
        ymax: torch.FloatTensor,
    ):
        map_embedding = self.map_embedder(map.unsqueeze(-1)) # (H', W', model_dim)
        xx = torch.linspace(xmin, xmax, map_embedding.size(1), device=map_embedding.device)
        yy = torch.linspace(ymin, ymax, map_embedding.size(0), device=map_embedding.device)
        gridx, gridy = torch.meshgrid(xx, yy, indexing='xy')
        gridxy = torch.stack([gridx, gridy], dim=-1) # (H', W', 2)
        map_embedding = map_embedding + self.positional_encoding(gridxy) # (H', W', model_dim)
        ltn_embedding = self.latent_attntn(self.latent_tokens, map_embedding.flatten(0, 1)) # (#latent_token, model_dim)
        self.map_embedding = map_embedding
        self.ltn_embedding = ltn_embedding
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin

    def set_sur_info(self):
        pos = self.pos
        xmax, xmin = self.xmax, self.xmin
        ymax, ymin = self.ymax, self.ymin
        map_embedding = self.map_embedding
        idx = pos[..., 1].sub(ymin).div(ymax-ymin).mul(map_embedding.size(0)).round().long().clamp(0, map_embedding.size(0) - 1)  # (batch_size, #pedestrian)
        jdx = pos[..., 0].sub(xmin).div(xmax-xmin).mul(map_embedding.size(1)).round().long().clamp(0, map_embedding.size(1) - 1)  # (batch_size, #pedestrian)
        idx = map_embedding.size(0) - 1 - idx # (batch_size, #pedestrian)
        sur_info = map_embedding[idx, jdx] # (batch_size, #pedestrian, model_dim)
        sur_info = F.layer_norm(sur_info, sur_info.shape[-1:])
        self.sur_info = sur_info

    def forward(
        self, 
        # pos: torch.FloatTensor, 
        # vel: torch.FloatTensor,
        # hst: torch.FloatTensor,
        # des: torch.FloatTensor,
        # spd: torch.FloatTensor,
        # veh: torch.FloatTensor,
        # map: torch.FloatTensor,
        # xmin: torch.FloatTensor, 
        # xmax: torch.FloatTensor, 
        # ymin: torch.FloatTensor, 
        # ymax: torch.FloatTensor,
        denoise_t: torch.LongTensor,
        noisy_acc: torch.FloatTensor,
        ped_length: torch.LongTensor,
        veh_length: torch.LongTensor,
        timer: NamedTimer = None,
    ):
        """根据行人、车辆、场景信息对行人下一步加速度 acc 进行去噪
        Args:
            pos (torch.FloatTensor): 行人当前位置 (batch_size, #pedestrian, 2)
            vel (torch.FloatTensor): 行人当前速度 (batch_size, #pedestrian, 2)
            hst (torch.FloatTensor): 行人历史轨迹 (batch_size, #pedestrian, hist_step, 2)
            des (torch.FloatTensor): 行人终点位置 (batch_size, #pedestrian, 2)
            spd (torch.FloatTensor): 行人预期速度 (batch_size, #pedestrian, 1)
            veh (torch.FloatTensor): 车辆历史轨迹 (batch_size, #vehicle, hist_step + 1, 2)
            map (torch.FloatTensor): 场景高度地图 (H, W)
            xmin (float): 地图x轴最小值, 与 pos & veh 处于同一坐标系
            xmax (float): 地图x轴最大值, 与 pos & veh 处于同一坐标系
            ymin (float): 地图y轴最小值, 与 pos & veh 处于同一坐标系
            ymax (float): 地图y轴最大值, 与 pos & veh 处于同一坐标系
            denoise_t (torch.LongTensor): 当前去噪时间步 (batch_size,)
            noisy_acc (torch.FloatTensor): （带噪的）行人下步加速度 (batch_size, #pedestrian, pred_step, 2)
            ped_length (torch.LongTensor): 每个batch中行人数量 (batch_size,)
            veh_length (torch.LongTensor): 每个batch中车辆数量 (batch_size,)
        """
        # Embedding Pedestrian
        # pos_embedding = self.pos_embedder(pos) # (batch_size, #pedestrian, model_dim)
        # vel_embedding = self.vel_embedder(vel) # (batch_size, #pedestrian, model_dim)
        # hst_embedding = self.hst_embedder(hst) # (batch_size, #pedestrian, model_dim)
        # des_embedding = self.des_embedder(des) # (batch_size, #pedestrian, model_dim)
        # spd_embedding = self.spd_embedder(spd) # (batch_size, #pedestrian, model_dim)
        # ped_embedding = pos_embedding + vel_embedding + hst_embedding + des_embedding + spd_embedding
        ped_embedding = self.ped_embedding
        denoise_t_embedding = self.denoise_t_embedder(denoise_t) # (batch_size, model_dim)
        denoise_t_embedding = denoise_t_embedding.unsqueeze(1) # (batch_size, 1, model_dim)
        noisy_acc_embedding = self.noisy_acc_embedder(noisy_acc) # (batch_size, #pedestrian, model_dim)
        ped_embedding = self.ped_encoder(ped_embedding + denoise_t_embedding + noisy_acc_embedding) # (batch_size, #pedestrian, model_dim)
        if timer: timer.add('Embedding Pedestrian')

        # Embedding Vehicle
        # veh_embedding = self.veh_embedder(veh) # (batch_size, #vehicle, model_dim)
        veh_embedding = self.veh_embedding

        # Embedding Map
        # map_embedding = self.map_embedder(map.unsqueeze(-1)) # (H', W', model_dim)
        # xx = torch.linspace(xmin, xmax, map_embedding.size(1), device=map_embedding.device)
        # yy = torch.linspace(ymin, ymax, map_embedding.size(0), device=map_embedding.device)
        # gridx, gridy = torch.meshgrid(xx, yy, indexing='xy')
        # gridxy = torch.stack([gridx, gridy], dim=-1) # (H', W', 2)
        # map_embedding = map_embedding + self.positional_encoding(gridxy) # (H', W', model_dim)
        # ltn_embedding, _ = self.latent_attntn(self.latent_tokens, map_embedding.flatten(0, 1), map_embedding.flatten(0, 1)) # (#latent_token, model_dim)
        # ltn_embedding = ltn_embedding.unsqueeze(0).expand(ped_embedding.size(0), *ltn_embedding.shape) # (batch_size, #latent_token, model_dim)
        ltn_embedding = self.ltn_embedding.unsqueeze(0).expand(ped_embedding.size(0), *self.ltn_embedding.shape) # (batch_size, #latent_token, model_dim)

        # Build Mask
        batch_size = ped_embedding.size(0)
        max_ped_num = ped_embedding.size(1)
        max_veh_num = veh_embedding.size(1)
        ped_mask = torch.arange(max_ped_num, device=ped_length.device).unsqueeze(0).expand(batch_size, max_ped_num) # (batch_size, max_ped_num)
        ped_mask = ped_mask >= ped_length.unsqueeze(1) # (batch_size, max_ped_num)
        veh_mask = torch.arange(max_veh_num, device=veh_length.device).unsqueeze(0).expand(batch_size, max_veh_num) # (batch_size, max_veh_num)
        veh_mask = veh_mask >= veh_length.unsqueeze(1) # (batch_size, max_veh_num)
        if timer: timer.add('Build Mask')

        # Social Attention
        ped_info = self.ped_attention(
            ped_embedding, ped_embedding,
            memory_key_padding_mask=ped_mask,
            tgt_key_padding_mask=ped_mask,
        ) # (batch_size, #pedestrian, model_dim)
        ped_info = F.layer_norm(ped_info, ped_info.shape[-1:])
        if timer: timer.add('Social Attention')

        # Vehicle Attention
        veh_info = self.veh_attention(
            ped_embedding, veh_embedding,
            memory_key_padding_mask=veh_mask,
            tgt_key_padding_mask=ped_mask,
        ) # (batch_size, #pedestrian, model_dim)
        veh_info = F.layer_norm(veh_info, veh_info.shape[-1:])
        if timer: timer.add('Vehicle Attention')

        # Map Attention
        pe = self.pe
        map_info = self.map_attention(
            ped_embedding + pe, ltn_embedding,
            tgt_key_padding_mask=ped_mask,
        ) # (batch_size, #pedestrian, model_dim)
        map_info = F.layer_norm(map_info, map_info.shape[-1:])
        if timer: timer.add('Map Attention')

        # Surrounding Info
        # idx = pos[..., 1].sub(ymin).div(ymax-ymin).mul(map_embedding.size(0)).round().long().clamp(0, map_embedding.size(0) - 1)  # (batch_size, #pedestrian)
        # jdx = pos[..., 0].sub(xmin).div(xmax-xmin).mul(map_embedding.size(1)).round().long().clamp(0, map_embedding.size(1) - 1)  # (batch_size, #pedestrian)
        # idx = map_embedding.size(0) - 1 - idx # (batch_size, #pedestrian)
        # sur_info = map_embedding[idx, jdx] # (batch_size, #pedestrian, model_dim)
        # sur_info = F.layer_norm(sur_info, sur_info.shape[-1:])
        sur_info = self.sur_info

        # Fusion
        ped_embedding = self.fusion_fc(
            ped_embedding + ped_info + veh_info + map_info + sur_info
        ) # (batch_size, #pedestrian, model_dim)
        # acc_embedding = self.acc_embedder(acc_true) # (batch_size, #pedestrian, model_dim)
        ped_embedding = torch.concat([
            ped_embedding,
            denoise_t_embedding + noisy_acc_embedding # + acc_embedding,
        ], dim=-1) # (batch_size, #pedestrian, 2*model_dim)
        if timer: timer.add('Fusion')

        # Output
        output = self.output_fc(ped_embedding) # (batch_size, #pedestrian, pred_step*2)
        output = output.view(*output.shape[:-1], self.args.pred_step, 2) # (batch_size, #pedestrian, pred_step, 2)
        if timer: timer.add('Output')
        return output
