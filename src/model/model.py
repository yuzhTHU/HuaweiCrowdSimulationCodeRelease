import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.timer import NamedTimer
from .sinusoidal_embedding import SinusoidalEmbedding
from .fourier_positional_encoding import FourierPositionalEncoding
from .residual import Residual
from .nan_embedding import NanEmbedding
from .permuted import Permuted
from .mean_pooling_lstm import MeanPoolingLSTM


class Model(nn.Module):
    """
    基于 Transformer 和 Diffusion 的行人轨迹预测模型。
    
    该模型融合了行人自身的历史状态、邻近行人的社交交互、周围车辆的交互
    以及静态地图环境信息，用于在扩散模型（DDPM/DDIM）的反向去噪过程中
    预测行人的运动意图（加速度或噪声）。
    """

    def __init__(self, args):
        """
        初始化模型层和各个嵌入模块。

        Args:
            args (Namespace): 配置参数对象，需包含以下关键参数：
                - model_dim (int): 模型内部特征维度 (Hidden Size)。
                - map_feature_dim (int): 地图特征提取的中间维度。
                - lstm_layer_num (int): 用于处理时序数据的 LSTM 层数。
                - head_num (int): 多头注意力机制的头数。
                - attention_layer_num (int): Transformer 解码器的层数。
                - latent_token_num (int): 用于压缩地图特征的 Latent Token 数量。
                - dropout (float): Dropout 比率。
                - pred_step (int): 预测步长。
                - use_spatial_anchor (bool): 是否使用空间锚点增强地图位置编码。
        """
        super().__init__()
        self.args = args

        self.denoise_t_embedder = SinusoidalEmbedding(args.model_dim)
        self.noisy_acc_embedder = nn.Sequential(
            nn.Linear(2, args.model_dim),
            MeanPoolingLSTM(args.model_dim, args.model_dim, args.lstm_layer_num),
            nn.LayerNorm(args.model_dim),
        )
        self.positional_encoding = FourierPositionalEncoding(out_dim=args.model_dim, num_bands=args.model_dim, min_freq=1e-3)

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
        if args.use_spatial_anchor:
            # 确保 token 数量是平方数 (e.g., 16, 64)
            grid_size = int(math.sqrt(args.latent_token_num))
            if grid_size ** 2 != args.latent_token_num:
                raise ValueError(f"latent_token_num ({args.latent_token_num}) must be a square number when use_spatial_anchor is True.")
            self.grid_size = grid_size
            # 生成 0~1 的相对坐标网格，用于后续映射到物理尺寸
            # 使用 buffer 注册，这样它会被保存到 state_dict 但不会作为参数更新
            x = torch.linspace(0, 1, grid_size)
            y = torch.linspace(0, 1, grid_size)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            anchor_norm = torch.stack([xx, yy], dim=-1) # (S, S, 2)
            self.register_buffer('anchor_norm', anchor_norm)
        self.fusion_fc = Residual(
            nn.LayerNorm(args.model_dim),
            nn.Linear(args.model_dim, 4*args.model_dim),
            nn.ReLU(),
            nn.Linear(4*args.model_dim, args.model_dim),
        )
        self.output_fc = Residual(
            nn.LayerNorm(args.model_dim),
            nn.Linear(args.model_dim, args.model_dim//2),
            nn.ReLU(),
            nn.Linear(args.model_dim//2, args.pred_step*2),
            input_dim=args.model_dim, 
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
        """
        计算并设置行人的综合特征嵌入 (Embedding)。
        
        该方法将行人的位置、速度、历史轨迹、目的地和期望速度分别映射到高维空间，
        并相加得到初始的行人特征向量。同时计算位置编码 (Positional Encoding)。

        Args:
            pos (torch.FloatTensor): 行人当前时刻的位置坐标 (x, y)。
                Shape: (batch_size, num_peds, 2)
            vel (torch.FloatTensor): 行人当前时刻的速度向量 (vx, vy)。
                Shape: (batch_size, num_peds, 2)
            hst (torch.FloatTensor): 行人的历史轨迹序列。
                Shape: (batch_size, num_peds, hist_step, 2)
            des (torch.FloatTensor): 行人的潜在目的地坐标。
                Shape: (batch_size, num_peds, 2)
            spd (torch.FloatTensor): 行人的期望速率标量。
                Shape: (batch_size, num_peds, 1)
        
        Side Effects:
            设置 self.ped_embedding: 融合后的行人特征 (batch_size, num_peds, model_dim)
            设置 self.pos: 缓存当前位置用于后续地图索引
            设置 self.pe: 位置编码特征
        """
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
        """
        计算并设置车辆的特征嵌入。
        
        处理场景中存在的车辆历史轨迹信息，通过 LSTM 提取时序特征。
        如果当前场景无车辆，会自动处理 NaN 填充。

        Args:
            veh (torch.FloatTensor): 车辆的历史轨迹序列。
                Shape: (batch_size, num_vehs, hist_step + 1, 2)
        
        Side Effects:
            设置 self.veh_embedding: 车辆特征向量 (batch_size, num_vehs, model_dim)
        """
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
        """
        计算并设置静态地图的特征嵌入。
        
        利用 CNN 提取栅格化地图的局部特征，并结合绝对位置编码。
        为了降低计算复杂度，使用 Latent Query (潜在令牌) 通过 Cross-Attention 
        从高维地图特征中提取关键的环境上下文信息 (Latent Embedding)。

        Args:
            map (torch.FloatTensor): 栅格化的环境地图，0代表可通行区域，1代表障碍物。
                Shape: (Map_W, Map_H), 第一维为 x（指向右方），第二维为 y（指向上方）
            xmin (float): 地图在世界坐标系下的 X 轴最小值。
            xmax (float): 地图在 world 坐标系下的 X 轴最大值。
            ymin (float): 地图在 world 坐标系下的 Y 轴最小值。
            ymax (float): 地图在 world 坐标系下的 Y 轴最大值。
        
        Side Effects:
            设置 self.map_embedding: 密集的网格地图特征 (Map_W, Map_H, model_dim)
            设置 self.ltn_embedding: 压缩后的地图潜在特征 (latent_token_num, model_dim)
            缓存地图边界信息 (self.xmin, self.xmax, etc.)
        """
        map_embedding = self.map_embedder(map.unsqueeze(-1)) # (W', H', model_dim)
        xx = torch.linspace(xmin, xmax, map_embedding.size(0), device=map_embedding.device)
        yy = torch.linspace(ymin, ymax, map_embedding.size(1), device=map_embedding.device)
        gridx, gridy = torch.meshgrid(xx, yy, indexing='ij')
        gridxy = torch.stack([gridx, gridy], dim=-1) # (W', H', 2)
        map_embedding = map_embedding + self.positional_encoding(gridxy) # (W', H', model_dim)
        latent_tokens = self.latent_tokens
        if self.args.use_spatial_anchor:
            anchor_phys_x = xmin + self.anchor_norm[..., 0] * (xmax - xmin)
            anchor_phys_y = ymin + self.anchor_norm[..., 1] * (ymax - ymin)
            anchor_phys = torch.stack([anchor_phys_x, anchor_phys_y], dim=-1) # (S, S, 2)
            anchor_pe = self.positional_encoding(anchor_phys) # (S, S, model_dim)
            latent_tokens = latent_tokens + anchor_pe.flatten(0, 1) # (S, S, D) -> (K, D)
        ltn_embedding = self.latent_attntn(latent_tokens, map_embedding.flatten(0, 1)) # (#latent_token, model_dim)
        self.map_embedding = map_embedding
        self.ltn_embedding = ltn_embedding
        self.map = map
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin

    def set_sur_info(self):
        """
        提取每个行人当前所在位置的局部环境特征 (Surrounding Info)。
        
        根据行人的世界坐标 (self.pos) 映射到栅格地图的索引，
        从 dense map embedding 中取出对应位置的特征向量。
        
        Side Effects:
            设置 self.sur_info: 行人脚下的环境特征 (batch_size, num_peds, model_dim)
        """
        pos = self.pos
        xmax, xmin = self.xmax, self.xmin
        ymax, ymin = self.ymax, self.ymin
        map_embedding = self.map_embedding
        idx = pos[..., 0].sub(xmin).div(xmax-xmin).mul(map_embedding.size(0)).round().long().clamp(0, map_embedding.size(0) - 1)  # (batch_size, #pedestrian)
        jdx = pos[..., 1].sub(ymin).div(ymax-ymin).mul(map_embedding.size(1)).round().long().clamp(0, map_embedding.size(1) - 1)  # (batch_size, #pedestrian)
        sur_info = map_embedding[idx, jdx] # (batch_size, #pedestrian, model_dim)
        sur_info = F.layer_norm(sur_info, sur_info.shape[-1:])
        self.sur_info = sur_info
        # """设置行人周边环境信息 sur_info (手动双线性插值 + 越界置零)"""
        # W, H = map_embedding.size(0), map_embedding.size(1)
        # EPS = 1e-6
        # # 1. 计算原始浮点坐标 (不截断，用于判断是否越界)
        # raw_grid_x = (pos[..., 0] - xmin).div(xmax - xmin).mul(W)
        # raw_grid_y = (pos[..., 1] - ymin).div(ymax - ymin).mul(H)
        # # 2. 生成有效性掩码 (Valid Mask)
        # # 只有在 [0, W-1] 和 [0, H-1] 范围内的才是有效点
        # # 注意：这里认为 W-0.5 依然在 W-1 的像素覆盖范围内，但 > W-1 即视为越界
        # # (根据具体定义，也可以用 W 或 W-0.5 作为边界，这里使用像素中心对齐的一般逻辑)
        # is_valid = (raw_grid_x >= 0) & (raw_grid_x <= W - 1) & \
        #            (raw_grid_y >= 0) & (raw_grid_y <= H - 1) # (batch, ped)
        # # 3. 截断坐标用于安全索引 (Safe Indexing)
        # # 即使是无效点，为了下面代码不报错，也得给它一个合法的索引(比如边缘)
        # grid_x = raw_grid_x.clamp(0, W - 1 - EPS)
        # grid_y = raw_grid_y.clamp(0, H - 1 - EPS)
        # x0 = grid_x.long()
        # y0 = grid_y.long()
        # x1 = (x0 + 1).clamp(max=W - 1)
        # y1 = (y0 + 1).clamp(max=H - 1)
        # # 4. 计算插值权重
        # wa = (grid_x - x0.float()).unsqueeze(-1) # (batch, ped, 1)
        # wb = (grid_y - y0.float()).unsqueeze(-1)
        # # 5. Gather 特征
        # Q00 = map_embedding[x0, y0] 
        # Q10 = map_embedding[x1, y0]
        # Q01 = map_embedding[x0, y1]
        # Q11 = map_embedding[x1, y1]
        # # 6. 双线性插值
        # sur_info = (
        #     Q00 * (1 - wa) * (1 - wb) +
        #     Q10 * wa * (1 - wb) +
        #     Q01 * (1 - wa) * wb +
        #     Q11 * wa * wb
        # )
        # # 7. LayerNorm (通常建议在 Mask 之前做，或者 Mask 后不再做 LN)
        # sur_info = F.layer_norm(sur_info, sur_info.shape[-1:])
        # # 8. 应用掩码：将越界区域强制置为 0
        # # is_valid 需要扩展维度以匹配 sur_info: (batch, ped) -> (batch, ped, 1)
        # sur_info = sur_info * is_valid.unsqueeze(-1).float()
        # self.sur_info = sur_info

    def forward(
        self, 
        denoise_t: torch.LongTensor,
        noisy_acc: torch.FloatTensor,
        ped_length: torch.LongTensor,
        veh_length: torch.LongTensor,
        timer: NamedTimer = None,
    ):
        """
        模型前向传播：根据上下文信息对带噪轨迹进行去噪预测。
        
        该方法必须在调用了 set_*_embedding 系列方法之后执行。
        它通过一系列 Transformer Decoder 层融合以下信息：
        1. 扩散时间步 embedding (t)
        2. 当前带噪的加速度 embedding (x_t)
        3. 社交交互 (Ped-Ped Attention)
        4. 人车交互 (Ped-Veh Attention)
        5. 环境交互 (Ped-Map Attention)
        
        Args:
            denoise_t (torch.LongTensor): 当前扩散过程的时间步 t。
                Shape: (batch_size,)
            noisy_acc (torch.FloatTensor): 加了噪声的未来加速度序列（扩散模型的输入 x_t）。
                Shape: (batch_size, num_peds, pred_step, 2)
            ped_length (torch.LongTensor): 一个 batch 中每个样本实际有效的行人数（用于 Mask）。
                Shape: (batch_size,)
            veh_length (torch.LongTensor): 一个 batch 中每个样本实际有效的车辆数（用于 Mask）。
                Shape: (batch_size,)
            timer (NamedTimer, optional): 用于性能分析的计时器对象。默认为 None。

        Returns:
            torch.FloatTensor: 模型预测的输出。
                如果 args.predict_noise 为 True，则输出预测的噪声 epsilon；
                否则输出预测的原始信号 x_0 (加速度)。
                Shape: (batch_size, num_peds, pred_step, 2)
        """

        # Embedding Pedestrian
        ped_embedding = self.ped_embedding
        denoise_t_embedding = self.denoise_t_embedder(denoise_t) # (batch_size, model_dim)
        denoise_t_embedding = denoise_t_embedding.unsqueeze(1) # (batch_size, 1, model_dim)
        noisy_acc_embedding = self.noisy_acc_embedder(noisy_acc) # (batch_size, #pedestrian, model_dim)
        ped_embedding = self.ped_encoder(ped_embedding + denoise_t_embedding + noisy_acc_embedding) # (batch_size, #pedestrian, model_dim)
        # ped_embedding = ped_embedding + denoise_t_embedding + noisy_acc_embedding # (batch_size, #pedestrian, model_dim)
        if timer: 
            torch.cuda.synchronize(device=self.args.device)
            timer.add('Embedding Pedestrian')

        # Embedding Vehicle
        veh_embedding = self.veh_embedding # (batch_size, #vehicle, model_dim)

        # Embedding Map
        ltn_embedding = self.ltn_embedding.unsqueeze(0).expand(ped_embedding.size(0), *self.ltn_embedding.shape) # (batch_size, #latent_token, model_dim)

        # Build Mask
        batch_size = ped_embedding.size(0)
        max_ped_num = ped_embedding.size(1)
        max_veh_num = veh_embedding.size(1)
        ped_mask = torch.arange(max_ped_num, device=ped_length.device).unsqueeze(0).expand(batch_size, max_ped_num) # (batch_size, max_ped_num)
        ped_mask = ped_mask >= ped_length.unsqueeze(1) # (batch_size, max_ped_num)
        veh_mask = torch.arange(max_veh_num, device=veh_length.device).unsqueeze(0).expand(batch_size, max_veh_num) # (batch_size, max_veh_num)
        veh_mask = veh_mask >= veh_length.unsqueeze(1) # (batch_size, max_veh_num)
        if timer: 
            torch.cuda.synchronize(device=self.args.device)
            timer.add('Build Mask')

        # Social Attention
        ped_info = self.ped_attention(
            ped_embedding, ped_embedding,
            memory_key_padding_mask=ped_mask,
            tgt_key_padding_mask=ped_mask,
        ) # (batch_size, #pedestrian, model_dim)
        ped_info = F.layer_norm(ped_info, ped_info.shape[-1:])
        if timer: 
            torch.cuda.synchronize(device=self.args.device)
            timer.add('Social Attention')
        # ped_info = 0

        # Vehicle Attention
        veh_info = self.veh_attention(
            ped_embedding, veh_embedding,
            memory_key_padding_mask=veh_mask,
            tgt_key_padding_mask=ped_mask,
        ) # (batch_size, #pedestrian, model_dim)
        veh_info = F.layer_norm(veh_info, veh_info.shape[-1:])
        if timer: 
            torch.cuda.synchronize(device=self.args.device)
            timer.add('Vehicle Attention')
        # veh_info = 0

        # Map Attention
        pe = self.pe
        map_info = self.map_attention(
            ped_embedding + pe, ltn_embedding,
            tgt_key_padding_mask=ped_mask,
        ) # (batch_size, #pedestrian, model_dim)
        map_info = F.layer_norm(map_info, map_info.shape[-1:])
        if timer: 
            torch.cuda.synchronize(device=self.args.device)
            timer.add('Map Attention')

        # Surrounding Info
        sur_info = self.sur_info

        # Fusion
        ped_embedding = self.fusion_fc(
            ped_embedding + ped_info + veh_info + map_info + sur_info + denoise_t_embedding
        ) # (batch_size, #pedestrian, model_dim)
        if timer: 
            torch.cuda.synchronize(device=self.args.device)
            timer.add('Fusion')

        # Output
        output = self.output_fc(ped_embedding) # (batch_size, #pedestrian, pred_step*2)
        output = output.view(*output.shape[:-1], self.args.pred_step, 2) # (batch_size, #pedestrian, pred_step, 2)
        if timer: 
            torch.cuda.synchronize(device=self.args.device)
            timer.add('Output')
        return output



