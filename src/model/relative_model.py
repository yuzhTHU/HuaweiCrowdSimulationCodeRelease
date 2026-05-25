import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Model
from .permuted import Permuted
from .nan_embedding import NanEmbedding
from ..utils.timer import NamedTimer


class RelativeModel(Model):
    """
    相对坐标模型 (RelativeModel)。
    
    继承自 Model 类。主要区别在于：
    1. 不直接使用绝对位置坐标进行 Embedding，而是更加依赖相对位置和速度。
    2. 使用傅里叶位置编码 (Fourier Positional Encoding) 显式编码位置信息。
    3. 车辆特征提取时也采用相对坐标处理。
    这通常能提高模型在不同场景坐标系下的泛化能力。
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
        super().__init__(args)
        self.map_embedder = nn.Sequential(
            NanEmbedding(1, args.map_feature_dim//4, disable=not args.use_nan_embedding),
            nn.ReLU(),
            Permuted(2, 0, 1),  # (H, W, C) -> (C, H, W)
            nn.Conv2d(args.map_feature_dim//4, args.map_feature_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.map_feature_dim//2, args.map_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.map_feature_dim, args.model_dim, kernel_size=1),  # 1x1 卷积，相当于每个 (h,w) 位置的 Linear(C->Df)
            Permuted(1, 2, 0),  # (C, H, W) -> (H, W, C)
            nn.LayerNorm(args.model_dim),
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

        相比于 Model，Relative Model 不使用 pos 中的绝对位置，而是使用 FourierPositionalEncoding 将 pos 编码到 pe 中

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
        # pos_embedding = self.pos_embedder(pos) # (batch_size, #pedestrian, model_dim)
        vel_embedding = self.vel_embedder(vel) # (batch_size, #pedestrian, model_dim)
        if self.args.use_relative_features:
            hst_embedding = self.hst_embedder(hst-pos.unsqueeze(-2)) # (batch_size, #pedestrian, model_dim)
            des_embedding = self.des_embedder(des-pos) # (batch_size, #pedestrian, model_dim)
        else:
            hst_embedding = self.hst_embedder(hst) # (batch_size, #pedestrian, model_dim)
            des_embedding = self.des_embedder(des) # (batch_size, #pedestrian, model_dim)
        spd_embedding = self.spd_embedder(spd) # (batch_size, #pedestrian, model_dim)
        if self.args.use_frequency_encoding:
            fourier_pe = self.positional_encoding(pos) # (batch_size, #pedestrian, model_dim)
            ped_embedding = vel_embedding + hst_embedding + des_embedding + spd_embedding + fourier_pe
        else:
            pos_embedding = self.pos_embedder(pos) # (batch_size, #pedestrian, model_dim)
            ped_embedding = pos_embedding + vel_embedding + hst_embedding + des_embedding + spd_embedding
        self.ped_embedding = ped_embedding
        self.pos = pos

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
            shape[1] = 2
            veh = torch.full(shape, float('nan'), device=veh.device)
        # 不使用 veh 中的绝对位置，而是使用 FourierPositionalEncoding 将 veh_pos 编码到 pe 中
        if self.args.use_relative_features:
            rel_veh_embedding = self.veh_embedder(veh - veh[..., (-1,), :]) # (batch_size, #vehicle, model_dim)
        else:
            rel_veh_embedding = self.veh_embedder(veh) # (batch_size, #vehicle, model_dim)
        if self.args.use_frequency_encoding:
            fourier_pe = self.positional_encoding(veh[..., -1, :]) # (batch_size, #vehicle, model_dim)
            veh_embedding = rel_veh_embedding + fourier_pe
        else:
            veh_pos_embedding = self.veh_embedder(veh[..., (-1,), :])
            veh_embedding = rel_veh_embedding + veh_pos_embedding
        self.veh_embedding = veh_embedding

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
        map_info = self.map_attention(
            ped_embedding, ltn_embedding,
            tgt_key_padding_mask=ped_mask,
        ) # (batch_size, #pedestrian, model_dim)
        map_info = F.layer_norm(map_info, map_info.shape[-1:])
        if timer: 
            torch.cuda.synchronize(device=self.args.device)
            timer.add('Map Attention')

        # Surrounding Info
        sur_info = self.sur_info

        # Fusion
        # _logger.debug(
        #     f"ped_embedding.shape={ped_embedding.shape}, "
        #     f"ped_info.shape={ped_info.shape}, "
        #     f"veh_info.shape={veh_info.shape}, "
        #     f"map_info.shape={map_info.shape}, "
        #     f"sur_info.shape={sur_info.shape}, "
        #     f"denoise_t_embedding.shape={denoise_t_embedding.shape}"
        # )
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

