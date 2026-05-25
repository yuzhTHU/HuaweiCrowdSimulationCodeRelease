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
    Pedestrian trajectory prediction model based on Transformer and diffusion.

    The model combines pedestrian history, social interaction with nearby
    pedestrians, interaction with surrounding vehicles, and static map context
    to predict motion intent, either acceleration or noise, during reverse
    denoising in DDPM/DDIM.
    """

    def __init__(self, args):
        """
        Initialize model layers and embedding modules.

        Args:
            args (Namespace): Configuration object containing:
                - model_dim (int): Internal model feature dimension.
                - map_feature_dim (int): Intermediate map feature dimension.
                - lstm_layer_num (int): Number of LSTM layers for temporal data.
                - head_num (int): Number of attention heads.
                - attention_layer_num (int): Number of Transformer decoder layers.
                - latent_token_num (int): Number of latent tokens used to compress map features.
                - dropout (float): Dropout ratio.
                - pred_step (int): Prediction horizon.
                - use_spatial_anchor (bool): Whether to enhance map positional encoding with spatial anchors.
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
            NanEmbedding(2, args.model_dim, disable=not args.use_nan_embedding),
            MeanPoolingLSTM(args.model_dim, args.model_dim, args.lstm_layer_num),
            nn.LayerNorm(args.model_dim),
        )
        self.des_embedder = nn.Sequential(
            NanEmbedding(2, args.model_dim, disable=not args.use_nan_embedding),
            nn.LayerNorm(args.model_dim),
        )
        self.spd_embedder = nn.Sequential(
            NanEmbedding(1, args.model_dim, disable=not args.use_nan_embedding),
            nn.LayerNorm(args.model_dim),
        )
        self.ped_encoder = nn.Sequential(
            nn.LayerNorm(args.model_dim),
            nn.Linear(args.model_dim, 4*args.model_dim),
            nn.ReLU(),
            nn.Linear(4*args.model_dim, args.model_dim),
        )
        self.veh_embedder = nn.Sequential(
            NanEmbedding(2, args.model_dim, disable=not args.use_nan_embedding),
            MeanPoolingLSTM(args.model_dim, args.model_dim, args.lstm_layer_num),
            nn.LayerNorm(args.model_dim),
        )
        self.map_embedder = nn.Sequential(
            NanEmbedding(1, args.map_feature_dim//4, disable=not args.use_nan_embedding),
            Permuted(2, 0, 1),  # (H, W, C) -> (C, H, W)
            nn.Conv2d(args.map_feature_dim//4, args.map_feature_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.map_feature_dim//2, args.map_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.map_feature_dim, args.model_dim, kernel_size=1),  # 1x1 convolution, equivalent to a per-location Linear(C->Df).
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
            # Ensure the token count is a square number, e.g. 16 or 64.
            grid_size = int(math.sqrt(args.latent_token_num))
            if grid_size ** 2 != args.latent_token_num:
                raise ValueError(f"latent_token_num ({args.latent_token_num}) must be a square number when use_spatial_anchor is True.")
            self.grid_size = grid_size
            # Build a normalized 0~1 coordinate grid for later mapping to physical space.
            # Register it as a buffer so it is saved in the state dict but not updated as a parameter.
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
        Compute and store the joint pedestrian embedding.

        This method maps pedestrian position, velocity, history, destination,
        and desired speed into a high-dimensional space and sums them to form
        the initial pedestrian feature vector. It also computes positional encoding.

        Args:
            pos (torch.FloatTensor): Current pedestrian coordinates `(x, y)`.
                Shape: (batch_size, num_peds, 2)
            vel (torch.FloatTensor): Current pedestrian velocity `(vx, vy)`.
                Shape: (batch_size, num_peds, 2)
            hst (torch.FloatTensor): Pedestrian history trajectory sequence.
                Shape: (batch_size, num_peds, hist_step, 2)
            des (torch.FloatTensor): Pedestrian destination coordinates.
                Shape: (batch_size, num_peds, 2)
            spd (torch.FloatTensor): Pedestrian desired speed scalar.
                Shape: (batch_size, num_peds, 1)
        
        Side Effects:
            Sets `self.ped_embedding`: fused pedestrian features.
            Sets `self.pos`: cached current positions for later map indexing.
            Sets `self.pe`: positional encoding features.
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
        Compute and store vehicle feature embeddings.

        Processes vehicle trajectory history with an LSTM. If the current scene
        contains no vehicles, NaN padding is inserted automatically.

        Args:
            veh (torch.FloatTensor): Vehicle history trajectory sequence.
                Shape: (batch_size, num_vehs, hist_step + 1, 2)
        
        Side Effects:
            Sets `self.veh_embedding`: vehicle feature vectors.
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
        Compute and store static map embeddings.

        A CNN extracts local features from the rasterized map and combines them
        with absolute positional encoding. To reduce computation, latent queries
        compress the dense map features through cross-attention into a compact
        environmental context representation.

        Args:
            map (torch.FloatTensor): Rasterized environment map, where `0`
                denotes walkable area and `1` denotes obstacles.
                Shape: `(Map_W, Map_H)`, with the first dimension as `x`
                pointing right and the second as `y` pointing up.
            xmin (float): Minimum x value of the map in world coordinates.
            xmax (float): Maximum x value of the map in world coordinates.
            ymin (float): Minimum y value of the map in world coordinates.
            ymax (float): Maximum y value of the map in world coordinates.
        
        Side Effects:
            Sets `self.map_embedding`: dense grid-map features.
            Sets `self.ltn_embedding`: compressed latent map features.
            Caches map boundary metadata such as `self.xmin` and `self.xmax`.
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
        if self.args.use_latent_query:
            ltn_embedding = self.latent_attntn(latent_tokens, map_embedding.flatten(0, 1)) # (#latent_token, model_dim)
        else:
            ltn_embedding = map_embedding.flatten(0, 1)
        self.map_embedding = map_embedding
        self.ltn_embedding = ltn_embedding
        self.map = map
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin

    def set_sur_info(self):
        """
        Extract local environment features at each pedestrian's current position.

        Maps pedestrian world coordinates in `self.pos` to raster-map indices
        and retrieves the corresponding vectors from the dense map embedding.
        
        Side Effects:
            Sets `self.sur_info`: environmental features under each pedestrian.
        """
        pos = self.pos
        xmax, xmin = self.xmax, self.xmin
        ymax, ymin = self.ymax, self.ymin
        map_embedding = self.map_embedding
        idx = pos[..., 0].detach().sub(xmin).div(xmax-xmin).mul(map_embedding.size(0)).round().long().clamp(0, map_embedding.size(0) - 1)  # (batch_size, #pedestrian)
        jdx = pos[..., 1].detach().sub(ymin).div(ymax-ymin).mul(map_embedding.size(1)).round().long().clamp(0, map_embedding.size(1) - 1)  # (batch_size, #pedestrian)
        sur_info = map_embedding[idx, jdx] # (batch_size, #pedestrian, model_dim)
        sur_info = F.layer_norm(sur_info, sur_info.shape[-1:])
        self.sur_info = sur_info

    def forward(
        self, 
        denoise_t: torch.LongTensor,
        noisy_acc: torch.FloatTensor,
        ped_length: torch.LongTensor,
        veh_length: torch.LongTensor,
        timer: NamedTimer = None,
    ):
        """
        Forward pass: predict denoised trajectories from noisy inputs.

        This method must be called after the `set_*_embedding` methods. It
        fuses the following information through Transformer decoder layers:
        1. Diffusion timestep embedding (`t`)
        2. Current noisy acceleration embedding (`x_t`)
        3. Social interaction (`Ped-Ped Attention`)
        4. Pedestrian-vehicle interaction (`Ped-Veh Attention`)
        5. Environment interaction (`Ped-Map Attention`)
        
        Args:
            denoise_t (torch.LongTensor): Current diffusion timestep `t`.
                Shape: (batch_size,)
            noisy_acc (torch.FloatTensor): Noisy future acceleration sequence, the diffusion input `x_t`.
                Shape: (batch_size, num_peds, pred_step, 2)
            ped_length (torch.LongTensor): Number of valid pedestrians per sample in the batch, used for masking.
                Shape: (batch_size,)
            veh_length (torch.LongTensor): Number of valid vehicles per sample in the batch, used for masking.
                Shape: (batch_size,)
            timer (NamedTimer, optional): Timer object for performance profiling.

        Returns:
            torch.FloatTensor: Model prediction.
                If `args.predict_noise` is `True`, this is the predicted noise
                `epsilon`; otherwise it is the predicted original signal `x_0`
                in acceleration space.
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
        if (batch_wo_ped := ped_mask.all(dim=1)).any():
            raise ValueError("Some batch samples have zero pedestrians, which is not allowed.")
            ped_mask[batch_wo_ped, 0] = False
        veh_mask = torch.arange(max_veh_num, device=veh_length.device).unsqueeze(0).expand(batch_size, max_veh_num) # (batch_size, max_veh_num)
        veh_mask = veh_mask >= veh_length.unsqueeze(1) # (batch_size, max_veh_num)
        if (batch_wo_veh := veh_mask.all(dim=1)).any():
            veh_mask[batch_wo_veh, 0] = False

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



