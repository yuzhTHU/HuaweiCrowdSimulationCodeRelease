import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Model
from .permuted import Permuted
from .nan_embedding import NanEmbedding
from ..utils.timer import NamedTimer


class RelativeModel(Model):
    """
    Relative-coordinate model.

    Inherits from `Model`. The main differences are:
    1. It relies more on relative position and velocity instead of embedding
       absolute position directly.
    2. It uses Fourier positional encoding to encode location explicitly.
    3. Vehicle features are also processed in relative coordinates.
    This often improves generalization across scenes with different coordinate systems.
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
        super().__init__(args)
        self.map_embedder = nn.Sequential(
            NanEmbedding(1, args.map_feature_dim//4, disable=not args.use_nan_embedding),
            nn.ReLU(),
            Permuted(2, 0, 1),  # (H, W, C) -> (C, H, W)
            nn.Conv2d(args.map_feature_dim//4, args.map_feature_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.map_feature_dim//2, args.map_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.map_feature_dim, args.model_dim, kernel_size=1),  # 1x1 convolution, equivalent to a per-location Linear(C->Df).
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
        Compute and store the joint pedestrian embedding.

        This method maps pedestrian position, velocity, history, destination,
        and desired speed into a high-dimensional space and sums them to form
        the initial pedestrian feature vector. It also computes positional encoding.

        Unlike `Model`, `RelativeModel` does not use the absolute coordinates in
        `pos` directly. Instead, it encodes `pos` with `FourierPositionalEncoding`.

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
            shape[1] = 2
            veh = torch.full(shape, float('nan'), device=veh.device)
        # Do not use absolute vehicle positions directly; encode the last vehicle position with Fourier positional encoding.
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

