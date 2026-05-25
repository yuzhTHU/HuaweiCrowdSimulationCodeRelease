import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import Residual
from .relative_model import RelativeModel
from ..utils.timer import NamedTimer


class NewModel(RelativeModel):
    """
    Revised model variant.

    Inherits from `RelativeModel`. The main changes are:
    - the pedestrian feature encoder (`ped_encoder`) uses a residual connection
    - local environment features (`sur_info`) are injected directly into the
      initial embedding instead of being fused later
    """

    def __init__(self, args):
        """
        Initialize model layers and embedding modules.

        Args:
            args (Namespace): Configuration object containing the standard model hyperparameters.
        """
        super().__init__(args)
        self.ped_encoder = Residual(
            nn.LayerNorm(args.model_dim),
            nn.Linear(args.model_dim, 4*args.model_dim),
            nn.ReLU(),
            nn.Linear(4*args.model_dim, args.model_dim),
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

        Position, velocity, history, destination, and desired speed are mapped
        into a shared feature space and summed into the initial pedestrian
        representation.

        Absolute position is not used directly. Instead, `pos` is encoded with
        `FourierPositionalEncoding`.

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
        hst_embedding = self.hst_embedder(hst-pos.unsqueeze(-2)) # (batch_size, #pedestrian, model_dim)
        des_embedding = self.des_embedder(des-pos) # (batch_size, #pedestrian, model_dim)
        spd_embedding = self.spd_embedder(spd) # (batch_size, #pedestrian, model_dim)
        fourier_pe = self.positional_encoding(pos) # (batch_size, #pedestrian, model_dim)

        xmax, xmin = self.xmax, self.xmin
        ymax, ymin = self.ymax, self.ymin
        map_embedding = self.map_embedding
        idx = pos[..., 0].sub(xmin).div(xmax-xmin).mul(map_embedding.size(0)).round().long().clamp(0, map_embedding.size(0) - 1)  # (batch_size, #pedestrian)
        jdx = pos[..., 1].sub(ymin).div(ymax-ymin).mul(map_embedding.size(1)).round().long().clamp(0, map_embedding.size(1) - 1)  # (batch_size, #pedestrian)
        sur_info = map_embedding[idx, jdx] # (batch_size, #pedestrian, model_dim)
        sur_info = F.layer_norm(sur_info, sur_info.shape[-1:])

        ped_embedding = vel_embedding + hst_embedding + des_embedding + spd_embedding + sur_info
        self.ped_embedding = ped_embedding
        self.fourier_pe = fourier_pe
        self.pos = pos

    def set_sur_info(self):
        pass

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
        fuses diffusion timestep, noisy acceleration, social interaction,
        pedestrian-vehicle interaction, and environment interaction.
        
        Args:
            denoise_t (torch.LongTensor): Current diffusion timestep `t`.
                Shape: (batch_size,)
            noisy_acc (torch.FloatTensor): Noisy future acceleration sequence, the diffusion input `x_t`.
                Shape: (batch_size, num_peds, pred_step, 2)
            ped_length (torch.LongTensor): Number of valid pedestrians per sample, used for masking.
                Shape: (batch_size,)
            veh_length (torch.LongTensor): Number of valid vehicles per sample, used for masking.
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
        ped_embedding = ped_embedding + self.fourier_pe
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

        # Fusion
        ped_embedding = self.fusion_fc(
            ped_embedding + ped_info + veh_info + map_info + denoise_t_embedding
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
