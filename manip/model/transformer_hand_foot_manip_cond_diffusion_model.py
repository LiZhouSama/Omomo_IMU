import os 
import math 

from tqdm.auto import tqdm

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from inspect import isfunction

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch3d.transforms as transforms 

from manip.model.transformer_module import Decoder 

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
        
class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,  # IMU编码(512) + 物体特征(256+3)
        d_output_feats,        # 输出维度：N+1个关节 × (3维位置 + 6维旋转) = (N+1)*9
        d_model,        # Transformer隐层维度
        n_dec_layers,   # Decoder层数
        n_head,         # 注意力头数
        d_k,           # 注意力键维度
        d_v,           # 注意力值维度
        max_timesteps, # 最大时间步
    ):
        super().__init__()
        
        self.d_output_feats = d_output_feats  # (N+1)*9维输出
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        # Transformer解码器
        self.motion_transformer = Decoder(
            d_feats=d_input_feats,  # 输入特征维度: IMU编码(512) + 物体特征(256+3)
            d_model=self.d_model,   
            n_layers=self.n_dec_layers,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            max_timesteps=self.max_timesteps,
            use_full_attention=True
        )  

        # 输出层：预测关节位置和旋转
        self.linear_out = nn.Linear(self.d_model, self.d_output_feats)  # 输出: (N+1)*9维 ((N+1)个关节×(3维位置+6维旋转))

        # 时间嵌入层
        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, imu_data, time_step, condition, padding_mask=None):
        """
        前向传播函数
        
        参数:
            imu_data: IMU输入数据 [BS X T X 512] (已经过IMU编码器处理)
            time_step: 扩散时间步 [BS] (batch_size)
            condition: 物体条件信息 [BS X T X 259] (batch_size, timesteps, 3维位置+256维BPS特征)
            padding_mask: 填充掩码 [BS X 1 X (T+1)] (batch_size, 1, timesteps+1)
            
        返回:
            output: 预测的关节位置和旋转 [BS X T X (N+1)*9] (batch_size, timesteps, (N+1)个关节×(3维位置+6维旋转))
        """
        # 直接使用编码后的IMU特征
        imu_features = imu_data  # [BS X T X 512]
        
        # print("imu_features.shape: ", imu_features.shape)   
        # print("condition.shape: ", condition.shape)

        # 合并IMU特征和物体特征
        combined_features = torch.cat((imu_features, condition), dim=-1)  # [BS X T X (512+259)]
        # print("combined_features.shape: ", combined_features.shape)
        # 3. 时间步编码
        time_embed = self.time_mlp(time_step)  # [BS X d_model]
        time_embed = time_embed[:, None, :]  # [BS X 1 X d_model]

        # 4. 准备位置编码
        bs = imu_data.shape[0]
        num_steps = imu_data.shape[1] + 1

        if padding_mask is None:
            padding_mask = torch.ones(bs, 1, num_steps).to(imu_data.device).bool()

        pos_vec = torch.arange(num_steps)+1
        pos_vec = pos_vec[None, None, :].to(imu_data.device).repeat(bs, 1, 1)

        # 5. Transformer处理
        data_input = combined_features.transpose(1, 2).detach()  # [BS X (512+259) X T]
        feat_pred, _ = self.motion_transformer(
            data_input,              # 输入特征
            padding_mask,            # 填充掩码
            pos_vec,                 # 位置编码
            obj_embedding=time_embed # 时间嵌入
        )

        # 6. 生成最终输出
        output = self.linear_out(feat_pred[:, 1:])  # [BS X T X (N+1)*9]

        return output

class CondGaussianDiffusion(nn.Module):
    def __init__(
        self, 
        opt, 
        input_dim=24,   # N+1个IMU x 6维 输入
        out_dim=27,     # N+1个关节 x (3维位置 + 6维旋转)维输出
        d_model=512, 
        n_dec_layers=4, 
        n_head=4, 
        d_k=256, 
        d_v=256, 
        max_timesteps=121, 
        timesteps=1000, 
        objective='pred_x0', 
        loss_type='l1', 
        beta_schedule = 'cosine', 
        p2_loss_weight_gamma = 0., 
        p2_loss_weight_k = 1, 
        batch_size=None):
        super().__init__()
        
        # IMU编码器 (N+1个IMU x 6维)
        self.imu_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model//2),  # (N+1)个IMU x 6维 = (N+1)*6维输入
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        # BPS编码器保持不变
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024*3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
        )
        
        # 更新条件特征维度
        obj_feats_dim = 256 
        d_input_feats = d_model + 3 + obj_feats_dim  # IMU编码维度(512) + 物体位置(3) + 物体特征(256)
        
        self.denoise_fn = TransformerDiffusionModel(
            d_input_feats=d_input_feats,
            d_output_feats=out_dim,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            n_dec_layers=n_dec_layers,
            max_timesteps=max_timesteps
        )

        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_mean_variance(self, x, t, imu_data, x_cond, padding_mask, clip_denoised):
        """计算预测的均值和方差
        参数:
            x: 当前噪声数据 (目标关节数据的噪声版本)
            t: 当前时间步
            imu_data: IMU输入数据 BS X T X (N+1)*6 - N个人体IMU + 1个物体IMU
            x_cond: 条件信息(物体特征) BS X T X 259
            padding_mask: 填充掩码
            clip_denoised: 是否裁剪去噪结果
        """
        # 编码IMU数据
        imu_encoded = self.imu_encoder(imu_data)  # BS X T X 512
        
        # 使用编码后的IMU数据进行预测
        model_output = self.denoise_fn(imu_encoded, t, x_cond, padding_mask)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, imu_data, x_cond, padding_mask=None, clip_denoised=True):
        """单步去噪采样
        参数:
            x: 当前噪声数据 (目标关节数据的噪声版本)
            t: 当前时间步
            imu_data: IMU输入数据 BS X T X (N+1)*6 - N个人体IMU + 1个物体IMU
            x_cond: 条件信息(物体特征) BS X T X 259
            padding_mask: 填充掩码
            clip_denoised: 是否裁剪去噪结果
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, imu_data=imu_data, 
                                                               x_cond=x_cond, padding_mask=padding_mask, 
                                                               clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, imu_data, x_cond, padding_mask=None):
        """完整的采样循环过程
        参数:
            shape: 采样数据的形状 (BS X T X (3*9)) - 3个关节的位置和旋转
            imu_data: IMU输入数据 (BS X T X (N+1)*6) - N个人体关节和1个物体的IMU数据
            x_cond: 条件信息(物体特征)
            padding_mask: 填充掩码
        """
        device = self.betas.device
        b = shape[0]
        # 初始化为随机噪声
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(
                x, 
                torch.full((b,), i, device=device, dtype=torch.long),
                imu_data=imu_data,
                x_cond=x_cond,
                padding_mask=padding_mask
            )    
        
        return x  # 返回预测的关节位置和旋转 (BS X T X (3*9))

    @torch.no_grad()
    def sample(self, imu_data, ori_x_cond, cond_mask=None, padding_mask=None):
        """生成最终的采样结果
        参数:
            imu_data: IMU输入数据 BS X T X (N+1)*6 - N个人体IMU + 1个物体IMU
            ori_x_cond: 原始条件数据(物体特征) BS X T X (3+1024*3)
            cond_mask: 条件掩码
            padding_mask: 填充掩码
        返回:
            sample_res: 预测的关节位置和旋转 BS X T X ((N+1)*9)
        """
        device = imu_data.device
        batch_size = imu_data.shape[0]
        shape = (batch_size, imu_data.shape[1], self.out_dim)

        # 编码物体特征
        x_cond = torch.cat((ori_x_cond[:, :, :3], self.bps_encoder(ori_x_cond[:, :, 3:])), dim=-1)

        if cond_mask is not None:
            x_pose_cond = torch.randn_like(x_cond) * cond_mask
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)

        # 从噪声开始采样
        img = torch.randn(shape, device=device)
        # print("sample_imu_data.shape: ", imu_data.shape)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((batch_size,), i, device=device, dtype=torch.long),
                              imu_data, x_cond, padding_mask)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, imu_data, target_data, x_cond, t, noise=None, padding_mask=None):
        """计算损失
        
        参数:
            imu_data: 编码的IMU输入数据 BS X T X 512  # 512维 (N个人体关节和1个物体的IMU)
            target_data: 目标关节数据 BS X T X (N*9)  # N*9维 (N个关节×3维位置+N个关节×6维旋转))
            x_cond: 物体条件信息 [BS X T X 259] (batch_size, timesteps, 3维位置+256维BPS特征)
            t: 时间步
            noise: 噪声
            padding_mask: 填充掩码
        """
        b, timesteps, _ = target_data.shape
        noise = default(noise, lambda: torch.randn_like(target_data))

        # 对目标数据添加噪声
        x = self.q_sample(x_start=target_data, t=t, noise=noise)

        # 使用IMU数据和带噪声的目标数据预测去噪结果
        model_out = self.denoise_fn(imu_data, t, x_cond, padding_mask) 

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = target_data
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction='none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction='none')

        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        
        return loss.mean()

    def forward(self, imu_data, target_data, ori_x_cond, cond_mask=None, padding_mask=None):
        """
        前向传播
        
        参数:
            imu_data: IMU输入数据 BS X T X (N+1)*6 - N个人体IMU + 1个物体IMU
            target_data: 目标关节数据 BS X T X (N*9)
            ori_x_cond: 原始条件数据 BS X T X (3+1024*3)
            cond_mask: 条件掩码
            padding_mask: 填充掩码
        """
        bs = target_data.shape[0]
        t = torch.randint(0, self.num_timesteps, (bs,), device=target_data.device).long()
        # 编码IMU数据
        # print("forward_imu_data.shape: ", imu_data.shape)
        imu_encoded = self.imu_encoder(imu_data)
        
        # 编码物体特征
        x_cond = torch.cat((ori_x_cond[:, :, :3], self.bps_encoder(ori_x_cond[:, :, 3:])), dim=-1)
        
        if cond_mask is not None:
            x_pose_cond = target_data * (1. - cond_mask) + cond_mask * torch.randn_like(target_data)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)

        curr_loss = self.p_losses(imu_encoded, target_data, x_cond, t, padding_mask=padding_mask)

        return curr_loss
        