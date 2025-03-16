import argparse
import os
import numpy as np
import yaml
import random
import json 
import time

# 设置CUDA调试模式
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import trimesh 

from tqdm import tqdm
from pathlib import Path

import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from ema_pytorch import EMA
from multiprocessing import cpu_count

from human_body_prior.body_model.body_model import BodyModel

from manip.data.hand_foot_dataset import HandFootManipDataset, quat_ik_torch, quat_fk_torch

from manip.model.transformer_hand_foot_manip_cond_diffusion_model import CondGaussianDiffusion 

from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object

from manip.model.transformer_fullbody_cond_diffusion_model import CondGaussianDiffusion as FullBodyCondGaussianDiffusion
from trainer_full_body_manip_diffusion import Trainer as FullBodyTrainer 

from evaluation_metrics import compute_metrics, compute_s1_metrics, compute_collision

from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from manip.data.config import IMU_JOINTS, IMU_JOINT_NAMES, NUM_IMU_JOINTS
# 循环数据加载器，使其能够无限迭代
def cycle(dl):
    """
    创建一个无限循环的数据加载器
    
    参数:
        dl: 数据加载器
        
    返回:
        一个无限循环的数据生成器
    """
    while True:
        for data in dl:
            yield data

class Trainer(object):
    """
    手部和脚部操作扩散模型的训练器类
    """
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=40000,
        results_folder='./results',
        use_wandb=True,
        pretrained_path=None  # 添加预训练模型路径参数
    ):
        """
        初始化训练器
        
        参数:
            opt: 配置选项
            diffusion_model: 扩散模型
            ema_decay: 指数移动平均衰减率
            train_batch_size: 训练批次大小
            train_lr: 学习率
            train_num_steps: 总训练步数
            gradient_accumulate_every: 梯度累积步数
            amp: 是否使用混合精度训练
            step_start_ema: 开始使用EMA的步数
            ema_update_every: EMA更新频率
            save_and_sample_every: 保存和采样频率
            results_folder: 结果保存文件夹
            use_wandb: 是否使用wandb记录
            pretrained_path: 预训练模型路径，如果为None则使用默认路径
        """
        super().__init__()

        self.pretrained_path = pretrained_path
        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, \
            name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)
        # 添加学习率调度器
        # 每 decay_steps 步将学习率衰减为原来的 decay_rate
        self.decay_steps = 100000  # 每40000步衰减一次
        self.decay_rate = 0.5    # 每次衰减为原来的0.5
        self.min_lr = 1e-6      # 最小学习率
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.decay_steps,
            gamma=self.decay_rate
        )

        self.step = 0
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.results_folder = results_folder
        self.vis_folder = results_folder.replace("weights", "vis_res")
        self.opt = opt 
        self.window = opt.window
        self.use_object_split = self.opt.use_object_split 
        self.data_root_folder = self.opt.data_root_folder 
        self.prep_dataloader(window_size=opt.window)
        self.bm_dict = self.ds.bm_dict 
        self.test_on_train = self.opt.test_sample_res_on_train 
        self.add_hand_processing = self.opt.add_hand_processing 
        self.for_quant_eval = self.opt.for_quant_eval 
        self.use_gt_hand_for_eval = self.opt.use_gt_hand_for_eval 
        
        # 如果提供了预训练模型路径，加载模型
        if self.pretrained_path is not None:
            self.load(self.pretrained_path)

    def prep_dataloader(self, window_size):
        """
        准备数据加载器
        
        参数:
            window_size: 时间窗口大小
        """
        # Define dataset
        train_dataset = HandFootManipDataset(train=True, data_root_folder=self.data_root_folder, \
            window=window_size, use_object_splits=self.use_object_split)
        val_dataset = HandFootManipDataset(train=False, data_root_folder=self.data_root_folder, \
            window=window_size, use_object_splits=self.use_object_split)

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
            shuffle=True, pin_memory=True, num_workers=16))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=16))

    def save(self, milestone):
        """
        保存模型检查点
        
        参数:
            milestone: 当前训练里程碑
        """
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, path):
        """
        加载预训练的模型和优化器状态
        
        参数:
            path: 预训练模型的路径
        """
        print(f"Loading pretrained model from {path}")
        data = torch.load(path)
        
        # 加载模型状态
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)

        # 加载优化器状态
        if 'scaler' in data:
            self.scaler.load_state_dict(data['scaler'])
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])
        if 'scheduler' in data:
            self.scheduler.load_state_dict(data['scheduler'])
        
        # 加载训练步数
        self.step = data['step']
        
        print(f"Successfully loaded model at step {self.step}")

    def prep_temporal_condition_mask(self, data, t_idx=0):
        """
        准备时间条件掩码
        
        参数:
            data: 输入数据
            t_idx: 时间索引
            
        返回:
            掩码张量，条件区域为0，缺失区域为1
        """
        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data).to(data.device) # BS X T X D 
        mask[:, t_idx, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 

    def compute_imu_data(self, imu_jpos, imu_rot_6d, obj_trans, obj_rot_mat, freq=1):
        """
        生成人体和物体的IMU数据
        
        参数:
            global_jpos: T X 24 X 3 全局关节位置
            global_rot_mat: T X 22 X 6 全局关节旋转矩阵
            obj_trans: T X 3 物体位置
            obj_rot_mat: T X 3 X 3 物体旋转矩阵
            
        返回:
            imu_data: T X (N+1) X 6，其中 N=6 是人体IMU数量
                    - 人体IMU: 左手、右手、头部、左脚、右脚、髋部，每个包含3维加速度+3维角速度
                    - 物体IMU: 1个，包含3维加速度+3维角速度
        """
        dt = 1.0 / freq
        BS, T, M3 = imu_jpos.shape
        N = M3 // 3  # 人体IMU数量
        device = imu_jpos.device
        # 确保物体数据也在同一设备上
        obj_trans = obj_trans.to(device)
        obj_rot_mat = obj_rot_mat.to(device)

        # 将输入转换为合适形状
        human_positions = imu_jpos.reshape(BS, T, N, 3)       # [BS, T, N, 3]
        human_rot_6d = imu_rot_6d.reshape(BS, T, N, 6)          # [BS, T, N, 6]
        # 将6D表示转换为旋转矩阵: 输出 [BS, T, N, 3, 3]
        human_rot_mat = transforms.rotation_6d_to_matrix(human_rot_6d)
        
        # 计算人体加速度和角速度
        # 对 human_positions 沿时间维度进行中心差分，先 pad 时间维度
        padded_pos = torch.cat([human_positions[:, :1], human_positions, human_positions[:, -1:]], dim=1)  # [BS, T+2, N, 3]
        velocities = (padded_pos[:, 2:] - padded_pos[:, :-2]) / (2.0 * dt)      # [BS, T, N, 3]
        accelerations = (padded_pos[:, 2:] + padded_pos[:, :-2] - 2 * padded_pos[:, 1:-1]) / (dt ** 2)  # [BS, T, N, 3]
        
        # 计算角速度：利用 human_rot_mat 的相邻帧差分
        angular_vel = torch.zeros(BS, T, N, 3, device=device)
        # 对于 t>=1: 相对旋转 = R[t] * (R[t-1])^T
        relative_rot = torch.matmul(human_rot_mat[:, 1:], human_rot_mat[:, :-1].transpose(-1, -2))  # [BS, T-1, N, 3, 3]
        # 转换为轴角表示，再除以 dt
        angular_vel[:, 1:] = transforms.matrix_to_axis_angle(relative_rot) / dt  # [BS, T-1, N, 3]
        
        # 计算物体IMU数据
        # 假设 obj_trans: [BS, T, 3]
        padded_obj_trans = torch.cat([obj_trans[:, :1], obj_trans, obj_trans[:, -1:]], dim=1)  # [BS, T+2, 3]
        obj_vel = (padded_obj_trans[:, 2:] - padded_obj_trans[:, :-2]) / (2.0 * dt)  # [BS, T, 3]
        obj_acc = (padded_obj_trans[:, 2:] + padded_obj_trans[:, :-2] - 2 * padded_obj_trans[:, 1:-1]) / (dt ** 2)  # [BS, T, 3]
        
        # 物体角速度：obj_rot_mat: [BS, T, 3, 3]
        obj_angular_vel = torch.zeros(BS, T, 3, device=device)
        obj_relative_rot = torch.matmul(obj_rot_mat[:, 1:], obj_rot_mat[:, :-1].transpose(-1, -2))  # [BS, T-1, 3, 3]
        obj_angular_vel[:, 1:] = transforms.matrix_to_axis_angle(obj_relative_rot) / dt  # [BS, T-1, 3]
        
        # 组合人体IMU数据：加速度和角速度拼接
        human_imu = torch.cat([accelerations, angular_vel], dim=-1)  # [BS, T, N, 6]
        # 组合物体IMU数据
        obj_imu = torch.cat([obj_acc.unsqueeze(2), obj_angular_vel.unsqueeze(2)], dim=-1)  # [BS, T, 1, 6]
        
        # 合并人体和物体IMU数据
        imu_data = torch.cat([human_imu, obj_imu], dim=2)  # [BS, T, N+1, 6]
        imu_data = imu_data.reshape(BS, T, -1)  # [BS, T, (N+1)*6]
        
        return imu_data
    
    def extract_joints_pos_rot_data(self, data_input, joint_indices, pos_only=False):
        """
        从输入数据中提取指定关节的位置和旋转数据
        
        参数:
            data_input: 输入数据 BS X T X D (24*3+24*3+22*6)
            joint_indices: 需要提取的关节索引列表
            
        返回:
            如果pos_only=True:
                目标关节位置数据 BS X T X (3*3)
            否则:
                目标关节位置和旋转数据 BS X T X (3*9)
        """
        bs, timesteps, _ = data_input.shape
        
        # 提取关节位置数据
        joint_positions = data_input[:, :, :24*3].reshape(bs, timesteps, 24, 3)  # BS X T X 24 X 3
        joint_positions = joint_positions[:, :, joint_indices]  # BS X T X N X 3
        
        if pos_only:
            return joint_positions.reshape(bs, timesteps, -1)  # BS X T X (N*3)
        
        # 提取关节旋转数据
        joint_rotations = data_input[:, :, 24*3:].reshape(bs, timesteps, 22, 6)  # BS X T X 22 X 6
        joint_rotations = joint_rotations[:, :, joint_indices]  # BS X T X N X 6
        
        # 合并位置和旋转数据
        joints_data = torch.cat([
            joint_positions.reshape(bs, timesteps, -1),  # BS X T X (N*3)
            joint_rotations.reshape(bs, timesteps, -1)   # BS X T X (N*6)
        ], dim=-1)  # BS X T X (N*9)

        return joints_data

    def train(self):
        """训练模型"""
        # 在训练开始前初始化计时器
        step_start_time = time.time()
        total_steps_for_timing = 0

        init_step = self.step 
        
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()
            
            nan_exists = False
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
                data = data_dict['motion'].cuda()
                obj_trans_data = data_dict['obj_trans'].cuda()
                obj_rot_mat_data = data_dict['obj_rot_mat'].cuda()
                obj_rot_6d_data = transforms.matrix_to_rotation_6d(obj_rot_mat_data)
                # imu_data = data_dict['imu_data'].cuda()  # BS X T X (N+1)*6 - 包含物体IMU
                # 提取目标关节数据
                human_imu_joints_data = self.extract_joints_pos_rot_data(data, IMU_JOINTS)
                human_imu_joints_data_pos = human_imu_joints_data[:, :, :NUM_IMU_JOINTS*3]
                human_imu_joints_data_rot = human_imu_joints_data[:, :, NUM_IMU_JOINTS*3:]
                imu_data = self.compute_imu_data(human_imu_joints_data_pos, human_imu_joints_data_rot, obj_trans_data, obj_rot_mat_data)
                target_data = torch.cat((human_imu_joints_data_pos, obj_trans_data, human_imu_joints_data_rot, obj_rot_6d_data), dim=-1)  # BS X T X (N+1)*9 
                
                # 准备物体特征条件
                obj_bps_data = data_dict['obj_bps'].cuda()
                obj_com_pos = data_dict['obj_com_pos'].cuda()
                ori_data_cond = torch.cat((obj_com_pos, obj_bps_data), dim=-1)
                
                # 生成填充掩码
                actual_seq_len = data_dict['seq_len'] + 1
                tmp_mask = torch.arange(self.window+1).expand(data.shape[0], self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(data.device)

                with autocast(enabled = self.amp):
                    loss_diffusion = self.model(imu_data, target_data, ori_data_cond, None, padding_mask)
                    loss = loss_diffusion

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data... ', loss)
                        nan_exists = True 
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    # check gradients
                    parameters = [p for p in self.model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(data.device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True 
                        continue

                    if self.use_wandb:
                        log_dict = {
                            "Train/Loss/Total Loss": loss.item(),
                            "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                            "Train/Learning Rate": self.optimizer.param_groups[0]['lr'],  # 记录当前学习率
                        }
                        wandb.log(log_dict)

                    if idx % 100 == 0 and i == 0:
                        print(f"Step: {idx}")
                        print(f"Loss: {loss.item():.4f}")
                        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")


            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr > self.min_lr:
                self.scheduler.step()

            self.ema.update()

            self.step += 1
            total_steps_for_timing += 1

            # 每100步计算一次平均时间
            if idx % 100 == 99:  # 使用99确保是在完成100步后计算
                step_duration = time.time() - step_start_time
                avg_time_per_step = step_duration / total_steps_for_timing
                print(f"最近 {total_steps_for_timing} 步平均耗时: {avg_time_per_step:.4f} 秒/步")
                print(f"总耗时: {step_duration:.4f} 秒")
                
                # 重置计时器和步数计数
                step_start_time = time.time()
                total_steps_for_timing = 0

            if self.step != 0 and self.step % 1000 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)
                    val_data = val_data_dict['motion'].cuda()   
                    val_obj_trans_data = val_data_dict['obj_trans'].cuda()
                    val_obj_rot_mat_data = val_data_dict['obj_rot_mat'].cuda()
                    val_obj_rot_6d_data = transforms.matrix_to_rotation_6d(val_obj_rot_mat_data)
                    val_human_imu_joints_data = self.extract_joints_pos_rot_data(val_data, IMU_JOINTS)
                    val_human_imu_joints_data_pos = val_human_imu_joints_data[:, :, :NUM_IMU_JOINTS*3]
                    val_human_imu_joints_data_rot = val_human_imu_joints_data[:, :, NUM_IMU_JOINTS*3:]
                    val_imu_data = self.compute_imu_data(val_human_imu_joints_data_pos, val_human_imu_joints_data_rot, val_obj_trans_data, val_obj_rot_mat_data)
                    val_target_data = torch.cat((val_human_imu_joints_data_pos, val_obj_trans_data, val_human_imu_joints_data_rot, val_obj_rot_6d_data), dim=-1)

                    # 准备验证数据的物体特征条件
                    val_obj_bps_data = val_data_dict['obj_bps'].cuda()
                    val_obj_com_pos = val_data_dict['obj_com_pos'].cuda()
                    val_ori_data_cond = torch.cat((val_obj_com_pos, val_obj_bps_data), dim=-1)  # BS X T X (3+1024*3)

                    # 生成验证数据的填充掩码
                    val_actual_seq_len = val_data_dict['seq_len'] + 1
                    val_tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], self.window+1) < \
                                   val_actual_seq_len[:, None].repeat(1, self.window+1)
                    val_padding_mask = val_tmp_mask[:, None, :].to(val_data.device)

                    # print("val_imu_data.shape: ", val_imu_data.shape)
                    # print("val_ori_data_cond.shape: ", val_ori_data_cond.shape)

                    with torch.no_grad():
                        # 修改这里的模型调用，确保参数顺序正确
                        val_loss_diffusion = self.model(
                            val_imu_data,           # IMU输入数据 BS X T X (N*6)
                            val_target_data,        # 目标关节数据 BS X T X ((N+1)*9)
                            val_ori_data_cond,      # 物体特征条件 BS X T X (3+1024*3)
                            None,                   # 条件掩码 (可选)
                            val_padding_mask        # 填充掩码
                        )

                    val_loss = val_loss_diffusion 
                    if self.use_wandb:
                        val_log_dict = {
                            "Validation/Loss/Total Loss": val_loss.item(),
                            "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                        }
                        wandb.log(val_log_dict)

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

                bs_for_vis = 1
                all_res_list = self.ema.ema_model.sample(val_imu_data, val_ori_data_cond, None, val_padding_mask)
                all_res_list = all_res_list[:bs_for_vis]
                print("all_res_list.shape: ", all_res_list.shape)

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()

    def cond_sample_res(self):
        """
        从训练好的模型中采样结果
        """
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(weight_path)
        self.ema.ema_model.eval()

        num_sample = 50
        
        with torch.no_grad():
            for s_idx in range(num_sample):
                if self.test_on_train:
                    val_data_dict = next(self.dl)
                else:
                    val_data_dict = next(self.val_dl)
                
                val_data = val_data_dict['motion'].cuda()   
                val_human_imu_joints_data = self.extract_joints_pos_rot_data(val_data, IMU_JOINTS)
                val_human_imu_joints_data_pos = val_human_imu_joints_data[:, :, :NUM_IMU_JOINTS*3]
                val_human_imu_joints_data_rot = val_human_imu_joints_data[:, :, NUM_IMU_JOINTS*3:]
                val_imu_data = self.compute_imu_data(val_human_imu_joints_data_pos, val_human_imu_joints_data_rot, val_data_dict['obj_trans'], val_data_dict['obj_rot_mat'])
     
                obj_bps_data = val_data_dict['obj_bps'].cuda()
                obj_com_pos = val_data_dict['obj_com_pos'].cuda() 

                ori_data_cond = torch.cat((obj_com_pos, obj_bps_data), dim=-1) # BS X T X (3+1024*3)

                cond_mask = None 

                # Generate padding mask 
                actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(val_imu_data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(val_imu_data.device)

                max_num = 1

                all_res_list = self.ema.ema_model.sample(val_imu_data, ori_data_cond, cond_mask, padding_mask)

                vis_tag = str(milestone)+"_stage1_sample_"+str(s_idx)

                if self.test_on_train:
                    vis_tag = vis_tag + "_on_train"
                
                self.gen_stage1_res(all_res_list[:max_num], val_data_dict, milestone, vis_tag=vis_tag)

    def process_contact_jpos(self, process_jpos, object_mesh_verts, object_mesh_faces, obj_rot):
        """
        处理HOI接触位置
        
        参数:
            process_jpos: 需要修正的关节位置 T X j X 3
            object_mesh_verts: 物体网格顶点 T X Nv X 3
            object_mesh_faces: 物体网格面 Nf X 3
            obj_rot: 物体旋转矩阵 T X 3 X 3
            
        返回:
            处理后的关节位置
        """
        # process_jpos: T X j X 3 
        # object_mesh_verts: T X Nv X 3 
        # object_mesh_faces: Nf X 3 
        # obj_rot: T X 3 X 3 
        all_contact_labels = []
        all_object_c_idx_list = []
        all_dist = []

        obj_rot = torch.from_numpy(obj_rot).to(process_jpos.device)
        object_mesh_verts = object_mesh_verts.to(process_jpos.device)

        num_joints = process_jpos.shape[1]
        num_steps = process_jpos.shape[0]

        threshold = 0.03 # Use palm position, should be smaller. 
       
        joint2object_dist = torch.cdist(process_jpos, object_mesh_verts.to(process_jpos.device)) # T X j X Nv 
     
        all_dist, all_object_c_idx_list = joint2object_dist.min(dim=2) # T X j 
        all_contact_labels = all_dist < threshold # T X j 

        new_process_jpos = process_jpos.clone() # T X j X 3 

        # For each joint, scan the sequence, if contact is true, then use the corresponding object idx for the 
        # rest of subsequence in contact. 
        for j_idx in range(num_joints):
            continue_prev_contact = False 
            for t_idx in range(num_steps):
                if continue_prev_contact:
                    relative_rot_mat = torch.matmul(obj_rot[t_idx], reference_obj_rot.inverse())
                    curr_contact_normal = torch.matmul(relative_rot_mat, contact_normal[:, None]).squeeze(-1)

                    new_process_jpos[t_idx, j_idx] = object_mesh_verts[t_idx, subseq_contact_v_id] + \
                        curr_contact_normal  # 3  
                
                elif all_contact_labels[t_idx, j_idx] and not continue_prev_contact: # The first contact frame 
                    subseq_contact_v_id = all_object_c_idx_list[t_idx, j_idx]
                    subseq_contact_pos = object_mesh_verts[t_idx, subseq_contact_v_id] # 3 

                    contact_normal = new_process_jpos[t_idx, j_idx] - subseq_contact_pos # Keep using this in the following frames. 

                    reference_obj_rot = obj_rot[t_idx] # 3 X 3 

                    continue_prev_contact = True 

        return new_process_jpos 

    def gen_stage1_res(self, all_res_list, data_dict, pose_only=True, human_only=True):
        """
        生成第一阶段标准化结果，输入为模型生成的结果（含物体预测值），输出为处理后的关节位置和真实关节位置（不含物体）
        
        参数:
            all_res_list: 模型生成的结果 BS X T X ((N+1)*9) 
            data_dict: 数据字典
            pose_only: 是否只返回关节位置
            human_only: 是否只返回人体关节位置
            
        返回:
            处理后的关节位置和真实关节位置
        """
        seq_names = data_dict['seq_name'] # BS 
        num_seq, num_steps, _ = all_res_list.shape
        all_res_list_pos = all_res_list[..., :(NUM_IMU_JOINTS+1)*3]
        all_res_list_rot = all_res_list[..., (NUM_IMU_JOINTS+1)*3:]
        res_human_pos = all_res_list_pos[:, :, :NUM_IMU_JOINTS*3]   #可视化中不用到物体，物体只用来监督stage1
        res_human_rot = all_res_list_rot[:, :, :NUM_IMU_JOINTS*6]
        res_obj_pos = all_res_list_pos[:, :, NUM_IMU_JOINTS*3:]
        res_obj_rot_mat = transforms.rotation_6d_to_matrix(all_res_list_rot[:, :, NUM_IMU_JOINTS*6:])

        normalized_gt_imu_joints_pos = self.extract_joints_pos_rot_data(data_dict['motion'], IMU_JOINTS, pos_only=pose_only) 
        gt_imu_joints_pos = self.ds.de_normalize_imu_joints_min_max(normalized_gt_imu_joints_pos) # BS X T X N X 3
        gt_imu_joints_pos = gt_imu_joints_pos.reshape(-1, num_steps, NUM_IMU_JOINTS, 3)

        # Denormalize predicted results
        pred_imu_joints_pos = self.ds.de_normalize_imu_joints_min_max(res_human_pos) # BS X T X N X 3
        all_processed_imu_joints_pos = pred_imu_joints_pos.clone() 

        if self.add_hand_processing:
            for seq_idx in range(num_seq):
                object_name = seq_names[seq_idx].split("_")[1]
                obj_scale = data_dict['obj_scale'][seq_idx].detach().cpu().numpy()
                obj_trans = data_dict['obj_trans'][seq_idx].detach().cpu().numpy()
                obj_rot = data_dict['obj_rot_mat'][seq_idx].detach().cpu().numpy() 
                if object_name in ["mop", "vacuum"]:
                    obj_bottom_scale = data_dict['obj_bottom_scale'][seq_idx].detach().cpu().numpy() 
                    obj_bottom_trans = data_dict['obj_bottom_trans'][seq_idx].detach().cpu().numpy()
                    obj_bottom_rot = data_dict['obj_bottom_rot_mat'][seq_idx].detach().cpu().numpy()
                else:
                    obj_bottom_scale = None 
                    obj_bottom_trans = None 
                    obj_bottom_rot = None 

                obj_mesh_verts, obj_mesh_faces = self.ds.load_object_geometry(object_name, \
                obj_scale, obj_trans, obj_rot, obj_bottom_scale, obj_bottom_trans, obj_bottom_rot)

                # Add postprocessing for hand positions. 
                curr_seq_pred_imu_joints_pos = self.process_contact_jpos(pred_imu_joints_pos[seq_idx], \
                                    obj_mesh_verts, obj_mesh_faces, obj_rot)

                all_processed_imu_joints_pos[seq_idx] = curr_seq_pred_imu_joints_pos

        if self.use_gt_hand_for_eval:
            all_processed_imu_joints_pos = self.ds.normalize_imu_joints_min_max(gt_imu_joints_pos.cuda())
        else:
            all_processed_imu_joints_pos = self.ds.normalize_imu_joints_min_max(all_processed_imu_joints_pos) # BS X T X 3 X 3

        gt_imu_joints_pos = self.ds.normalize_imu_joints_min_max(gt_imu_joints_pos.cuda())

        #TODO: 后续完善增加旋转的第一阶段输出
        if not pose_only:
            all_processed_imu_joints = torch.cat((all_processed_imu_joints_pos, res_human_rot), dim=-1)
            gt_imu_joints = torch.cat((gt_imu_joints_pos, res_human_rot), dim=-1)

        if human_only:
            return all_processed_imu_joints_pos, gt_imu_joints_pos  
        else:
            return all_processed_imu_joints_pos, gt_imu_joints_pos, res_obj_pos, res_obj_rot_mat

    def run_two_stage_pipeline(self):
        """
        运行两阶段生成管道：第一阶段预测手部位置，第二阶段生成完整人体动作
        
        流程：
        1. 加载预训练的全身扩散模型
        2. 加载预训练的手部位置预测模型
        3. 对测试数据集中的每个序列：
        - 使用第一阶段模型预测手部位置
        - 将预测的手部位置输入第二阶段模型生成完整人体动作
        - 计算各种评估指标
        4. 输出平均评估指标
        
        返回：
            无返回值，但会计算并打印评估指标
        """
        # 加载全身扩散模型
        fullbody_wdir = os.path.join(self.opt.project, self.opt.fullbody_exp_name, "weights")
        repr_dim = 24 * 3 + 22 * 6 
        loss_type = "l1"
        
        # 创建全身扩散模型
        fullbody_diffusion_model = FullBodyCondGaussianDiffusion(
            self.opt,
            d_feats=repr_dim,  # 输出维度：24个关节位置(24×3) + 22个关节旋转(22×6)
            d_model=self.opt.d_model,
            n_dec_layers=self.opt.n_dec_layers,
            n_head=self.opt.n_head,
            d_k=self.opt.d_k,
            d_v=self.opt.d_v,
            max_timesteps=self.opt.window+1,
            out_dim=repr_dim,
            timesteps=1000,
            objective="pred_x0",
            loss_type=loss_type,
            batch_size=self.opt.batch_size
        )
        fullbody_diffusion_model.to(device)

        # 创建全身训练器
        self.fullbody_trainer = FullBodyTrainer(
            self.opt,
            fullbody_diffusion_model,
            train_batch_size=self.batch_size, # 32
            train_lr=1e-4, # 1e-4
            train_num_steps=8000000,         # total training steps
            gradient_accumulate_every=2,    # gradient accumulation steps
            ema_decay=0.995,                # exponential moving average decay
            amp=True,                        # turn on mixed precision
            results_folder=fullbody_wdir,
            use_wandb=False,
            pretrained_path=self.opt.fullbody_pretrained_path if hasattr(self.opt, 'fullbody_pretrained_path') else None
        )
        self.fullbody_trainer.ema.ema_model.eval()
    
        # 加载第一阶段预训练模型
        self.load(self.pretrained_path)
        self.ema.ema_model.eval()

        # 初始化评估指标列表
        s1_global_imu_jpe_list = []    # 第一阶段IMU关节总体误差
        global_hand_jpe_list = [] 
        global_lhand_jpe_list = []
        global_rhand_jpe_list = [] 

        mpvpe_list = []
        mpjpe_list = []

        rot_dist_list = []
        root_trans_err_list = []
        
        collision_percent_list = []
        collision_depth_list = []

        gt_collision_percent_list = []
        gt_collision_depth_list = []

        gt_foot_sliding_jnts_list = []
        foot_sliding_jnts_list = []

        contact_precision_list = []
        contact_recall_list = [] 

        contact_acc_list = []
        contact_f1_score_list = []

        gt_contact_dist_list = []
        contact_dist_list = []

        # 准备数据加载器
        if self.test_on_train:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=8, shuffle=False,
                num_workers=0, pin_memory=False, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=8, shuffle=False,
                num_workers=0, pin_memory=False, drop_last=False) 
        
        num_samples_per_seq = 20 if self.for_quant_eval else 1

        with torch.no_grad():
            for s_idx, val_data_dict in enumerate(test_loader):
                if (not s_idx % 8 == 0) and (not self.for_quant_eval): # 只可视化部分数据
                    continue 

                val_data = val_data_dict['motion'].cuda()
                bs, num_steps, _ = val_data.shape 

                # 生成IMU数据
                val_obj_trans_data = val_data_dict['obj_trans'].cuda()
                val_obj_rot_mat_data = val_data_dict['obj_rot_mat'].cuda()
                val_obj_rot_6d_data = transforms.matrix_to_rotation_6d(val_obj_rot_mat_data)
                val_human_imu_joints_data = self.extract_joints_pos_rot_data(val_data, IMU_JOINTS)
                val_human_imu_joints_data_pos = val_human_imu_joints_data[:, :, :NUM_IMU_JOINTS*3]
                val_human_imu_joints_data_rot = val_human_imu_joints_data[:, :, NUM_IMU_JOINTS*3:]
                val_imu_data = self.compute_imu_data(val_human_imu_joints_data_pos, val_human_imu_joints_data_rot, val_obj_trans_data, val_obj_rot_mat_data)

                # 准备物体特征条件
                val_obj_bps_data = val_data_dict['obj_bps'].cuda()
                val_obj_com_pos = val_data_dict['obj_com_pos'].cuda() 

                val_ori_data_cond = torch.cat((val_obj_com_pos, val_obj_bps_data), dim=-1) # BS X T X (3+1024*3)

                # 生成填充掩码
                val_actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                val_tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                self.window+1) < val_actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                val_padding_mask = val_tmp_mask[:, None, :].to(val_data.device)

                # 为每个序列初始化评估指标列表
                s1_imu_jpe_per_seq = []
            
                hand_jpe_per_seq = []
                lhand_jpe_per_seq = []
                rhand_jpe_per_seq = []

                mpvpe_per_seq = []
                mpjpe_per_seq = []
                
                rot_dist_per_seq = []
                trans_err_per_seq = []
                
                gt_foot_sliding_jnts_per_seq = []
                foot_sliding_jnts_per_seq = []
                
                contact_precision_per_seq = []
                contact_recall_per_seq = [] 

                contact_acc_per_seq = []
                contact_f1_score_per_seq = [] 

                gt_contact_dist_per_seq = []
                contact_dist_per_seq = []

                sampled_all_res_per_seq = []
                for sample_idx in range(num_samples_per_seq):
                    # 第一阶段：预测手部和头部位置与旋转
                    stage1_res = self.ema.ema_model.sample(val_imu_data, val_ori_data_cond, cond_mask=None, padding_mask=val_padding_mask)
                    # stage1_res: BS X T X (3*9) - 3个关节的位置和旋转

                    # 生成可视化标签
                    vis_tag = f"stage1_sample_{s_idx}"
                    if self.add_hand_processing:
                        vis_tag += "_add_hand_processing"
                    if self.test_on_train:
                        vis_tag += "_on_train"
                    if self.use_object_split:
                        vis_tag += "_unseen_objects"

                    # 获取处理后的数据
                    stage1_res, gt_imu_joints_pos = self.gen_stage1_res(stage1_res, val_data_dict)
                
                    # 计算第一阶段评估指标
                    tmp_pred_joints = self.ds.de_normalize_imu_joints_min_max(stage1_res.reshape(bs, num_steps, -1))
                    tmp_gt_joints = self.ds.de_normalize_imu_joints_min_max(gt_imu_joints_pos.reshape(bs, num_steps, -1))

                    for s1_s_idx in range(bs): 
                        s1_imu_jpe = compute_s1_metrics(
                            tmp_pred_joints[s1_s_idx, :val_actual_seq_len[s1_s_idx]], 
                            tmp_gt_joints[s1_s_idx, :val_actual_seq_len[s1_s_idx]]
                        )
                        s1_imu_jpe_per_seq.append(s1_imu_jpe)

                    # 第二阶段：生成全身姿态
                    stage2_res = self.fullbody_trainer.gen_fullbody_from_predicted_joints(stage1_res, val_data_dict)
                    sampled_all_res_per_seq.append(stage2_res) 

                    vis_tag = "two_stage_pipeline_sample_"+str(s_idx)+"_try_"+str(sample_idx)

                    if self.add_hand_processing:
                        vis_tag = vis_tag + "_add_hand_processing"

                    if self.test_on_train:
                        vis_tag = vis_tag + "_on_train"

                    if self.use_object_split:
                        vis_tag += "_unseen_objects"

                    if self.use_gt_hand_for_eval:
                        vis_tag += "_use_gt_hand"

                    # 对每个序列计算评估指标
                    num_seq = stage2_res.shape[0]
                    for seq_idx in range(num_seq):

                        # 修复使用add_hand_processing时的伪影
                        # 当手部位置连续相同时，根部平移会突然改变
                        if self.add_hand_processing:
                            tmp_pred_hand_jpos = stage1_res[seq_idx] # T X 2 X 3 
                            tmp_num_steps = val_actual_seq_len[seq_idx]-1
                            
                            repeat_idx = None 
                            for tmp_idx in range(tmp_num_steps-5, tmp_num_steps):
                                hand_jpos_diff = tmp_pred_hand_jpos[tmp_idx] - tmp_pred_hand_jpos[tmp_idx-1] # 2 X 3 
                                threshold = 0.001
                            
                                if (torch.abs(hand_jpos_diff[0, 0]) < threshold and torch.abs(hand_jpos_diff[0, 1]) < threshold \
                                and torch.abs(hand_jpos_diff[0, 2]) < threshold) or (torch.abs(hand_jpos_diff[1, 0]) < threshold \
                                and torch.abs(hand_jpos_diff[1, 1]) < threshold and torch.abs(hand_jpos_diff[1, 2]) < threshold):
                                    repeat_idx = tmp_idx 
                                    break 
                            
                            if repeat_idx is not None:
                                padding_last = stage2_res[seq_idx:seq_idx+1, repeat_idx-1:repeat_idx] # 1 X 1 X 198 
                                padding_last = padding_last.repeat(1, stage1_res.shape[1]-repeat_idx, 1) # 1 X t' X D 
                                
                                curr_seq_res_list = torch.cat((stage2_res[seq_idx:seq_idx+1, :repeat_idx], padding_last), dim=1)
                            else:
                                curr_seq_res_list = stage2_res[seq_idx:seq_idx+1]
                        else:
                            curr_seq_res_list = stage2_res[seq_idx:seq_idx+1]

                        curr_vis_tag = vis_tag + "_seq_idx_in_bs_"+str(seq_idx) 
                    
                        # 生成预测结果的可视化
                        pred_human_trans_list, pred_human_rot_list, pred_human_jnts_list, pred_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                        self.fullbody_trainer.gen_vis_res(curr_seq_res_list, val_data_dict, \
                        0, vis_tag=curr_vis_tag, for_quant_eval=self.for_quant_eval, selected_seq_idx=seq_idx)
                        
                        # 计算真值指标
                        gt_human_trans_list, gt_human_rot_list, gt_human_jnts_list, gt_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                        self.fullbody_trainer.gen_vis_res(val_data_dict['motion'].cuda()[seq_idx:seq_idx+1], val_data_dict, \
                        0, vis_tag=curr_vis_tag, for_quant_eval=True, selected_seq_idx=seq_idx, compare_with_gt=True)
                    
                        # 计算各种评估指标
                        lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, gt_contact_dist, contact_dist, \
                            gt_foot_sliding_jnts, foot_sliding_jnts, contact_precision, contact_recall, \
                            contact_acc, contact_f1_score = \
                        compute_metrics(gt_human_verts_list, pred_human_verts_list, gt_human_jnts_list, pred_human_jnts_list, human_faces_list, \
                        gt_human_trans_list, pred_human_trans_list, gt_human_rot_list, pred_human_rot_list, \
                        obj_verts_list, obj_faces_list, actual_len_list, use_joints24=True)

                        # 收集评估指标
                        hand_jpe_per_seq.append(hand_jpe)
                        lhand_jpe_per_seq.append(lhand_jpe)
                        rhand_jpe_per_seq.append(rhand_jpe) 

                        mpvpe_per_seq.append(mpvpe)
                        mpjpe_per_seq.append(mpjpe)
                        
                        rot_dist_per_seq.append(rot_dist)
                        trans_err_per_seq.append(trans_err) 
                        
                        gt_foot_sliding_jnts_per_seq.append(gt_foot_sliding_jnts)
                        foot_sliding_jnts_per_seq.append(foot_sliding_jnts)
                        
                        contact_precision_per_seq.append(contact_precision)
                        contact_recall_per_seq.append(contact_recall) 

                        contact_acc_per_seq.append(contact_acc) 
                        contact_f1_score_per_seq.append(contact_f1_score) 

                        gt_contact_dist_per_seq.append(gt_contact_dist)
                        contact_dist_per_seq.append(contact_dist)

                # 如果进行定量评估，选择每个序列中最佳的样本
                if self.for_quant_eval:
                    # 重塑评估指标数组
                    s1_imu_jpe_per_seq = np.asarray(s1_imu_jpe_per_seq).reshape(num_samples_per_seq, num_seq)

                    hand_jpe_per_seq = np.asarray(hand_jpe_per_seq).reshape(num_samples_per_seq, num_seq)
                    lhand_jpe_per_seq = np.asarray(lhand_jpe_per_seq).reshape(num_samples_per_seq, num_seq)
                    rhand_jpe_per_seq = np.asarray(rhand_jpe_per_seq).reshape(num_samples_per_seq, num_seq) 

                    mpvpe_per_seq = np.asarray(mpvpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                    mpjpe_per_seq = np.asarray(mpjpe_per_seq).reshape(num_samples_per_seq, num_seq) # Sample_num X BS 
                    
                    rot_dist_per_seq = np.asarray(rot_dist_per_seq).reshape(num_samples_per_seq, num_seq)
                    trans_err_per_seq = np.asarray(trans_err_per_seq).reshape(num_samples_per_seq, num_seq) 
                    
                    gt_foot_sliding_jnts_per_seq = np.asarray(gt_foot_sliding_jnts_per_seq).reshape(num_samples_per_seq, num_seq)
                    foot_sliding_jnts_per_seq = np.asarray(foot_sliding_jnts_per_seq).reshape(num_samples_per_seq, num_seq) 
                    
                    contact_precision_per_seq = np.asarray(contact_precision_per_seq).reshape(num_samples_per_seq, num_seq) 
                    contact_recall_per_seq = np.asarray(contact_recall_per_seq).reshape(num_samples_per_seq, num_seq) 

                    contact_acc_per_seq = np.asarray(contact_acc_per_seq).reshape(num_samples_per_seq, num_seq) 
                    contact_f1_score_per_seq = np.asarray(contact_f1_score_per_seq).reshape(num_samples_per_seq, num_seq) 

                    gt_contact_dist_per_seq = np.asarray(gt_contact_dist_per_seq).reshape(num_samples_per_seq, num_seq)
                    contact_dist_per_seq = np.asarray(contact_dist_per_seq).reshape(num_samples_per_seq, num_seq) 

                    # 根据MPJPE选择最佳样本
                    best_sample_idx = mpjpe_per_seq.argmin(axis=0) # sample_num 

                    # 获取最佳样本的评估指标
                    s1_imu_jpe = s1_imu_jpe_per_seq[best_sample_idx, list(range(num_seq))]

                    hand_jpe = hand_jpe_per_seq[best_sample_idx, list(range(num_seq))] # BS 
                    lhand_jpe = lhand_jpe_per_seq[best_sample_idx, list(range(num_seq))]
                    rhand_jpe = rhand_jpe_per_seq[best_sample_idx, list(range(num_seq))]
                    
                    mpvpe = mpvpe_per_seq[best_sample_idx, list(range(num_seq))]
                    mpjpe = mpjpe_per_seq[best_sample_idx, list(range(num_seq))]
                    
                    rot_dist = rot_dist_per_seq[best_sample_idx, list(range(num_seq))]
                    trans_err = trans_err_per_seq[best_sample_idx, list(range(num_seq))]
                
                    gt_foot_sliding_jnts = gt_foot_sliding_jnts_per_seq[best_sample_idx, list(range(num_seq))]
                    foot_sliding_jnts = foot_sliding_jnts_per_seq[best_sample_idx, list(range(num_seq))]

                    contact_precision_seq = contact_precision_per_seq[best_sample_idx, list(range(num_seq))]
                    contact_recall_seq = contact_recall_per_seq[best_sample_idx, list(range(num_seq))] 
                    contact_acc_seq = contact_acc_per_seq[best_sample_idx, list(range(num_seq))]
                    contact_f1_score_seq = contact_f1_score_per_seq[best_sample_idx, list(range(num_seq))]

                    gt_contact_dist_seq = gt_contact_dist_per_seq[best_sample_idx, list(range(num_seq))]
                    contact_dist_seq = contact_dist_per_seq[best_sample_idx, list(range(num_seq))] 

                    # 获取最佳样本的结果
                    sampled_all_res_per_seq = torch.stack(sampled_all_res_per_seq) # K X BS X T X D 
                    best_sampled_all_res = sampled_all_res_per_seq[best_sample_idx, list(range(num_seq))] # BS X T X D 
                    
                    # 计算最佳样本的碰撞指标
                    num_seq = best_sampled_all_res.shape[0]
                    for seq_idx in range(num_seq):
                        pred_human_trans_list, pred_human_rot_list, pred_human_jnts_list, pred_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                        self.fullbody_trainer.gen_vis_res(best_sampled_all_res[seq_idx:seq_idx+1], val_data_dict, \
                        0, vis_tag=vis_tag, for_quant_eval=True, selected_seq_idx=seq_idx)
                        gt_human_trans_list, gt_human_rot_list, gt_human_jnts_list, gt_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                        self.fullbody_trainer.gen_vis_res(val_data_dict['motion'].cuda()[seq_idx:seq_idx+1], val_data_dict, \
                        0, vis_tag=vis_tag, for_quant_eval=True, selected_seq_idx=seq_idx, compare_with_gt=True)

                        obj_scale = val_data_dict['obj_scale'][seq_idx]
                        obj_trans = val_data_dict['obj_trans'][seq_idx]
                        obj_rot_mat = val_data_dict['obj_rot_mat'][seq_idx]
                        actual_len = val_data_dict['seq_len'][seq_idx]
                        object_name = val_data_dict['obj_name'][seq_idx]

                        # 计算预测结果的碰撞指标
                        pred_collision_percent, pred_collision_depth = compute_collision(pred_human_verts_list.cpu(), \
                            human_faces_list, obj_verts_list.cpu(), obj_faces_list, object_name, \
                            obj_scale, obj_rot_mat, obj_trans, actual_len)
                            
                        # 计算真实数据的碰撞指标
                        gt_collision_percent, gt_collision_depth = compute_collision(gt_human_verts_list.cpu(), \
                            human_faces_list, obj_verts_list.cpu(), obj_faces_list, object_name, \
                            obj_scale, obj_rot_mat, obj_trans, actual_len)

                        collision_percent_list.append(pred_collision_percent)
                        collision_depth_list.append(pred_collision_depth)
                        gt_collision_percent_list.append(gt_collision_percent)
                        gt_collision_depth_list.append(gt_collision_depth) 
                        
                    # 收集全局评估指标
                    for tmp_seq_idx in range(num_seq):
                        s1_global_imu_jpe_list.append(s1_imu_jpe[tmp_seq_idx])

                        global_hand_jpe_list.append(hand_jpe[tmp_seq_idx])
                        global_lhand_jpe_list.append(lhand_jpe[tmp_seq_idx])
                        global_rhand_jpe_list.append(rhand_jpe[tmp_seq_idx])

                        mpvpe_list.append(mpvpe[tmp_seq_idx])
                        mpjpe_list.append(mpjpe[tmp_seq_idx])
                        
                        rot_dist_list.append(rot_dist[tmp_seq_idx])
                        root_trans_err_list.append(trans_err[tmp_seq_idx])
                        
                        gt_foot_sliding_jnts_list.append(gt_foot_sliding_jnts[tmp_seq_idx])
                        foot_sliding_jnts_list.append(foot_sliding_jnts[tmp_seq_idx])

                        contact_precision_list.append(contact_precision_seq[tmp_seq_idx])
                        contact_recall_list.append(contact_recall_seq[tmp_seq_idx])

                        contact_acc_list.append(contact_acc_seq[tmp_seq_idx])
                        contact_f1_score_list.append(contact_f1_score_seq[tmp_seq_idx])

                        gt_contact_dist_list.append(gt_contact_dist_seq[tmp_seq_idx])
                        contact_dist_list.append(contact_dist_seq[tmp_seq_idx])

        # 如果进行定量评估，计算并打印平均评估指标
        if self.for_quant_eval:
            s1_mean_imu_jpe = np.asarray(s1_global_imu_jpe_list).mean() 

            mean_hand_jpe = np.asarray(global_hand_jpe_list).mean() 
            mean_lhand_jpe = np.asarray(global_lhand_jpe_list).mean()
            mean_rhand_jpe = np.asarray(global_rhand_jpe_list).mean()
            
            mean_mpvpe = np.asarray(mpvpe_list).mean()
            mean_mpjpe = np.asarray(mpjpe_list).mean() 

            mean_rot_dist = np.asarray(rot_dist_list).mean() 
            mean_root_trans_err = np.asarray(root_trans_err_list).mean()
            
            mean_collision_percent = np.asarray(collision_percent_list).mean()
            mean_collision_depth = np.asarray(collision_depth_list).mean() 

            gt_mean_collision_percent = np.asarray(gt_collision_percent_list).mean()
            gt_mean_collision_depth = np.asarray(gt_collision_depth_list).mean() 
            
            mean_gt_fsliding_jnts = np.asarray(gt_foot_sliding_jnts_list).mean()
            mean_fsliding_jnts = np.asarray(foot_sliding_jnts_list).mean() 

            mean_contact_precision = np.asarray(contact_precision_list).mean()
            mean_contact_recall = np.asarray(contact_recall_list).mean() 

            mean_contact_acc = np.asarray(contact_acc_list).mean()
            mean_contact_f1_score = np.asarray(contact_f1_score_list).mean() 

            mean_gt_contact_dist = np.asarray(gt_contact_dist_list).mean()
            mean_contact_dist = np.asarray(contact_dist_list).mean()


            print("*****************************************Quantitative Evaluation*****************************************")
            print("The number of sequences: {0}".format(len(mpjpe_list)))
            print("Stage 1 IMU JPE: {0}".format(s1_mean_imu_jpe))
            print("Left Hand JPE: {0}, Right Hand JPE: {1}, Two Hands JPE: {2}".format(mean_lhand_jpe, mean_rhand_jpe, mean_hand_jpe))
            print("MPJPE: {0}, MPVPE: {1}, Root Trans: {2}, Global Rot Err: {3}".format(mean_mpjpe, mean_mpvpe, mean_root_trans_err, mean_rot_dist))
            print("Foot sliding jnts: {0}, GT Foot sliding jnts: {1}".format(mean_fsliding_jnts, mean_gt_fsliding_jnts))
            print("Collision percent: {0}, Collision depth: {1}".format(mean_collision_percent, mean_collision_depth))
            print("GT Collision percent: {0}, GT Collision depth: {1}".format(gt_mean_collision_percent, gt_mean_collision_depth))
            print("Contact precision: {0}, Contact recall: {1}".format(mean_contact_precision, mean_contact_recall))
            print("Contact Acc: {0}, Contact F1 score: {1}".format(mean_contact_acc, mean_contact_f1_score)) 
            print("Contact dist: {0}, GT Contact dist: {1}".format(mean_contact_dist, mean_gt_contact_dist))

    def process_single_sequence(self, val_data_dict, s_idx):
        """
        处理单个序列：包括第一阶段预测、第二阶段全身生成、计算评估指标等，
        返回一个字典，包含该序列的各项评估指标。
        注意：这里简化了内部细节，你需要根据实际代码将各步骤完整封装进去。
        """
        with torch.no_grad():
            val_data = val_data_dict['motion'].cuda()
            val_obj_trans_data = val_data_dict['obj_trans'].cuda()
            val_obj_rot_mat_data = val_data_dict['obj_rot_mat'].cuda()
            val_obj_rot_6d_data = transforms.matrix_to_rotation_6d(val_obj_rot_mat_data)
            val_human_imu_joints_data = self.extract_joints_pos_rot_data(val_data, IMU_JOINTS)
            val_human_imu_joints_data_pos = val_human_imu_joints_data[:, :, :NUM_IMU_JOINTS*3]
            val_human_imu_joints_data_rot = val_human_imu_joints_data[:, :, NUM_IMU_JOINTS*3:]
            val_imu_data = self.compute_imu_data(val_human_imu_joints_data_pos, val_human_imu_joints_data_rot, val_obj_trans_data, val_obj_rot_mat_data)

            # 准备物体特征条件
            val_obj_bps_data = val_data_dict['obj_bps'].cuda()
            val_obj_com_pos = val_data_dict['obj_com_pos'].cuda() 

            val_ori_data_cond = torch.cat((val_obj_com_pos, val_obj_bps_data), dim=-1) # BS X T X (3+1024*3)

            # 生成填充掩码
            val_actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
            val_tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
            self.window+1) < val_actual_seq_len[:, None].repeat(1, self.window+1)
            # BS X max_timesteps
            val_padding_mask = val_tmp_mask[:, None, :].to(val_data.device)

            # 第一阶段：预测手部和头部位置与旋转
            stage1_res = self.ema.ema_model.sample(val_imu_data, val_ori_data_cond, cond_mask=None, padding_mask=val_padding_mask)
            # stage1_res: BS X T X ((N+1)*9)

            # 生成可视化标签
            vis_tag = f"stage1_sample_{s_idx}"
            if self.add_hand_processing:
                vis_tag += "_add_hand_processing"
            if self.test_on_train:
                vis_tag += "_on_train"
            if self.use_object_split:
                vis_tag += "_unseen_objects"

            # 获取处理后的数据, res_human: BS X T X (N*3), res_obj_pos: BS X T X 3, res_obj_rot_mat: BS X T X 3 X 3
            res_human, gt_imu_joints_pos, res_obj_pos, res_obj_rot_mat = self.gen_stage1_res(stage1_res, val_data_dict, human_only=False)
        
            # 第二阶段：生成全身姿态
            stage2_res = self.fullbody_trainer.gen_fullbody_from_predicted_joints(res_human, val_data_dict)

            vis_tag = "vis_sample_"+str(s_idx)

            if self.add_hand_processing:
                vis_tag = vis_tag + "_add_hand_processing"

            if self.test_on_train:
                vis_tag = vis_tag + "_on_train"

            if self.use_object_split:
                vis_tag += "_unseen_objects"

            if self.use_gt_hand_for_eval:
                vis_tag += "_use_gt_hand"

            # 对每个序列计算结果
            num_seq = stage2_res.shape[0]
            for seq_idx in range(num_seq):

                # 修复使用add_hand_processing时的伪影
                # 当手部位置连续相同时，根部平移会突然改变
                if self.add_hand_processing:
                    tmp_pred_hand_jpos = res_human[seq_idx] # T X 2 X 3 
                    tmp_num_steps = val_actual_seq_len[seq_idx]-1
                    
                    repeat_idx = None 
                    for tmp_idx in range(tmp_num_steps-5, tmp_num_steps):
                        hand_jpos_diff = tmp_pred_hand_jpos[tmp_idx] - tmp_pred_hand_jpos[tmp_idx-1] # 2 X 3 
                        threshold = 0.001
                    
                        if (torch.abs(hand_jpos_diff[0, 0]) < threshold and torch.abs(hand_jpos_diff[0, 1]) < threshold \
                        and torch.abs(hand_jpos_diff[0, 2]) < threshold) or (torch.abs(hand_jpos_diff[1, 0]) < threshold \
                        and torch.abs(hand_jpos_diff[1, 1]) < threshold and torch.abs(hand_jpos_diff[1, 2]) < threshold):
                            repeat_idx = tmp_idx 
                            break 
                    
                    if repeat_idx is not None:
                        padding_last = stage2_res[seq_idx:seq_idx+1, repeat_idx-1:repeat_idx] # 1 X 1 X 198 
                        padding_last = padding_last.repeat(1, res_human.shape[1]-repeat_idx, 1) # 1 X t' X D 
                        
                        curr_seq_res_list = torch.cat((stage2_res[seq_idx:seq_idx+1, :repeat_idx], padding_last), dim=1)
                    else:
                        curr_seq_res_list = stage2_res[seq_idx:seq_idx+1]
                else:
                    curr_seq_res_list = stage2_res[seq_idx:seq_idx+1]

                curr_vis_tag = vis_tag + "_seq_idx_in_bs_"+str(seq_idx) 
            
                # 生成预测结果的可视化
                self.fullbody_trainer.gen_vis_res(curr_seq_res_list, val_data_dict, \
                    0, vis_tag=curr_vis_tag, for_quant_eval=self.for_quant_eval, selected_seq_idx=seq_idx, \
                    res_obj_pos=res_obj_pos[seq_idx], res_obj_rot_mat=res_obj_rot_mat[seq_idx])

    def run_two_stage_pipeline_vis_only(self):
        """
        运行两阶段生成管道，使用多线程同时处理多个序列，生成可视化结果。
        """
        # 加载全身扩散模型、第一阶段模型和相关 EMA 模型，这部分保持不变
        fullbody_wdir = os.path.join(self.opt.project, self.opt.fullbody_exp_name, "weights")
        repr_dim = 24 * 3 + 22 * 6 
        loss_type = "l1"
        
        # 创建全身扩散模型
        fullbody_diffusion_model = FullBodyCondGaussianDiffusion(
            self.opt,
            d_feats=repr_dim,  # 输出维度：24个关节位置(24×3) + 22个关节旋转(22×6)
            d_model=self.opt.d_model,
            n_dec_layers=self.opt.n_dec_layers,
            n_head=self.opt.n_head,
            d_k=self.opt.d_k,
            d_v=self.opt.d_v,
            max_timesteps=self.opt.window+1,
            out_dim=repr_dim,
            timesteps=1000,
            objective="pred_x0",
            loss_type=loss_type,
            batch_size=self.opt.batch_size
        )
        fullbody_diffusion_model.to(device)

        # 创建全身训练器
        self.fullbody_trainer = FullBodyTrainer(
            self.opt,
            fullbody_diffusion_model,
            train_batch_size=self.batch_size, # 32
            train_lr=1e-4, # 1e-4
            train_num_steps=8000000,         # total training steps
            gradient_accumulate_every=2,    # gradient accumulation steps
            ema_decay=0.995,                # exponential moving average decay
            amp=True,                        # turn on mixed precision
            results_folder=fullbody_wdir,
            use_wandb=False,
            pretrained_path=self.opt.fullbody_pretrained_path if hasattr(self.opt, 'fullbody_pretrained_path') else None
        )
        self.fullbody_trainer.ema.ema_model.eval()
    
        # 加载第一阶段预训练模型
        self.load(self.pretrained_path)
        self.ema.ema_model.eval()
        
        # 选择测试数据加载器
        if self.test_on_train:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=False, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=False, drop_last=False)
        
        # # 将 test_loader 中的每个样本（序列）转换为列表
        # sequences = []
        # for s_idx, val_data_dict in enumerate(test_loader):
        #     if not s_idx % 8 == 0: # 只可视化部分数据
        #         continue 
        #     sequences.append(val_data_dict)
        
        # # 使用多线程处理多个序列，注意我们用 ThreadPoolExecutor 而非 ProcessPoolExecutor
        # with ThreadPoolExecutor(max_workers=2) as executor:
        #     futures = { executor.submit(self.process_single_sequence, seq, idx): idx
        #                 for idx, seq in enumerate(sequences) }
        #     for future in tqdm(as_completed(futures), total=len(futures), desc="处理序列"):
        #         try:
        #             res = future.result()
        #         except Exception as e:
        #             print("处理序列出错：", e)
        for s_idx, val_data_dict in enumerate(test_loader):
            if not s_idx % 8 == 0: # 只可视化部分数据
                continue 
            with torch.no_grad():
                val_data = val_data_dict['motion'].cuda()
                val_obj_trans_data = val_data_dict['obj_trans'].cuda()
                val_obj_rot_mat_data = val_data_dict['obj_rot_mat'].cuda()
                val_obj_rot_6d_data = transforms.matrix_to_rotation_6d(val_obj_rot_mat_data)
                val_human_imu_joints_data = self.extract_joints_pos_rot_data(val_data, IMU_JOINTS)
                val_human_imu_joints_data_pos = val_human_imu_joints_data[:, :, :NUM_IMU_JOINTS*3]
                val_human_imu_joints_data_rot = val_human_imu_joints_data[:, :, NUM_IMU_JOINTS*3:]
                val_imu_data = self.compute_imu_data(val_human_imu_joints_data_pos, val_human_imu_joints_data_rot, val_obj_trans_data, val_obj_rot_mat_data)

                # 准备物体特征条件
                val_obj_bps_data = val_data_dict['obj_bps'].cuda()
                val_obj_com_pos = val_data_dict['obj_com_pos'].cuda() 

                val_ori_data_cond = torch.cat((val_obj_com_pos, val_obj_bps_data), dim=-1) # BS X T X (3+1024*3)

                # 生成填充掩码
                val_actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                val_tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                self.window+1) < val_actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                val_padding_mask = val_tmp_mask[:, None, :].to(val_data.device)

                # 第一阶段：预测手部和头部位置与旋转
                stage1_res = self.ema.ema_model.sample(val_imu_data, val_ori_data_cond, cond_mask=None, padding_mask=val_padding_mask)
                # stage1_res: BS X T X ((N+1)*9)

                # 生成可视化标签
                vis_tag = f"stage1_sample_{s_idx}"
                if self.add_hand_processing:
                    vis_tag += "_add_hand_processing"
                if self.test_on_train:
                    vis_tag += "_on_train"
                if self.use_object_split:
                    vis_tag += "_unseen_objects"

                # 获取处理后的数据, res_human: BS X T X (N*3), res_obj_pos: BS X T X 3, res_obj_rot_mat: BS X T X 3 X 3
                res_human, gt_imu_joints_pos, res_obj_pos, res_obj_rot_mat = self.gen_stage1_res(stage1_res, val_data_dict, human_only=False)
            
                # 第二阶段：生成全身姿态
                stage2_res = self.fullbody_trainer.gen_fullbody_from_predicted_joints(res_human, val_data_dict)

                vis_tag = "vis_sample_"+str(s_idx)

                if self.add_hand_processing:
                    vis_tag = vis_tag + "_add_hand_processing"

                if self.test_on_train:
                    vis_tag = vis_tag + "_on_train"

                if self.use_object_split:
                    vis_tag += "_unseen_objects"

                if self.use_gt_hand_for_eval:
                    vis_tag += "_use_gt_hand"

                # 对每个序列计算结果
                num_seq = stage2_res.shape[0]
                for seq_idx in range(num_seq):

                    # 修复使用add_hand_processing时的伪影
                    # 当手部位置连续相同时，根部平移会突然改变
                    if self.add_hand_processing:
                        tmp_pred_hand_jpos = res_human[seq_idx] # T X 2 X 3 
                        tmp_num_steps = val_actual_seq_len[seq_idx]-1
                        
                        repeat_idx = None 
                        for tmp_idx in range(tmp_num_steps-5, tmp_num_steps):
                            hand_jpos_diff = tmp_pred_hand_jpos[tmp_idx] - tmp_pred_hand_jpos[tmp_idx-1] # 2 X 3 
                            threshold = 0.001
                        
                            if (torch.abs(hand_jpos_diff[0, 0]) < threshold and torch.abs(hand_jpos_diff[0, 1]) < threshold \
                            and torch.abs(hand_jpos_diff[0, 2]) < threshold) or (torch.abs(hand_jpos_diff[1, 0]) < threshold \
                            and torch.abs(hand_jpos_diff[1, 1]) < threshold and torch.abs(hand_jpos_diff[1, 2]) < threshold):
                                repeat_idx = tmp_idx 
                                break 
                        
                        if repeat_idx is not None:
                            padding_last = stage2_res[seq_idx:seq_idx+1, repeat_idx-1:repeat_idx] # 1 X 1 X 198 
                            padding_last = padding_last.repeat(1, res_human.shape[1]-repeat_idx, 1) # 1 X t' X D 
                            
                            curr_seq_res_list = torch.cat((stage2_res[seq_idx:seq_idx+1, :repeat_idx], padding_last), dim=1)
                        else:
                            curr_seq_res_list = stage2_res[seq_idx:seq_idx+1]
                    else:
                        curr_seq_res_list = stage2_res[seq_idx:seq_idx+1]

                    curr_vis_tag = vis_tag + "_seq_idx_in_bs_"+str(seq_idx) 
                
                    # 生成预测结果的可视化
                    self.fullbody_trainer.gen_vis_res(curr_seq_res_list, val_data_dict, \
                        0, vis_tag=curr_vis_tag, for_quant_eval=self.for_quant_eval, selected_seq_idx=seq_idx, \
                        res_obj_pos=res_obj_pos[seq_idx], res_obj_rot_mat=res_obj_rot_mat[seq_idx])

            


def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

   
    loss_type = "l1"
  
    diffusion_model = CondGaussianDiffusion(
        opt, 
        input_dim=(NUM_IMU_JOINTS+1)*6,
        out_dim=(NUM_IMU_JOINTS+1)*9,
        d_model=opt.d_model,
        n_dec_layers=opt.n_dec_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        max_timesteps=opt.window+1,
        timesteps=1000,
        objective="pred_x0",
        loss_type=loss_type,
        batch_size=opt.batch_size
    )
   
    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size,
        train_lr=opt.learning_rate,
        train_num_steps=400000,
        save_and_sample_every=40000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        results_folder=str(wdir),
        pretrained_path=opt.pretrained_path if hasattr(opt, 'pretrained_path') else None
    )

    trainer.train()


def run_sample(opt, device, run_pipeline=False, run_pipeline_vis_only=False):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    loss_type = "l1"
    
    diffusion_model = CondGaussianDiffusion(
        opt, 
        input_dim=(NUM_IMU_JOINTS+1)*6,
        out_dim=(NUM_IMU_JOINTS+1)*9,
        d_model=opt.d_model,
        n_dec_layers=opt.n_dec_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        max_timesteps=opt.window+1,
        timesteps=1000,
        objective="pred_x0",
        loss_type=loss_type,
        batch_size=opt.batch_size
    )

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size,
        train_lr=opt.learning_rate,
        train_num_steps=400000,
        save_and_sample_every=40000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        results_folder=str(wdir),
        pretrained_path=opt.pretrained_path if hasattr(opt, 'pretrained_path') else None,
        use_wandb=False
    )
    
    if run_pipeline:
        trainer.run_two_stage_pipeline()
    elif run_pipeline_vis_only:
        trainer.run_two_stage_pipeline_vis_only()
    else:
        trainer.cond_sample_res()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='output folder for weights and visualizations')
    parser.add_argument('--wandb_pj_name', type=str, default='wandb_proj_name', help='wandb project name')
    parser.add_argument('--entity', default='wandb_account_name', help='W&B entity')
    parser.add_argument('--exp_name', default='stage1_exp_out', help='save to project/exp_name')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--fullbody_exp_name', default='stage2_exp_out', help='project/fullbody_exp_name')
    parser.add_argument('--fullbody_pretrained_path', type=str, default="", help='pretrained_path')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--pretrained_path', type=str, default=None, help='path to pretrained model')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results on training dataset 
    parser.add_argument("--test_sample_res_on_train", action="store_true")

    # For running the whole pipeline. 
    parser.add_argument("--run_whole_pipeline", action="store_true")

    parser.add_argument("--run_pipeline_vis_only", action="store_true")

    parser.add_argument("--add_hand_processing", action="store_true")

    parser.add_argument("--for_quant_eval", action="store_true")

    parser.add_argument("--use_gt_hand_for_eval", action="store_true")

    parser.add_argument("--use_object_split", action="store_true")

    parser.add_argument('--data_root_folder', default='data', help='root folder for dataset')


    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    elif opt.run_whole_pipeline:
        run_sample(opt, device, run_pipeline=True)
    elif opt.run_pipeline_vis_only:
        run_sample(opt, device, run_pipeline_vis_only=True)
    else:
        run_train(opt, device)
