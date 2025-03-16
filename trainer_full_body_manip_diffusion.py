import argparse
import os
import numpy as np
import yaml
import random
import json 

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
from manip.data.config import IMU_JOINTS, IMU_JOINT_NAMES, NUM_IMU_JOINTS
from manip.model.transformer_fullbody_cond_diffusion_model import CondGaussianDiffusion, cosine_beta_schedule 

from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object, save_verts_faces_to_mesh_file_w_object_and_gt

from evaluation_metrics import compute_metrics 
from evaluation_metrics import compute_collision

from matplotlib import pyplot as plt

# def run_smplx_model(root_trans, aa_rot_rep, betas, gender, bm_dict, return_joints24=False):
#     # root_trans: BS X T X 3
#     # aa_rot_rep: BS X T X 22 X 3 
#     # betas: BS X 16
#     # gender: BS 
#     bs, num_steps, num_joints, _ = aa_rot_rep.shape
#     if num_joints != 52:
#         padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
#         aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

#     aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3 
#     betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16 
#     gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
#     gender = gender.reshape(-1).tolist() # (BS*T)

#     smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3  
#     smpl_betas = betas # (BS*T) X 16
#     smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
#     smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
#     smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 

#     B = smpl_trans.shape[0] # (BS*T) 

#     smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
#     # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
#     gender_names = ['male', 'female']
#     # pred_joints = []
#     # pred_verts = []
#     # prev_nbidx = 0
#     # cat_idx_map = np.ones((B), dtype=int)*-1
#     # for gender_name in gender_names:
#     #     gender_idx = np.array(gender) == gender_name
#     #     nbidx = np.sum(gender_idx)

#     #     cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=int)
#     #     prev_nbidx += nbidx

#     #     gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

#     #     if nbidx == 0:
#     #         # skip if no frames for this gender
#     #         continue
        
#     #     # reconstruct SMPL
#     #     cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
#     #     bm = bm_dict[gender_name]

#     #     pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
#     #             betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
        
#     #     pred_joints.append(pred_body.Jtr)
#     #     pred_verts.append(pred_body.v)

#     pred_joints = []
#     pred_verts = []
#     prev_nbidx = 0
#     cat_idx_map = np.ones((B), dtype=int)*-1
#     device = aa_rot_rep.device  # 获取输入数据的设备信息

#     for gender_name in gender_names:
#         gender_idx = np.array(gender) == gender_name
#         nbidx = np.sum(gender_idx)

#         cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=int)
#         prev_nbidx += nbidx

#         # 生成占位符张量（即使 nbidx=0）
#         if nbidx == 0:
#             # 添加空张量以保持维度一致
#             pred_joints.append(torch.zeros((0, 52, 3), device=device))  # 52 是 SMPL-X 关节数
#             pred_verts.append(torch.zeros((0, 6890, 3), device=device))  # 6890 是顶点数
#             continue

#         # 处理当前性别的数据
#         gender_smpl_vals = [val[gender_idx] for val in smpl_vals]
#         cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
#         bm = bm_dict[gender_name]

#         # 添加维度检查
#         assert cur_pred_trans.shape[0] == nbidx, f"维度不匹配: {cur_pred_trans.shape[0]} vs {nbidx}"
        
#         pred_body = bm(
#             pose_body=cur_pred_pose,
#             pose_hand=cur_pred_pose_hand,
#             betas=cur_betas,
#             root_orient=cur_pred_orient,
#             trans=cur_pred_trans,
#             pose_jaw=torch.zeros((nbidx, 3), device=device),
#             pose_eye=torch.zeros((nbidx, 6), device=device),
#             expression=torch.zeros((nbidx, 10), device=device)  # num_expressions通常是10
#         )

#         # 检查输出维度
#         assert pred_body.Jtr.shape == (nbidx, 55, 3), f"关节维度错误: {pred_body.Jtr.shape}"
#         assert pred_body.v.shape == (nbidx, 6890, 3), f"顶点维度错误: {pred_body.v.shape}"

#         pred_joints.append(pred_body.Jtr)
#         pred_verts.append(pred_body.v)

#     # cat all genders and reorder to original batch ordering
#     if return_joints24:
#         x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0) # () X 52 X 3 
#         lmiddle_index= 28 
#         rmiddle_index = 43 
#         x_pred_smpl_joints = torch.cat((x_pred_smpl_joints_all[:, :22, :], \
#             x_pred_smpl_joints_all[:, lmiddle_index:lmiddle_index+1, :], \
#             x_pred_smpl_joints_all[:, rmiddle_index:rmiddle_index+1, :]), dim=1) 
#     else:
#         x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
        
#     x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

#     x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
#     x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 

    
#     x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3  
#     x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

#     mesh_faces = pred_body.f 
    
#     return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces 

def run_smplx_model(root_trans, aa_rot_rep, betas, gender, bm_dict, return_joints24=False):
    """
    运行SMPLX模型并返回关节和顶点数据
    
    参数:
    root_trans: BS X T X 3 - 根节点平移
    aa_rot_rep: BS X T X 22 X 3 - 轴角旋转表示
    betas: BS X 16 - 体型参数
    gender: BS - 性别列表 ('male'或'female')
    bm_dict: 字典 - 包含各性别的身体模型
    return_joints24: 布尔值 - 是否返回24个关节点
    
    返回:
    x_pred_smpl_joints: 关节坐标
    x_pred_smpl_verts: 顶点坐标
    mesh_faces: 网格面片
    """
    # 获取张量的形状信息
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    device = aa_rot_rep.device
    
    # 处理关节数不为52的情况
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3, device=device)  # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2)  # BS X T X 52 X 3 

    # 重塑张量形状以适应SMPLX模型
    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3)  # (BS*T) X n_joints X 3 
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1)  # (BS*T) X 16 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist()  # (BS*T)

    # 准备SMPLX模型需要的输入
    smpl_trans = root_trans.reshape(-1, 3)  # (BS*T) X 3  
    smpl_betas = betas  # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :]  # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63)  # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90)  # (BS*T) X 90 

    # 总批次大小
    B = smpl_trans.shape[0]  # (BS*T) 

    # 将输入整合到一个列表
    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    
    # 分性别处理
    gender_names = ['male', 'female']
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=int)*-1
    
    for gender_name in gender_names:
        # 获取当前性别的样本索引
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)
        
        # 更新索引映射
        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=int)
        prev_nbidx += nbidx
        
        # 如果没有当前性别的样本，添加空张量并继续
        if nbidx == 0:
            # 添加空张量以保持维度一致
            pred_joints.append(torch.zeros((0, 55, 3), device=device))  # SMPLX有55个关节
            pred_verts.append(torch.zeros((0, 10475, 3), device=device))  # SMPLX有10475个顶点
            continue
        
        # 处理当前性别的数据
        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]
        
        # 获取模型中expression参数的大小
        num_expressions = 10  # 默认值，如果可能请从模型中获取
        if hasattr(bm, 'expression'):
            num_expressions = bm.expression.shape[1]
        
        # 调用身体模型，显式提供所有必要参数
        pred_body = bm(
            pose_body=cur_pred_pose,
            pose_hand=cur_pred_pose_hand,
            betas=cur_betas,
            root_orient=cur_pred_orient,
            trans=cur_pred_trans,
            # 添加必要的参数，确保维度匹配
            pose_jaw=torch.zeros((nbidx, 3), device=device),
            pose_eye=torch.zeros((nbidx, 6), device=device),
            expression=torch.zeros((nbidx, num_expressions), device=device)
        )
        
        # 保存预测结果
        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)
    
    # 合并所有性别的结果并恢复原始批次顺序
    if return_joints24:
        # 特殊处理，返回24个关节
        x_pred_smpl_joints_all = torch.cat(pred_joints, dim=0)  # (BS*T) X 55 X 3 
        lmiddle_index = 28 
        rmiddle_index = 43 
        x_pred_smpl_joints = torch.cat((
            x_pred_smpl_joints_all[:, :22, :],
            x_pred_smpl_joints_all[:, lmiddle_index:lmiddle_index+1, :],
            x_pred_smpl_joints_all[:, rmiddle_index:rmiddle_index+1, :]
        ), dim=1)  # 返回选定的24个关节
    else:
        # 返回所有关节
        x_pred_smpl_joints = torch.cat(pred_joints, dim=0)[:, :num_joints, :]
    
    # 恢复原始批次顺序
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map]  # (BS*T) X num_joints X 3 
    
    # 处理顶点
    x_pred_smpl_verts = torch.cat(pred_verts, dim=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map]  # (BS*T) X num_verts X 3 
    
    # 恢复原始的批次和时间步维度
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3)  # BS X T X num_joints X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3)  # BS X T X num_verts X 3 
    
    # 获取网格面片信息
    mesh_faces = pred_body.f 
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
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
        pretrained_path=None
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
        self.decay_steps = 100000  # 每100000步衰减一次
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
        self.data_root_folder = self.opt.data_root_folder 
        self.window = opt.window
        self.use_object_split = self.opt.use_object_split
        self.prep_dataloader(window_size=opt.window)
        self.bm_dict = self.ds.bm_dict 
        self.test_on_train = self.opt.test_sample_res_on_train 
        self.add_hand_processing = self.opt.add_hand_processing  
        self.for_quant_eval = self.opt.for_quant_eval 

        # 如果提供了预训练模型路径，加载模型
        if self.pretrained_path is not None:
            self.load(self.pretrained_path)

    def prep_dataloader(self, window_size):
        # Define dataset
        train_dataset = HandFootManipDataset(train=True, data_root_folder=self.data_root_folder, \
            window=window_size, use_object_splits=self.use_object_split)
        val_dataset = HandFootManipDataset(train=False, data_root_folder=self.data_root_folder, \
            window=window_size, use_object_splits=self.use_object_split)

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
            shuffle=True, pin_memory=False, num_workers=4))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=False, num_workers=4))

    def save(self, milestone):
        """
        保存模型检查点
        """
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.ema_model.state_dict(),
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
        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data).to(data.device) # BS X T X D 
        mask[:, t_idx, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 

    def prep_joint_condition_mask(self, data, joint_idx, pos_only):
        """
        准备关节条件掩码
        
        参数:
            data: BS X T X D 
            joint_idx: 关节索引
            pos_only: 是否只考虑位置
            
        返回:
            mask: BS X T X D，条件部分为0，缺失部分为1
        """
        bs = data.shape[0]  # 获取实际批次大小
        
        # 创建与输入数据相同形状的掩码
        mask = torch.ones_like(data).to(data.device)  # BS X T X D

        # 设置关节位置的掩码
        cond_pos_dim_idx = joint_idx * 3 
        cond_rot_dim_idx = 24 * 3 + joint_idx * 6
        
        # 确保索引在有效范围内
        assert cond_pos_dim_idx + 3 <= data.shape[2], f"Position index {cond_pos_dim_idx + 3} exceeds data dimension {data.shape[2]}"
        
        # 设置位置掩码
        mask[:, :, cond_pos_dim_idx:cond_pos_dim_idx+3] = torch.zeros(bs, data.shape[1], 3).to(data.device)
        
        # 如果需要，设置旋转掩码
        if not pos_only:
            assert cond_rot_dim_idx + 6 <= data.shape[2], f"Rotation index {cond_rot_dim_idx + 6} exceeds data dimension {data.shape[2]}"
            mask[:, :, cond_rot_dim_idx:cond_rot_dim_idx+6] = torch.zeros(bs, data.shape[1], 6).to(data.device)

        return mask

    def train(self):
        """训练函数"""
        init_step = self.step 
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = False 
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
                data = data_dict['motion'].cuda()

                bs = data.shape[0]  # 获取当前批次的实际大小

                # 准备条件掩码
                cond_mask = None 
                for joint_idx in IMU_JOINTS:
                    mask = self.prep_joint_condition_mask(data, joint_idx=joint_idx, pos_only=True)
                    # 确保所有掩码的批次维度相同
                    assert mask.shape[0] == bs, f"mask batch size {mask.shape[0]} != data batch size {bs}"
                    if cond_mask is None:
                        cond_mask = mask
                    else:
                        cond_mask = cond_mask * mask

                # 生成填充掩码
                actual_seq_len = data_dict['seq_len'] + 1
                tmp_mask = torch.arange(self.window+1).expand(bs, self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                padding_mask = tmp_mask[:, None, :].to(data.device)  # BS X 1 X T

                with autocast(enabled = self.amp):    
                    loss_diffusion = self.model(data, cond_mask, padding_mask)
                    loss = loss_diffusion

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    # check gradients
                    parameters = [p for p in self.model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(data.device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
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

            if self.step != 0 and self.step % 10 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)
                    val_data = val_data_dict['motion'].cuda()

                    cond_mask = None 
                    for joint_idx in IMU_JOINTS:
                        mask = self.prep_joint_condition_mask(val_data, joint_idx=joint_idx, pos_only=True)
                        # 确保所有掩码的批次维度相同
                        assert mask.shape[0] == val_data.shape[0], f"mask batch size {mask.shape[0]} != data batch size {val_data.shape[0]}"
                        if cond_mask is None:
                            cond_mask = mask
                        else:
                            cond_mask = cond_mask * mask

                    # Generate padding mask 
                    actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                    tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                    self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                    # BS X max_timesteps
                    padding_mask = tmp_mask[:, None, :].to(val_data.device)

                    # Get validation loss 
                    val_loss_diffusion = self.model(val_data, cond_mask, padding_mask)
                    val_loss = val_loss_diffusion 
                    if self.use_wandb:
                        val_log_dict = {
                            "Validation/Loss/Total Loss": val_loss.item(),
                            "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                        }
                        wandb.log(val_log_dict)

                    milestone = self.step // self.save_and_sample_every
                    bs_for_vis = 1

                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)

                        all_res_list = self.ema.ema_model.sample(val_data, cond_mask, padding_mask)
                        all_res_list = all_res_list[:bs_for_vis]

                        self.gen_vis_res(all_res_list, val_data_dict, self.step, for_quant_eval=True, compare_with_gt=False)

            self.step += 1

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()

    def cond_sample_res(self):
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(weight_path)
        self.ema.ema_model.eval()

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
        
        foot_sliding_jnts_list = []
        gt_foot_sliding_jnts_list = []
        
        contact_precision_list = []
        contact_recall_list = [] 
        contact_acc_list = []
        contact_f1_score_list = [] 

        contact_dist_list = []
        gt_contact_dist_list = []
      
        if self.test_on_train:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=8, shuffle=False,
                num_workers=0, pin_memory=False, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=8, shuffle=False,
                num_workers=0, pin_memory=False, drop_last=False) 

        if self.for_quant_eval:
            num_samples_per_seq = 20
        else:
            num_samples_per_seq = 1
        
        with torch.no_grad():
            for s_idx, val_data_dict in enumerate(test_loader):
                val_data = val_data_dict['motion'].cuda()

                cond_mask = None 

                left_hand_mask = self.prep_joint_condition_mask(val_data, joint_idx=20, pos_only=True)
                righthand_mask = self.prep_joint_condition_mask(val_data, joint_idx=21, pos_only=True)
                head_mask = self.prep_joint_condition_mask(val_data, joint_idx=12, pos_only=True)

                if cond_mask is not None:
                    cond_mask = cond_mask * left_hand_mask * righthand_mask * head_mask
                else:
                    cond_mask = left_hand_mask * righthand_mask * head_mask

                # Generate padding mask 
                actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(val_data.device)

                hand_jpe_per_seq = []
                lhand_jpe_per_seq = []
                rhand_jpe_per_seq = []

                mpvpe_per_seq = []
                mpjpe_per_seq = []
                
                rot_dist_per_seq = []
                trans_err_per_seq = []
                
                gt_foot_sliding_jnts_per_seq = []
                foot_sliding_jnts_per_seq = []
                
                gt_contact_dist_per_seq = []
                contact_dist_per_seq = [] 

                contact_precision_per_seq = []
                contact_recall_per_seq = [] 

                contact_acc_per_seq = [] 
                contact_f1_score_per_seq = [] 

                sampled_all_res_per_seq = [] 

                for sample_idx in range(num_samples_per_seq):
                    all_res_list = self.ema.ema_model.sample(val_data, \
                    cond_mask=cond_mask, padding_mask=padding_mask) # BS X T X D 

                    sampled_all_res_per_seq.append(all_res_list) 

                    vis_tag = str(milestone)+"_sidx_"+str(s_idx)+"_sample_cnt_"+str(sample_idx)
                 
                    if self.test_on_train:
                        vis_tag = vis_tag + "_on_train"

                    num_seq = all_res_list.shape[0]
                    for seq_idx in range(num_seq):
                        curr_vis_tag = vis_tag + "_seq_idx_in_bs_"+str(seq_idx) 
                        pred_human_trans_list, pred_human_rot_list, pred_human_jnts_list, pred_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            self.gen_vis_res(all_res_list[seq_idx:seq_idx+1], val_data_dict, \
                            milestone, vis_tag=curr_vis_tag, for_quant_eval=self.for_quant_eval, selected_seq_idx=seq_idx, compare_with_gt=True)
                        gt_human_trans_list, gt_human_rot_list, gt_human_jnts_list, gt_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            self.gen_vis_res(val_data_dict['motion'].cuda()[seq_idx:seq_idx+1], val_data_dict, \
                            milestone, vis_tag=curr_vis_tag, for_quant_eval=self.for_quant_eval, selected_seq_idx=seq_idx, compare_with_gt=True)
                    
                        lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, \
                                gt_contact_dist, contact_dist, \
                                gt_foot_sliding_jnts, foot_sliding_jnts, contact_precision, contact_recall, \
                                contact_acc, contact_f1_score = \
                                compute_metrics(gt_human_verts_list, pred_human_verts_list, gt_human_jnts_list, pred_human_jnts_list, human_faces_list, \
                                gt_human_trans_list, pred_human_trans_list, gt_human_rot_list, pred_human_rot_list, \
                                obj_verts_list, obj_faces_list, actual_len_list, use_joints24=True)

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

                if self.for_quant_eval:
                    hand_jpe_per_seq = np.asarray(hand_jpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                    lhand_jpe_per_seq = np.asarray(lhand_jpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                    rhand_jpe_per_seq = np.asarray(rhand_jpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                  
                    mpvpe_per_seq = np.asarray(mpvpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                    mpjpe_per_seq = np.asarray(mpjpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                  
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

                    best_sample_idx = mpjpe_per_seq.argmin(axis=0) # sample_num 

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

                    sampled_all_res_per_seq = torch.stack(sampled_all_res_per_seq) # K X BS X T X D 
                    best_sampled_all_res = sampled_all_res_per_seq[best_sample_idx, list(range(num_seq))] # BS X T X D 
                    num_seq = best_sampled_all_res.shape[0]
                    for seq_idx in range(num_seq):
                        pred_human_trans_list, pred_human_rot_list, pred_human_jnts_list, pred_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            self.gen_vis_res(best_sampled_all_res[seq_idx:seq_idx+1], val_data_dict, \
                            milestone, vis_tag=vis_tag, for_quant_eval=True, selected_seq_idx=seq_idx, compare_with_gt=True)
                        gt_human_trans_list, gt_human_rot_list, gt_human_jnts_list, gt_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            self.gen_vis_res(val_data_dict['motion'].cuda()[seq_idx:seq_idx+1], val_data_dict, \
                            milestone, vis_tag=vis_tag, for_quant_eval=True, selected_seq_idx=seq_idx, compare_with_gt=True)

                        obj_scale = val_data_dict['obj_scale'][seq_idx]
                        obj_trans = val_data_dict['obj_trans'][seq_idx]
                        obj_rot_mat = val_data_dict['obj_rot_mat'][seq_idx]
                        actual_len = val_data_dict['seq_len'][seq_idx]
                        object_name = val_data_dict['obj_name'][seq_idx]
                        pred_collision_percent, pred_collision_depth = compute_collision(pred_human_verts_list.cpu(), \
                            human_faces_list, obj_verts_list.cpu(), obj_faces_list, object_name, \
                            obj_scale, obj_rot_mat, obj_trans, actual_len)
                            
                        gt_collision_percent, gt_collision_depth = compute_collision(gt_human_verts_list.cpu(), \
                            human_faces_list, obj_verts_list.cpu(), obj_faces_list, object_name, \
                            obj_scale, obj_rot_mat, obj_trans, actual_len)

                        collision_percent_list.append(pred_collision_percent)
                        collision_depth_list.append(pred_collision_depth)
                        gt_collision_percent_list.append(gt_collision_percent)
                        gt_collision_depth_list.append(gt_collision_depth) 

                    # Get the min error 
                    for tmp_seq_idx in range(num_seq):
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

        if self.for_quant_eval:
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
            print("Left Hand JPE: {0}, Right Hand JPE: {1}, Two Hands JPE: {2}".format(mean_lhand_jpe, mean_rhand_jpe, mean_hand_jpe))
            print("MPJPE: {0}, MPVPE: {1}, Root Trans: {2}, Global Rot Err: {3}".format(mean_mpjpe, mean_mpvpe, mean_root_trans_err, mean_rot_dist))
            print("Foot sliding jnts: {0}, GT Foot sliding jnts: {1}".format(mean_fsliding_jnts, mean_gt_fsliding_jnts))
            print("Collision percent: {0}, Collision depth: {1}".format(mean_collision_percent, mean_collision_depth))
            print("GT Collision percent: {0}, GT Collision depth: {1}".format(gt_mean_collision_percent, gt_mean_collision_depth))
            print("Contact precision: {0}, Contact recall: {1}".format(mean_contact_precision, mean_contact_recall))
            print("Contact Acc: {0}, COntact F1 score: {1}".format(mean_contact_acc, mean_contact_f1_score))
            print("Contact dist: {0}, GT Contact dist: {1}".format(mean_contact_dist, mean_gt_contact_dist))

    def gen_vis_res(self, all_res_list, data_dict, step, vis_tag=None, \
        for_quant_eval=False, selected_seq_idx=None, compare_with_gt=True, res_obj_pos=None, res_obj_rot_mat=None):
        """
        生成可视化结果
        
        参数:
            all_res_list: 预测的人体姿态数据
            data_dict: 数据字典
            step: 当前步数
            vis_tag: 可视化标签
            for_quant_eval: 是否用于定量评估
            selected_seq_idx: 选定的序列索引
            compare_with_gt: 是否与真值进行对比
            res_obj_pos: 预测的物体位置 (N x T x 3)
            res_obj_rot_mat: 预测的物体旋转矩阵 (N x T x 3 x 3)
        """
        # all_res_list: N X T X D 
        num_seq = all_res_list.shape[0]

        num_joints = 24
        
        normalized_global_jpos = all_res_list[:, :, :num_joints*3].reshape(num_seq, -1, num_joints, 3)
        global_jpos = self.ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, num_joints, 3))
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3) # N X T X 24 X 3 
        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3

        global_rot_6d = all_res_list[:, :, -22*6:].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 
        if compare_with_gt and selected_seq_idx is not None:
            normalized_gt_global_jpos = data_dict['motion'].cuda()[selected_seq_idx:selected_seq_idx+1][:, :, :num_joints*3].reshape(num_seq, -1, num_joints, 3)
            gt_global_jpos = self.ds.de_normalize_jpos_min_max(normalized_gt_global_jpos.reshape(-1, num_joints, 3))
            gt_global_jpos = gt_global_jpos.reshape(num_seq, -1, num_joints, 3) # N X T X 22 X 3 
            gt_global_root_jpos = gt_global_jpos[:, :, 0, :].clone() # N X T X 3
            gt_global_rot_6d = data_dict['motion'].cuda()[selected_seq_idx:selected_seq_idx+1][:, :, -22*6:].reshape(num_seq, -1, 22, 6)
            gt_global_rot_mat = transforms.rotation_6d_to_matrix(gt_global_rot_6d) # N X T X 22 X 3 X 3 

        trans2joint = data_dict['trans2joint'].to(all_res_list.device) # N X 3

        seq_len = data_dict['seq_len'].detach().cpu().numpy() # BS 
      
        # Used for quantitative evaluation. 
        human_trans_list = [] 
        human_rot_list = [] 
        human_jnts_list = []
        human_verts_list = []
        human_faces_list = []

        obj_verts_list = []
        obj_faces_list = [] 

        actual_len_list = []

        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
          
            if selected_seq_idx is None:
                curr_trans2joint = trans2joint[idx:idx+1].clone()
            else:
                curr_trans2joint = trans2joint[selected_seq_idx:selected_seq_idx+1].clone()

            root_trans = curr_global_root_jpos + curr_trans2joint # T X 3 
            if compare_with_gt and selected_seq_idx is not None:
                curr_gt_global_root_jpos = gt_global_root_jpos[idx] # T X 3
                curr_gt_global_rot_mat = gt_global_rot_mat[idx] # T X 22 X 3 X 3 
                curr_gt_local_rot_mat = quat_ik_torch(curr_gt_global_rot_mat) # T X 22 X 3 X 3 
                curr_gt_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_gt_local_rot_mat) # T X 22 X 3 
                gt_root_trans = curr_gt_global_root_jpos + curr_trans2joint # T X 3 
            # Generate global joint position 
            bs = 1
            if selected_seq_idx is None:
                betas = data_dict['betas'][idx]
                gender = data_dict['gender'][idx]
                curr_obj_rot_mat = data_dict['obj_rot_mat'][idx]
                curr_obj_trans = data_dict['obj_trans'][idx]
                curr_obj_scale = data_dict['obj_scale'][idx]
                curr_seq_name = data_dict['seq_name'][idx]
                object_name = curr_seq_name.split("_")[1]
            else:
                betas = data_dict['betas'][selected_seq_idx]
                gender = data_dict['gender'][selected_seq_idx]
                curr_obj_rot_mat = data_dict['obj_rot_mat'][selected_seq_idx]
                curr_obj_trans = data_dict['obj_trans'][selected_seq_idx]
                curr_obj_scale = data_dict['obj_scale'][selected_seq_idx]
                curr_seq_name = data_dict['seq_name'][selected_seq_idx]
                object_name = curr_seq_name.split("_")[1]
            
            # Get human verts 
            mesh_jnts, mesh_verts, mesh_faces = \
                run_smplx_model(root_trans[None].cuda(), curr_local_rot_aa_rep[None].cuda(), betas.cuda(), [gender], self.ds.bm_dict, return_joints24=True)

            # Get object verts 
            if object_name in ["mop", "vacuum"]:
                if selected_seq_idx is None:
                    curr_obj_bottom_rot_mat = data_dict['obj_bottom_rot_mat'][idx]
                    curr_obj_bottom_trans = data_dict['obj_bottom_trans'][idx]
                    curr_obj_bottom_scale = data_dict['obj_bottom_scale'][idx]
                else:
                    curr_obj_bottom_rot_mat = data_dict['obj_bottom_rot_mat'][selected_seq_idx]
                    curr_obj_bottom_trans = data_dict['obj_bottom_trans'][selected_seq_idx]
                    curr_obj_bottom_scale = data_dict['obj_bottom_scale'][selected_seq_idx]

                gt_obj_mesh_verts, obj_mesh_faces = self.ds.load_object_geometry(object_name, \
                    curr_obj_scale.detach().cpu().numpy(), curr_obj_trans.detach().cpu().numpy(), \
                    curr_obj_rot_mat.detach().cpu().numpy(), \
                    curr_obj_bottom_scale.detach().cpu().numpy(), \
                    curr_obj_bottom_trans.detach().cpu().numpy(), \
                    curr_obj_bottom_rot_mat.detach().cpu().numpy())

                if res_obj_pos is not None and res_obj_rot_mat is not None:
                    pred_obj_mesh_verts, _ = self.ds.load_object_geometry(object_name, \
                        curr_obj_scale.detach().cpu().numpy(), res_obj_pos.detach().cpu().numpy(), \
                        res_obj_rot_mat.detach().cpu().numpy(), \
                        curr_obj_bottom_scale.detach().cpu().numpy(), \
                        curr_obj_bottom_trans.detach().cpu().numpy(), \
                        curr_obj_bottom_rot_mat.detach().cpu().numpy())
            else:
                gt_obj_mesh_verts, obj_mesh_faces = self.ds.load_object_geometry(object_name, \
                    curr_obj_scale.detach().cpu().numpy(), curr_obj_trans.detach().cpu().numpy(), \
                    curr_obj_rot_mat.detach().cpu().numpy())

                if res_obj_pos is not None and res_obj_rot_mat is not None:
                    pred_obj_mesh_verts, _ = self.ds.load_object_geometry(object_name, \
                        curr_obj_scale.detach().cpu().numpy(), res_obj_pos.detach().cpu().numpy(), \
                        res_obj_rot_mat.detach().cpu().numpy())

            human_trans_list.append(root_trans) 
            human_jnts_list.append(mesh_jnts)
            human_verts_list.append(mesh_verts)
            human_faces_list.append(mesh_faces) 

            human_rot_list.append(curr_global_rot_mat)

            obj_verts_list.append(gt_obj_mesh_verts)
            obj_faces_list.append(obj_mesh_faces) 

            if selected_seq_idx is None:
                actual_len = seq_len[idx]
            else:
                actual_len = seq_len[selected_seq_idx]
            
            actual_len_list.append(actual_len)
            
            if vis_tag is None:
                dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
            else:
                # dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag, str(step))
                dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag)
            if not for_quant_eval:
                if not os.path.exists(dest_mesh_vis_folder):
                    os.makedirs(dest_mesh_vis_folder)

                if compare_with_gt and selected_seq_idx is not None:
                    # 获取真值数据
                    gt_mesh_jnts, gt_mesh_verts, gt_mesh_faces = \
                        run_smplx_model(gt_root_trans[None].cuda(), curr_gt_local_rot_aa_rep[None].cuda(), betas.cuda(), [gender], self.ds.bm_dict, return_joints24=True)
                    
                    # 对预测值应用偏移和颜色
                    pred_verts = mesh_verts.detach().cpu().numpy()[0][:actual_len]
                    # pred_verts[:, 0] += 1.0  # x轴偏移1米
                    pred_color = np.array([0.8, 0.2, 0.2])  # 红色
                    
                    # 对真值应用颜色
                    gt_verts = gt_mesh_verts.detach().cpu().numpy()[0][:actual_len]
                    gt_color = np.array([0.2, 0.2, 0.8])  # 蓝色
                    
                    # 保存到同一个场景
                    mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                    "objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_compare")
                    out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                    "imgs_step_"+str(step)+"_bs_idx_"+str(idx)+"_compare")
                    out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                    "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_compare.mp4")
                    
                    # 保存预测值、真值和物体到同一个场景
                    save_verts_faces_to_mesh_file_w_object_and_gt(
                        pred_verts, mesh_faces.detach().cpu().numpy(),
                        gt_verts, gt_mesh_faces.detach().cpu().numpy(),
                        gt_obj_mesh_verts.detach().cpu().numpy()[:actual_len],
                        obj_mesh_faces,
                        mesh_save_folder,
                        pred_color=pred_color,
                        gt_color=gt_color,
                        pred_obj_verts=pred_obj_mesh_verts.detach().cpu().numpy()[:actual_len] if res_obj_pos is not None else None
                    )
                else:
                    mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                    "objs_step_"+str(step)+"_bs_idx_"+str(idx))
                    out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                    "imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                    out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                    "vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")
                    save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy()[0][:actual_len], 
                        mesh_faces.detach().cpu().numpy(),
                        gt_obj_mesh_verts.detach().cpu().numpy()[:actual_len], 
                        obj_mesh_faces, mesh_save_folder)
                    
                run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, 
                    out_vid_file_path, vis_object=True, vis_gt=compare_with_gt)
         
        human_trans_list = torch.stack(human_trans_list)[0] # T X 3
        human_rot_list = torch.stack(human_rot_list)[0] # T X 22 X 3 X 3 
        human_jnts_list = torch.stack(human_jnts_list)[0, 0] # T X 22 X 3 
        human_verts_list = torch.stack(human_verts_list)[0, 0] # T X Nv X 3 
        human_faces_list = torch.stack(human_faces_list)[0].detach().cpu().numpy() # Nf X 3 

        obj_verts_list = torch.stack(obj_verts_list)[0] # T X Nv' X 3 
        obj_faces_list = np.asarray(obj_faces_list)[0] # Nf X 3

        actual_len_list = np.asarray(actual_len_list)[0] # scalar value 

        return human_trans_list, human_rot_list, human_jnts_list, human_verts_list, human_faces_list,\
        obj_verts_list, obj_faces_list, actual_len_list
    
    def convert_to_data_input(self, joints_data):
        """
        将关节位置数据转换为模型输入格式
        
        参数:
            joints_data: BS X T X N X 3 (batch_size, timesteps, num_joints, xyz)
        """
        bs, T, num_joints, _ = joints_data.shape
        # 初始化完整输入（这里保持与全身模型输入一致）
        data_input = torch.zeros(bs, T, 24*3 + 22*6, device=joints_data.device)

        # 对于每个IMU关节，将 joints_data 中对应的值赋给 data_input 中相应的关节位置部分
        # 注意：这里假设 joints_data 中第 i 个数据对应 IMU_JOINTS[i] 位置（例如第0个数据对应全身模型中索引为 IMU_JOINTS[0] 的关节）
        for i, joint_idx in enumerate(IMU_JOINTS):
            # 赋值到 data_input 中：对于全身模型位置部分，通常是 24 个关节，每个3维，因此目标索引为 joint_idx*3 ~ joint_idx*3+3
            data_input[:, :, joint_idx*3 : joint_idx*3+3] = joints_data[:, :, i]
        
        return data_input

    def gen_fullbody_from_predicted_joints(self, joints_data, val_data_dict):
        """
        从预测的关节数据生成全身姿态
        
        参数:
            joints_data: BS X T X N X 3 (batch_size, timesteps, num_joints, xyz)
            val_data_dict: 验证数据字典
        """
        # bs = joints_data.shape[0]
        # num_steps = joints_data.shape[1] 
        # num_joints = joints_data.shape[2]
        # joints_data = joints_data.reshape(bs, num_steps, num_joints, 3) 
        
        with torch.no_grad():
            val_data = self.convert_to_data_input(joints_data)
          
            # 准备条件掩码
            cond_mask = None 
            for joint_idx in IMU_JOINTS:
                mask = self.prep_joint_condition_mask(val_data, joint_idx=joint_idx, pos_only=True)
                if cond_mask is None:
                    cond_mask = mask
                else:
                    cond_mask = cond_mask * mask

            # 生成填充掩码
            actual_seq_len = val_data_dict['seq_len'] + 1
            tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
            self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
            padding_mask = tmp_mask[:, None, :].to(val_data.device)

            # 生成全身姿态
            all_res_list = self.ema.ema_model.sample(val_data, \
            cond_mask=cond_mask, padding_mask=padding_mask)

        return all_res_list

def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model  
    repr_dim = 24 * 3 + 22 * 6 
    
    loss_type = "l1"
  
    diffusion_model = CondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                batch_size=opt.batch_size)
   
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

    torch.cuda.empty_cache()

def run_sample(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model     
    repr_dim = 24 * 3 + 22 * 6 
   
    loss_type = "l1"
    
    diffusion_model = CondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                batch_size=opt.batch_size)

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
        use_wandb=False 
    )
   
    trainer.cond_sample_res()

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--wandb_pj_name', type=str, default='', help='project name')
    parser.add_argument('--entity', default='wandb_account_name', help='W&B entity')
    parser.add_argument('--exp_name', default='', help='save to project/name')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--fullbody_checkpoint', type=str, default="", help='checkpoint')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results on training dataset 
    parser.add_argument("--test_sample_res_on_train", action="store_true")

    parser.add_argument("--add_hand_processing", action="store_true")

    parser.add_argument("--for_quant_eval", action="store_true")

    parser.add_argument("--use_object_split", action="store_true")

    parser.add_argument('--data_root_folder', default='data', help='root folder for dataset')

    parser.add_argument('--pretrained_path', type=str, default=None, help='path to pretrained model')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    else:
        run_train(opt, device)
