import sys
#sys.path.append("../../")

import os
import numpy as np
import joblib 
import json 
import trimesh 
import time 

import torch
from torch.utils.data import Dataset, DataLoader
from torch import jit

import pytorch3d.transforms as transforms 

from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from bps_torch.tools import sample_uniform_cylinder

from human_body_prior.body_model.body_model import BodyModel

from manip.lafan1.utils import rotate_at_frame_w_obj 

from tqdm import tqdm
import argparse
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

SMPLH_PATH = "./data/smpl_all_models/smplh_amass"

from torch.utils.data._utils.collate import default_collate

IMU_JOINTS = [20, 21, 15, 7, 8, 0]  # 左手、右手、头部、左脚、右脚、髋部
IMU_JOINT_NAMES = ['left_hand', 'right_hand', 'head', 'left_foot', 'right_foot', 'hip']
NUM_IMU_JOINTS = len(IMU_JOINTS)

def custom_collate(batch):
    # 对于字典类型数据，将每个键的元素直接放入列表
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: [d[key] for d in batch] for key in elem}
    return default_collate(batch)

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)

def rotate(points, R):
    shape = list(points.shape)
    points = to_tensor(points)
    R = to_tensor(R)
    if len(shape)>3:
        points = points.squeeze()
    if len(shape)<3:
        points = points.unsqueeze(dim=1)
    if R.shape[0] > shape[0]:
        shape[0] = R.shape[0]
    r_points = torch.matmul(points, R.transpose(1,2))
    return r_points.reshape(shape)

def get_smpl_parents(use_joints24=False):
    bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 

    if use_joints24:
        parents = ori_kintree_table[0, :23] # 23 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list) # 24 
    else:
        parents = ori_kintree_table[0, :22] # 22 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.
    
    return parents

def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3 
    kintree = get_smpl_parents() 

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose # T X J X 3 X 3 

def quat_ik_torch(grot_mat):
    # grot: T X J X 3 X 3 
    parents = get_smpl_parents() 

    grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

    res = torch.cat(
            [
                grot[..., :1, :],
                transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                grot[..., 1:, :]),
            ],
            dim=-2) # T X J X 4 

    res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

    return res_mat 

def quat_fk_torch(lrot_mat, lpos, use_joints24=False):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J/(J+2) X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    if use_joints24:
        parents = get_smpl_parents(use_joints24=True)
    else:
        parents = get_smpl_parents() 

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        if i < lrot.shape[-2]:
            gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res

def merge_two_parts(verts_list, faces_list):
    verts_num = 0
    merged_verts_list = []
    merged_faces_list = []
    for p_idx in range(len(verts_list)):
        # part_verts = torch.from_numpy(verts_list[p_idx]) # T X Nv X 3 
        part_verts = verts_list[p_idx] # T X Nv X 3 
        part_faces = torch.from_numpy(faces_list[p_idx]) # T X Nf X 3 

        if p_idx == 0:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces)
        else:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces+verts_num)

        verts_num += part_verts.shape[1] 

    # merged_verts = torch.cat(merged_verts_list, dim=1).data.cpu().numpy()
    merged_verts = torch.cat(merged_verts_list, dim=1)
    merged_faces = torch.cat(merged_faces_list, dim=0).data.cpu().numpy() 

    return merged_verts, merged_faces 


@jit.script
def batch_global_rotations(local_rot_mat: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    # local_rot_mat: [T, J, 3, 3]
    # parents: [J] (int64)
    T, J, _, _ = local_rot_mat.size()
    global_rot = local_rot_mat.clone()
    for j in range(1, J):
        p = int(parents[j])
        if p >= 0:
            global_rot[:, j, :, :] = torch.matmul(global_rot[:, p, :, :], local_rot_mat[:, j, :, :])
    return global_rot

@jit.script
def batch_forward_kinematics(local_rot_mat: torch.Tensor, rest_jpos: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    # local_rot_mat: [T, J, 3, 3]，每帧各关节的局部旋转矩阵
    # rest_jpos: [T, J, 3]，每帧各关节的 rest pose 位置（经过根位置替换）
    # parents: [J] (int64)，每个关节的父节点索引
    T, J, _ = rest_jpos.size()
    # 先计算全局旋转
    global_rot = batch_global_rotations(local_rot_mat, parents)  # [T, J, 3, 3]
    # 初始化全局关节位置
    global_jpos = torch.empty_like(rest_jpos)
    # 根关节位置直接使用 rest_jpos 的根位置（已由外部赋值）
    global_jpos[:, 0, :] = rest_jpos[:, 0, :]
    # 对于其他关节，利用父关节全局旋转和 rest pose 的偏移进行计算
    for j in range(1, J):
        p = int(parents[j])
        # 对于每一帧，偏移量为当前关节在 rest pose 下与父关节的相对位置
        offset = rest_jpos[:, j, :] - rest_jpos[:, p, :]
        # 计算父关节全局旋转作用于偏移后的结果，并加上父关节位置
        global_jpos[:, j, :] = global_jpos[:, p, :] + torch.matmul(global_rot[:, p, :, :], offset.unsqueeze(-1)).squeeze(-1)
    return global_jpos


class HandFootManipDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        window=120,
        use_object_splits=False,
    ):
        self.train = train
        self.window = window
        self.use_joints24 = True 
        self.use_object_splits = use_object_splits 
        self.train_objects = ["largetable", "woodchair", "plasticbox", "largebox", "smallbox", \
                    "trashcan", "monitor", "floorlamp", "clothesstand", "vacuum"] # 10 objects 
        self.test_objects = ["smalltable", "whitechair", "suitcase", "tripod", "mop"]
        self.parents = torch.tensor(get_smpl_parents(use_joints24=True), dtype=torch.int64).cuda()
        self.data_root_folder = data_root_folder 
        self.obj_geo_root_folder = os.path.join(data_root_folder, "captured_objects")
        self.bps_path = "./manip/data/bps.pt"
        self.window_data_dict = {}

        train_subjects = []
        test_subjects = []
        num_subjects = 17 
        for s_idx in range(1, num_subjects+1):
            if s_idx >= 16:
                test_subjects.append("sub"+str(s_idx))
            else:
                train_subjects.append("sub"+str(s_idx))

        dest_obj_bps_npy_folder = os.path.join(data_root_folder, "object_bps_npy_files_joints24")
        dest_obj_bps_npy_folder_for_test = os.path.join(data_root_folder, "object_bps_npy_files_for_eval_joints24")
        if not os.path.exists(dest_obj_bps_npy_folder):
            os.makedirs(dest_obj_bps_npy_folder)
        if not os.path.exists(dest_obj_bps_npy_folder_for_test):
            os.makedirs(dest_obj_bps_npy_folder_for_test)

        if self.train:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder 
        else:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder_for_test 

        if self.train:   
            self.seq_data_path = os.path.join(data_root_folder, "train_diffusion_manip_seq_joints24.p")  
            self.processed_data_path = os.path.join(data_root_folder, "train_diffusion_manip_window_"+str(self.window)+"_cano_joints24.p")
        else: 
            self.seq_data_path = os.path.join(data_root_folder, "test_diffusion_manip_seq_joints24.p")
            self.processed_data_path = os.path.join(data_root_folder, "test_diffusion_manip_window_"+str(self.window)+"_processed_joints24.p")
        self.data_dict = joblib.load(self.seq_data_path)
        self.indices = list(self.data_dict.keys())

        self.min_max_mean_std_data_path = os.path.join(data_root_folder, "min_max_mean_std_data_window_"+str(self.window)+"_cano_joints24.p")
        self.prep_bps_data()

        # Prepare SMPLX model 
        soma_work_base_dir = os.path.join(self.data_root_folder, 'smpl_all_models')
        support_base_dir = soma_work_base_dir 
        surface_model_type = "smplx"
        # surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "male", 'model.npz')
        # surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "female", 'model.npz')
        surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_MALE.npz")
        surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_FEMALE.npz")
        num_expressions = 0
        num_betas = 16 

        # self.male_bm = BodyModel(bm_path=surface_model_male_fname,
        #                 num_betas=num_betas,
        #                 num_expressions=num_expressions)
        # self.female_bm = BodyModel(bm_path=surface_model_female_fname,
        #                 num_betas=num_betas,
        #                 num_expressions=num_expressions)
        
        self.male_bm = BodyModel(bm_fname=surface_model_male_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        model_type=surface_model_type)
        self.female_bm = BodyModel(bm_fname=surface_model_female_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        model_type=surface_model_type)

        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False 

        self.male_bm = self.male_bm.cuda()
        self.female_bm = self.female_bm.cuda()
        
        self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm}

        # 人体上的imu关节点
        self.imu_joints = IMU_JOINTS
        self.imu_joint_names = IMU_JOINT_NAMES
        self.num_imu_joints = len(self.imu_joints)

    def filter_out_object_split(self):
        # Remove some sequences from window_data_dict such that we have some unseen objects during testing. 
        new_cnt = 0
        new_window_data_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            seq_name = window_data['seq_name']
            object_name = seq_name.split("_")[1]
            if self.train and object_name in self.train_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_window_data_dict[new_cnt]['ori_w_idx'] = k 
                new_cnt += 1

            if (not self.train) and object_name in self.test_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_window_data_dict[new_cnt]['ori_w_idx'] = k 
                new_cnt += 1

        return new_window_data_dict

    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 1.0 
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(1, -1, 3)
            
            bps = {
                'obj': bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            print("Generate new bps data to:{0}".format(self.bps_path))
            torch.save(bps, self.bps_path)
        
        self.bps = torch.load(self.bps_path)

        self.bps_torch = bps_torch()

        self.obj_bps = self.bps['obj']

    def apply_transformation_to_obj_geometry(self, obj_mesh_path, obj_scale, obj_rot, obj_trans):
        t0 = time.time()
        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh_verts = np.asarray(mesh.vertices)  # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces)       # Nf X 3

        ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1)  # T X Nv X 3

        seq_scale = torch.from_numpy(obj_scale).float()  # T
        seq_rot_mat = torch.from_numpy(obj_rot).float()    # T X 3 X 3
        if obj_trans.shape[-1] != 1:
            seq_trans = torch.from_numpy(obj_trans).float()[:, :, None]  # T X 3 X 1
        else:
            seq_trans = torch.from_numpy(obj_trans).float()  # T X 3 X 1

        transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
                                seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2)) + seq_trans
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2)  # T X Nv X 3
        t1 = time.time()
        #print(f"apply_transformation_to_obj_geometry耗时：{t1 - t0:.4f}秒")
        return transformed_obj_verts, obj_mesh_faces

    def load_object_geometry(self, object_name, obj_scale, obj_trans, obj_rot, 
                           obj_bottom_scale=None, obj_bottom_trans=None, obj_bottom_rot=None):
        t0 = time.time()
        obj_mesh_path = os.path.join(self.obj_geo_root_folder, object_name + "_cleaned_simplified.obj")
        if object_name in ["vacuum", "mop"]:
            two_parts = True 
        else:
            two_parts = False 

        if two_parts:
            top_obj_mesh_path = os.path.join(self.obj_geo_root_folder, object_name + "_cleaned_simplified_top.obj")
            bottom_obj_mesh_path = os.path.join(self.obj_geo_root_folder, object_name + "_cleaned_simplified_bottom.obj")
            
            t1 = time.time()
            top_obj_mesh_verts, top_obj_mesh_faces = self.apply_transformation_to_obj_geometry(top_obj_mesh_path, 
                                                                                            obj_scale, obj_rot, obj_trans)
            t2 = time.time()
            bottom_obj_mesh_verts, bottom_obj_mesh_faces = self.apply_transformation_to_obj_geometry(bottom_obj_mesh_path, 
                                                                                                obj_bottom_scale, obj_bottom_rot, obj_bottom_trans)
            t3 = time.time()
            obj_mesh_verts, obj_mesh_faces = merge_two_parts([top_obj_mesh_verts, bottom_obj_mesh_verts], 
                                                            [top_obj_mesh_faces, bottom_obj_mesh_faces])
            t4 = time.time()
            #print(f"load_object_geometry (两部分): top耗时 {t2-t1:.4f}秒, bottom耗时 {t3-t2:.4f}秒, merge耗时 {t4-t3:.4f}秒")
        else:
            obj_mesh_verts, obj_mesh_faces = self.apply_transformation_to_obj_geometry(obj_mesh_path, 
                                                                                        obj_scale, obj_rot, obj_trans)
            t1 = time.time()
            #print(f"load_object_geometry (单部分)耗时：{t1 - t0:.4f}秒")
        return obj_mesh_verts, obj_mesh_faces

    def compute_object_geo_bps(self, obj_verts, obj_trans):
        t0 = time.time()
        bps_object_geo = self.bps_torch.encode(
            x=obj_verts,
            feature_type=['deltas'],
            custom_basis=self.obj_bps.repeat(obj_trans.shape[0], 1, 1) + obj_trans[:, None, :]
        )['deltas']
        t1 = time.time()
        #print(f"compute_object_geo_bps耗时：{t1 - t0:.4f}秒")
        return bps_object_geo

    
    def get_bps_from_window_data_dict(self):
        """
        顺序处理 self.window_data_dict 中每个窗口数据，计算对应的 BPS 表示，并保存结果到目标文件夹中。
        """
        dest_folder = self.dest_obj_bps_npy_folder  # 目标保存目录
        for k, window_data in tqdm(self.window_data_dict.items(), total=len(self.window_data_dict), desc="处理 BPS 数据"):
            try:
                seq_name = window_data['seq_name']
                # 构造保存路径
                dest_path = os.path.join(dest_folder, f"{seq_name}_{str(k)}.npy")
                if not os.path.exists(dest_path):
                    object_name = seq_name.split("_")[1]
                    curr_obj_scale = window_data['obj_scale']
                    new_obj_x = window_data['obj_trans']
                    new_obj_rot_mat = window_data['obj_rot_mat']
                    # 针对 mop/vacuum 进行特殊处理
                    if object_name in ["mop", "vacuum"]:
                        curr_obj_bottom_scale = window_data['obj_bottom_scale']
                        new_obj_bottom_x = window_data['obj_bottom_trans']
                        new_obj_bottom_rot_mat = window_data['obj_bottom_rot_mat']
                        obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale,
                                                                            new_obj_x, new_obj_rot_mat,
                                                                            curr_obj_bottom_scale, new_obj_bottom_x,
                                                                            new_obj_bottom_rot_mat)
                    else:
                        obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale,
                                                                            new_obj_x, new_obj_rot_mat)
                    center_verts = obj_verts.mean(dim=1)  # [T, 3]
                    object_bps = self.compute_object_geo_bps(obj_verts, center_verts)
                    np.save(dest_path, object_bps.data.cpu().numpy())
            except Exception as e:
                print(f"窗口 {k} 处理出错：", e)
    # def process_bps_chunk(self, chunk_items, dest_folder):
    #     """
    #     处理一块窗口数据，chunk_items 是列表，每个元素是 (k, window_data)。
    #     对每个窗口数据，计算 BPS 表示并保存到 dest_folder 下。
    #     """
    #     for k, window_data in chunk_items:
    #         try:
    #             seq_name = window_data['seq_name']
    #             dest_path = os.path.join(dest_folder, f"{seq_name}_{str(k)}.npy")
    #             if not os.path.exists(dest_path):
    #                 object_name = seq_name.split("_")[1]
    #                 curr_obj_scale = window_data['obj_scale']
    #                 new_obj_x = window_data['obj_trans']
    #                 new_obj_rot_mat = window_data['obj_rot_mat']
    #                 if object_name in ["mop", "vacuum"]:
    #                     curr_obj_bottom_scale = window_data['obj_bottom_scale']
    #                     new_obj_bottom_x = window_data['obj_bottom_trans']
    #                     new_obj_bottom_rot_mat = window_data['obj_bottom_rot_mat']
    #                     # 假设 load_object_geometry 已经在模块中定义
    #                     obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale,
    #                                                                     new_obj_x, new_obj_rot_mat,
    #                                                                     curr_obj_bottom_scale, new_obj_bottom_x,
    #                                                                     new_obj_bottom_rot_mat)
    #                 else:
    #                     obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale,
    #                                                                     new_obj_x, new_obj_rot_mat)
    #                 center_verts = obj_verts.mean(dim=1)  # [T, 3]
    #                 object_bps = self.compute_object_geo_bps(obj_verts, center_verts)
    #                 np.save(dest_path, object_bps.data.cpu().numpy())
    #         except Exception as e:
    #             print(f"窗口 {k} 处理出错：", e)
    #     return None
    # def get_bps_from_window_data_dict(self, chunk_count=4):
    #     """
    #     将 self.window_data_dict 拆分为 chunk_count 块，
    #     并并行处理各块窗口数据计算 BPS 表示，同时记录每块的起始键值。
    #     """
    #     dest_folder = self.dest_obj_bps_npy_folder  # 目标保存目录
    #     window_items = list(self.window_data_dict.items())  # [(k, window_data), ...]
    #     total = len(window_items)
    #     # 拆分为 chunk_count 块
    #     chunk_size = (total + chunk_count - 1) // chunk_count
    #     chunks = [window_items[i:i+chunk_size] for i in range(0, total, chunk_size)]
        
    #     from concurrent.futures import ProcessPoolExecutor, as_completed
    #     futures = []
    #     with ProcessPoolExecutor(max_workers=chunk_count) as executor:
    #         for chunk in chunks:
    #             futures.append(executor.submit(self.process_bps_chunk, chunk, dest_folder))
            
    #         # 使用 tqdm 包裹 as_completed 迭代器来显示进度
    #         for future in tqdm(as_completed(futures), total=len(futures), desc="处理 BPS 数据"):
    #             try:
    #                 future.result()  # 等待任务完成
    #             except Exception as e:
    #                 print("分块处理出错：", e)



    def process_window_data_vectorized(self, rest_human_offsets, trans2joint, 
                                   window_root_trans, window_root_orient, window_pose_body, 
                                   window_obj_trans, window_obj_rot, window_obj_scale, window_obj_com_pos,
                                   object_name, window_obj_bottom_trans=None, window_obj_bottom_rot=None, window_obj_bottom_scale=None):
        """
        向量化处理一个窗口的数据，所有输入均为 GPU 上的 Tensor，输出包含处理后的 motion、object 信息等。
        """
        T = window_root_trans.shape[0]
        # 1. 构造人体关节初始位置（局部坐标）: [T, J, 3]
        # rest_human_offsets: [J, 3]，重复 T 次，然后将 root joint 替换为 window_root_trans
        X = rest_human_offsets.unsqueeze(0).repeat(T, 1, 1)
        X[:, 0, :] = window_root_trans  # 用当前窗口的根部位置

        # 2. 构造轴角表示: 将 root_orient 和 pose_body 合并 [T, 22, 3]
        joint_aa_rep = torch.cat((window_root_orient.unsqueeze(1), window_pose_body), dim=1)

        # 3. 批量计算局部旋转矩阵
        local_rot_mat = transforms.axis_angle_to_matrix(joint_aa_rep)  # [T, 22, 3, 3]

        # 4. 利用 JIT 编译的批量前向运动学计算全局关节位置
        global_jpos = batch_forward_kinematics(local_rot_mat, X, self.parents)  # [T, 24, 3]

        # 5. 计算关节速度（简单差分）
        global_jvel = torch.zeros_like(global_jpos)
        global_jvel[1:] = global_jpos[1:] - global_jpos[:-1]    # [T, 24, 3]
        # 6. 批量计算全局旋转矩阵
        global_rot_mat = batch_global_rotations(local_rot_mat, self.parents)  # [T, 22, 3, 3]
        global_rot_6d = transforms.matrix_to_rotation_6d(global_rot_mat)        # [T, 22, 6]
        # 7. 处理 object 数据（这里直接使用传入的 window_obj_trans、window_obj_rot 等）
        # 此处可根据需要添加对 obj_bottom 的处理
        # 计算 object 的中心位置，这里假设已在 window_obj_com_pos 中
        # 8. 组合所有数据构造查询结果
        motion = torch.cat((global_jpos.reshape(T, -1), global_jvel.reshape(T, -1), global_rot_6d.reshape(T, -1)), dim=1)
        print("motion.shape: ", motion.shape)
        query = {
            'motion': motion,
            'obj_trans': window_obj_trans,
            'obj_rot_mat': window_obj_rot,
            'obj_scale': window_obj_scale,
            'obj_com_pos': window_obj_com_pos,
            # 这里简单地令 window_obj_com_pos 为物体中心，实际可根据需要计算
            'window_obj_com_pos': window_obj_com_pos,
            'imu_data': self.generate_imu_data(global_jpos, global_rot_6d, window_obj_trans, window_obj_rot)
        }
        if object_name in ["mop", "vacuum"]:
            query['obj_bottom_trans'] = window_obj_bottom_trans
            query['obj_bottom_rot_mat'] = window_obj_bottom_rot
            query['obj_bottom_scale'] = window_obj_bottom_scale
        return query

    def cal_normalize_data_input(self):
        """
        改写后的版本：先将原始数据转换为 Tensor，再利用向量化操作批量处理每个窗口，
        最后将处理结果存入 self.window_data_dict 中。
        """
        self.window_data_dict = {}
        s_idx = 0
        pbar = tqdm(self.data_dict, desc="处理数据")
        for index in pbar:
            seq_data = self.data_dict[index]
            seq_name = seq_data['seq_name']
            betas = seq_data['betas']
            gender = seq_data['gender']
            # 转换为 Tensor（一次性转换）
            seq_root_trans = torch.from_numpy(seq_data['trans']).float()      # [T, 3]
            seq_root_orient = torch.from_numpy(seq_data['root_orient']).float()  # [T, 3]
            seq_pose_body = torch.from_numpy(seq_data['pose_body']).float().view(-1, 21, 3)  # [T, 21, 3]
            rest_human_offsets = torch.from_numpy(seq_data['rest_offsets']).float()  # [J, 3]
            trans2joint = torch.tensor(seq_data['trans2joint']).float()             # [3]
            obj_trans = torch.from_numpy(seq_data['obj_trans'][:, :, 0]).float()       # [T, 3]
            obj_rot = torch.from_numpy(seq_data['obj_rot']).float()                  # [T, 3, 3]
            obj_scale = torch.from_numpy(seq_data['obj_scale']).float()              # [T]
            obj_com_pos = torch.from_numpy(seq_data['obj_com_pos']).float()          # [T, 3]
            object_name = seq_name.split("_")[1]
            if object_name in ["mop", "vacuum"]:
                obj_bottom_trans = torch.from_numpy(seq_data['obj_bottom_trans'][:, :, 0]).float()  # [T, 3]
                obj_bottom_rot = torch.from_numpy(seq_data['obj_bottom_rot']).float()               # [T, 3, 3]
                obj_bottom_scale = torch.from_numpy(seq_data['obj_bottom_scale']).float()           # [T]
            num_steps = seq_root_trans.shape[0]
            start_indices = list(range(0, num_steps, self.window // 2))
            for start_t_idx in start_indices:
                end_t_idx = min(start_t_idx + self.window - 1, num_steps)
                if end_t_idx - start_t_idx < 30:
                    continue
                # 切出当前窗口数据（所有操作均在 GPU 上批量执行）
                window_root_trans = seq_root_trans[start_t_idx:end_t_idx + 1].cuda()
                window_root_orient = seq_root_orient[start_t_idx:end_t_idx + 1].cuda()
                window_pose_body = seq_pose_body[start_t_idx:end_t_idx + 1].cuda()
                window_obj_trans = obj_trans[start_t_idx:end_t_idx + 1].cuda()
                window_obj_rot = obj_rot[start_t_idx:end_t_idx + 1].cuda()
                window_obj_scale = obj_scale[start_t_idx:end_t_idx + 1].cuda()
                window_obj_com_pos = obj_com_pos[start_t_idx:end_t_idx + 1].cuda()
                if object_name in ["mop", "vacuum"]:
                    window_obj_bottom_trans = obj_bottom_trans[start_t_idx:end_t_idx + 1].cuda()
                    window_obj_bottom_rot = obj_bottom_rot[start_t_idx:end_t_idx + 1].cuda()
                    window_obj_bottom_scale = obj_bottom_scale[start_t_idx:end_t_idx + 1].cuda()
                # 调用向量化版本的 process_window_data 函数
                query = self.process_window_data_vectorized(
                    rest_human_offsets.cuda(),  # 保持在 GPU 上
                    trans2joint.cuda(),
                    window_root_trans,
                    window_root_orient,
                    window_pose_body,
                    window_obj_trans,
                    window_obj_rot,
                    window_obj_scale,
                    window_obj_com_pos,
                    object_name,
                    window_obj_bottom_trans if object_name in ["mop", "vacuum"] else None,
                    window_obj_bottom_rot if object_name in ["mop", "vacuum"] else None,
                    window_obj_bottom_scale if object_name in ["mop", "vacuum"] else None
                )
                # 存入结果（转回 CPU 并转换为 numpy 数组）
                self.window_data_dict[s_idx] = {
                    'motion': query['motion'].detach().cpu().numpy(),
                    'obj_trans': query['obj_trans'].detach().cpu().numpy(),
                    'obj_rot_mat': query['obj_rot_mat'].detach().cpu().numpy(),
                    'obj_scale': query['obj_scale'].detach().cpu().numpy(),
                    'obj_com_pos': query['obj_com_pos'].detach().cpu().numpy(),
                    'window_obj_com_pos': query['window_obj_com_pos'].detach().cpu().numpy(),
                    'seq_name': seq_name,
                    'start_t_idx': start_t_idx,
                    'end_t_idx': end_t_idx,
                    'betas': betas,
                    'gender': gender,
                    'trans2joint': trans2joint.cpu().numpy()
                }
                if object_name in ["mop", "vacuum"]:
                    self.window_data_dict[s_idx]['obj_bottom_trans'] = query['obj_bottom_trans'].detach().cpu().numpy()
                    self.window_data_dict[s_idx]['obj_bottom_rot_mat'] = query['obj_bottom_rot_mat'].detach().cpu().numpy()
                    self.window_data_dict[s_idx]['obj_bottom_scale'] = query['obj_bottom_scale'].detach().cpu().numpy()
                self.window_data_dict[s_idx]['imu_data'] = query['imu_data'].detach().cpu().numpy()
                s_idx += 1
                pbar.set_postfix({'当前序列': seq_name, '已处理窗口': s_idx})
        print(f"数据处理完成! 总共处理了 {s_idx} 个窗口")


    def extract_min_max_mean_std_from_data(self):
        """
        从数据中提取最小值、最大值、均值和标准差
        
        返回:
            stats_dict: 包含统计信息的字典
        """
        all_global_jpos_data = []
        all_global_jvel_data = []

        # 使用tqdm创建进度条
        print("正在处理数据统计信息...")
        for s_idx in tqdm(self.window_data_dict, desc="处理数据窗口"):
            curr_window_data = self.window_data_dict[s_idx]['motion'] # T X D 

            all_global_jpos_data.append(curr_window_data[:, :24*3])
            all_global_jvel_data.append(curr_window_data[:, 24*3:2*24*3])

            start_t_idx = self.window_data_dict[s_idx]['start_t_idx'] 
            end_t_idx = self.window_data_dict[s_idx]['end_t_idx']
            curr_seq_name = self.window_data_dict[s_idx]['seq_name']

        print("正在计算统计值...")
        all_global_jpos_data = np.vstack(all_global_jpos_data).reshape(-1, 72) # (N*T) X 72 
        all_global_jvel_data = np.vstack(all_global_jvel_data).reshape(-1, 72)

        min_jpos = all_global_jpos_data.min(axis=0)
        max_jpos = all_global_jpos_data.max(axis=0)
        min_jvel = all_global_jvel_data.min(axis=0)
        max_jvel = all_global_jvel_data.max(axis=0)

        stats_dict = {}
        stats_dict['global_jpos_min'] = min_jpos 
        stats_dict['global_jpos_max'] = max_jpos 
        stats_dict['global_jvel_min'] = min_jvel 
        stats_dict['global_jvel_max'] = max_jvel  

        print("统计信息计算完成!")
        return stats_dict 


    def generate_imu_data(self, global_jpos, global_rot_6d, obj_trans, obj_rot_mat, freq=1):
        """
        生成人体和物体的IMU数据
        
        参数:
            global_jpos: T X 24 X 3 全局关节位置
            global_rot_6d: T X 22 X 3 X 3 全局关节旋转矩阵
            obj_trans: T X 3 物体位置
            obj_rot_mat: T X 3 X 3 物体旋转矩阵
            
        返回:
            imu_data: T X (N+1) X 6，其中 N=6 是人体IMU数量
                    - 人体IMU: 左手、右手、头部、左脚、右脚、髋部，每个包含3维加速度+3维角速度
                    - 物体IMU: 1个，包含3维加速度+3维角速度
        """
        dt = 1 / freq
        T = global_jpos.shape[0]
        device = global_jpos.device
        
        # 1. 计算人体关节的IMU数据
        # 1.1 计算加速度 (二阶导数)
        human_positions = global_jpos[:, self.imu_joints]  # T X N X 3
        
        # 使用矩阵运算计算速度和加速度
        padded_positions = torch.cat([
            human_positions[[0]], 
            human_positions, 
            human_positions[[-1]]
        ], dim=0)  # (T+2) X N X 3
        
        # 计算速度 (中心差分)
        velocities = (padded_positions[2:] - padded_positions[:-2]) / (2.0 * dt)  # T X N X 3
        
        # 计算加速度
        accelerations = (padded_positions[2:] + padded_positions[:-2] - 2 * padded_positions[1:-1]) / (dt ** 2)  # T X N X 3
        
        # 1.2 计算角速度 (旋转矩阵的差分)
        human_rotations = global_rot_6d[:, self.imu_joints]  # T X N X 3 X 3
        
        # 计算相邻帧之间的相对旋转
        angular_velocities = torch.zeros(T, len(self.imu_joints), 3, device=device)
        # 使用批量矩阵乘法计算相对旋转
        relative_rotations = torch.matmul(
            human_rotations[1:],  # (T-1) X N X 3 X 3
            human_rotations[:-1].transpose(-1, -2)  # (T-1) X N X 3 X 3
        )  # (T-1) X N X 3 X 3
        
        # 将相对旋转转换为轴角表示
        angular_velocities[1:] = transforms.matrix_to_axis_angle(relative_rotations) / dt  # (T-1) X N X 3
        
        # 2. 计算物体的IMU数据
        # 2.1 计算物体加速度
        padded_obj_trans = torch.cat([
            obj_trans[[0]], 
            obj_trans, 
            obj_trans[[-1]]
        ], dim=0)  # (T+2) X 3
        
        obj_velocities = (padded_obj_trans[2:] - padded_obj_trans[:-2]) / (2.0 * dt)  # T X 3
        obj_accelerations = (padded_obj_trans[2:] + padded_obj_trans[:-2] - 2 * padded_obj_trans[1:-1]) / (dt ** 2)  # T X 3
        
        # 2.2 计算物体角速度
        obj_angular_velocities = torch.zeros(T, 3, device=device)
        # 使用批量矩阵乘法计算物体的相对旋转
        obj_relative_rotations = torch.matmul(
            obj_rot_mat[1:],  # (T-1) X 3 X 3
            obj_rot_mat[:-1].transpose(-1, -2)  # (T-1) X 3 X 3
        )  # (T-1) X 3 X 3
        
        # 将相对旋转转换为轴角表示
        obj_angular_velocities[1:] = transforms.matrix_to_axis_angle(obj_relative_rotations) / dt  # (T-1) X 3
        
        # 3. 组合所有IMU数据
        human_imu_data = torch.cat([
            accelerations,  # T X 3 X 3
            angular_velocities  # T X 3 X 3
        ], dim=-1)  # T X 3 X 6
        
        obj_imu_data = torch.cat([
            obj_accelerations.unsqueeze(1),  # T X 1 X 3
            obj_angular_velocities.unsqueeze(1)  # T X 1 X 3
        ], dim=-1)  # T X 1 X 6
        
        imu_data = torch.cat([human_imu_data, obj_imu_data], dim=1)  # T X (N+1) X 6
        
        return imu_data

    def process_window_data(self, rest_human_offsets, trans2joint, seq_root_trans, seq_root_orient, seq_pose_body, \
        obj_trans, obj_rot, obj_scale, obj_com_pos, center_verts, \
        obj_bottom_trans=None, obj_bottom_rot=None, obj_bottom_scale=None):
        random_t_idx = 0 
        # end_t_idx = seq_root_trans.shape[0] - 1

        # window_root_trans = torch.from_numpy(seq_root_trans[random_t_idx:end_t_idx+1]).cuda()
        # window_root_orient = torch.from_numpy(seq_root_orient[random_t_idx:end_t_idx+1]).float().cuda()
        # window_pose_body  = torch.from_numpy(seq_pose_body[random_t_idx:end_t_idx+1]).float().cuda()

        # window_obj_scale = torch.from_numpy(obj_scale[random_t_idx:end_t_idx+1]).float().cuda() # T

        # window_obj_rot_mat = torch.from_numpy(obj_rot[random_t_idx:end_t_idx+1]).float().cuda() # T X 3 X 3 
        # window_obj_trans = torch.from_numpy(obj_trans[random_t_idx:end_t_idx+1]).float().cuda() # T X 3
        # if obj_bottom_trans is not None:
        #     window_obj_bottom_scale = torch.from_numpy(obj_bottom_scale[random_t_idx:end_t_idx+1]).float().cuda() # T

        #     window_obj_bottom_rot_mat = torch.from_numpy(obj_bottom_rot[random_t_idx:end_t_idx+1]).float().cuda() # T X 3 X 3 
        #     window_obj_bottom_trans = torch.from_numpy(obj_bottom_trans[random_t_idx:end_t_idx+1]).float().cuda() # T X 3

        # window_obj_com_pos = torch.from_numpy(obj_com_pos[random_t_idx:end_t_idx+1]).float().cuda() # T X 3
        # window_center_verts = center_verts[random_t_idx:end_t_idx+1].to(window_obj_com_pos.device)

        # move_to_zero_trans = window_root_trans[0:1, :].clone() # 1 X 3 
        # move_to_zero_trans[:, 2] = 0 

        # # Move motion and object translation to make the initial pose trans 0. 
        # window_root_trans = window_root_trans - move_to_zero_trans 
        # window_obj_trans = window_obj_trans - move_to_zero_trans 
        # window_obj_com_pos = window_obj_com_pos - move_to_zero_trans 
        # window_center_verts = window_center_verts - move_to_zero_trans 
        # if obj_bottom_trans is not None:
        #     window_obj_bottom_trans = window_obj_bottom_trans - move_to_zero_trans 

        # window_root_rot_mat = transforms.axis_angle_to_matrix(window_root_orient) # T' X 3 X 3 
        # window_root_quat = transforms.matrix_to_quaternion(window_root_rot_mat)

        # window_pose_rot_mat = transforms.axis_angle_to_matrix(window_pose_body) # T' X 21 X 3 X 3 

        # # Generate global joint rotation 
        # local_joint_rot_mat = torch.cat((window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1) # T' X 22 X 3 X 3 
        # global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 
        # global_joint_rot_quat = transforms.matrix_to_quaternion(global_joint_rot_mat) # T' X 22 X 4 

        # curr_seq_pose_aa = torch.cat((window_root_orient[:, None, :], window_pose_body), dim=1) # T' X 22 X 3/T' X 24 X 3 
        # rest_human_offsets = torch.from_numpy(rest_human_offsets).float()[None] 
        # curr_seq_local_jpos = rest_human_offsets.repeat(curr_seq_pose_aa.shape[0], 1, 1).cuda() # T' X 22 X 3/T' X 24 X 3  
        # curr_seq_local_jpos[:, 0, :] = window_root_trans - torch.from_numpy(trans2joint).cuda()[None] # T' X 22/24 X 3 

        # local_joint_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa)
        # _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos, use_joints24=True)

        # global_jpos = human_jnts # T' X 22/24 X 3 
        # global_jvel = global_jpos[1:] - global_jpos[:-1] # (T'-1) X 22/24 X 3 

        # global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 

        # local_rot_6d = transforms.matrix_to_rotation_6d(local_joint_rot_mat)
        # global_rot_6d = transforms.matrix_to_rotation_6d(global_joint_rot_mat)

        # # 生成IMU数据(包含物体IMU)
        # imu_data = self.generate_imu_data(
        #     global_jpos, 
        #     global_joint_rot_mat,
        #     window_obj_trans,
        #     window_obj_rot_mat
        # )  # T X (N+1) X 6

        # query = {}
        # query['local_rot_mat'] = local_joint_rot_mat # T' X 22 X 3 X 3 
        # query['local_rot_6d'] = local_rot_6d # T' X 22 X 6

        # query['global_jpos'] = global_jpos # T' X 22/24 X 3 
        # query['global_jvel'] = torch.cat((global_jvel, \
        #     torch.zeros(1, global_jvel.shape[1], 3).to(global_jvel.device)), dim=0) # T' X 22/24 X 3 
        
        # query['global_rot_mat'] = global_joint_rot_mat # T' X 22 X 3 X 3 
        # query['global_rot_6d'] = global_rot_6d # T' X 22 X 6

        # query['obj_trans'] = window_obj_trans # T' X 3 
        # query['obj_rot_mat'] = window_obj_rot_mat # T' X 3 X 3 

        # query['obj_scale'] = window_obj_scale # T'

        # query['obj_com_pos'] = window_obj_com_pos # T' X 3 

        # query['window_obj_com_pos'] = window_center_verts # T X 3 

        # if obj_bottom_trans is not None:
        #     query['obj_bottom_trans'] = window_obj_bottom_trans
        #     query['obj_bottom_rot_mat'] = window_obj_bottom_rot_mat 

        #     query['obj_bottom_scale'] = window_obj_bottom_scale # T'

        # # 添加IMU数据到query
        # query['imu_data'] = imu_data  # T X (N+1) X 6

        # return query 

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        s_idx = 0
        temp_window_data_dict = {}
        seq_key = self.indices[index]
        seq_data = self.data_dict[seq_key]
        seq_name = seq_data['seq_name']
        betas = seq_data['betas']
        gender = seq_data['gender']

        # 修改：将数据转换为 Tensor 后保留在 CPU 上（不调用 .cuda()）
        seq_root_trans = torch.from_numpy(seq_data['trans']).float()      # [T, 3]
        seq_root_orient = torch.from_numpy(seq_data['root_orient']).float()  # [T, 3]
        seq_pose_body = torch.from_numpy(seq_data['pose_body']).float().view(-1, 21, 3)  # [T, 21, 3]
        rest_human_offsets = torch.from_numpy(seq_data['rest_offsets']).float()  # [J, 3]
        trans2joint = torch.from_numpy(seq_data['trans2joint']).float()            # [3]
        obj_trans = torch.from_numpy(seq_data['obj_trans'][:, :, 0]).float()       # [T, 3]
        obj_rot = torch.from_numpy(seq_data['obj_rot']).float()                  # [T, 3, 3]
        obj_scale = torch.from_numpy(seq_data['obj_scale']).float()              # [T]
        obj_com_pos = torch.from_numpy(seq_data['obj_com_pos']).float()          # [T, 3]
        object_name = seq_name.split("_")[1]
        if object_name in ["mop", "vacuum"]:
            obj_bottom_trans = torch.from_numpy(seq_data['obj_bottom_trans'][:, :, 0]).float()  # [T, 3]
            obj_bottom_rot = torch.from_numpy(seq_data['obj_bottom_rot']).float()               # [T, 3, 3]
            obj_bottom_scale = torch.from_numpy(seq_data['obj_bottom_scale']).float()           # [T]
        
        num_steps = seq_root_trans.shape[0]
        start_indices = list(range(0, num_steps, self.window // 2))
        for start_t_idx in start_indices:
            end_t_idx = min(start_t_idx + self.window - 1, num_steps)
            if end_t_idx - start_t_idx < 30:
                continue
            # 直接对整个窗口做切片，不调用 .cuda()，保持在 CPU 上
            window_root_trans = seq_root_trans[start_t_idx:end_t_idx + 1]
            window_root_orient = seq_root_orient[start_t_idx:end_t_idx + 1]
            window_pose_body = seq_pose_body[start_t_idx:end_t_idx + 1]
            window_obj_trans = obj_trans[start_t_idx:end_t_idx + 1]
            window_obj_rot = obj_rot[start_t_idx:end_t_idx + 1]
            window_obj_scale = obj_scale[start_t_idx:end_t_idx + 1]
            window_obj_com_pos = obj_com_pos[start_t_idx:end_t_idx + 1]
            if object_name in ["mop", "vacuum"]:
                window_obj_bottom_trans = obj_bottom_trans[start_t_idx:end_t_idx + 1]
                window_obj_bottom_rot = obj_bottom_rot[start_t_idx:end_t_idx + 1]
                window_obj_bottom_scale = obj_bottom_scale[start_t_idx:end_t_idx + 1]
            # 调用向量化版本的 process_window_data_vectorized 函数（确保它支持 CPU Tensor 运算）
            query = self.process_window_data_vectorized(
                rest_human_offsets,  
                trans2joint,
                window_root_trans,
                window_root_orient,
                window_pose_body,
                window_obj_trans,
                window_obj_rot,
                window_obj_scale,
                window_obj_com_pos,
                object_name,
                window_obj_bottom_trans if object_name in ["mop", "vacuum"] else None,
                window_obj_bottom_rot if object_name in ["mop", "vacuum"] else None,
                window_obj_bottom_scale if object_name in ["mop", "vacuum"] else None
            )
            # 存入结果，将 query 中的 GPU 数据转换回 CPU（这里 query 已在 CPU）
            temp_window_data_dict[s_idx] = {
                'motion': query['motion'].detach().cpu().numpy(),
                'obj_trans': query['obj_trans'].detach().cpu().numpy(),
                'obj_rot_mat': query['obj_rot_mat'].detach().cpu().numpy(),
                'obj_scale': query['obj_scale'].detach().cpu().numpy(),
                'obj_com_pos': query['obj_com_pos'].detach().cpu().numpy(),
                'window_obj_com_pos': query['window_obj_com_pos'].detach().cpu().numpy(),
                'seq_name': seq_name,
                'start_t_idx': start_t_idx,
                'end_t_idx': end_t_idx,
                'betas': betas,
                'gender': gender,
                'trans2joint': trans2joint.detach().cpu().numpy()
            }
            if object_name in ["mop", "vacuum"]:
                temp_window_data_dict[s_idx]['obj_bottom_trans'] = query['obj_bottom_trans'].detach().cpu().numpy()
                temp_window_data_dict[s_idx]['obj_bottom_rot_mat'] = query['obj_bottom_rot_mat'].detach().cpu().numpy()
                temp_window_data_dict[s_idx]['obj_bottom_scale'] = query['obj_bottom_scale'].detach().cpu().numpy()
            temp_window_data_dict[s_idx]['imu_data'] = query['imu_data'].detach().cpu().numpy()
            s_idx += 1
        return temp_window_data_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_folder', default='data', help='root folder for dataset')
    parser.add_argument('--window', type=int, default=120, help='window size for sequences')
    parser.add_argument('--use_object_splits', action='store_true', help='whether to use object splits')
    args = parser.parse_args()
    
    print("开始生成预处理数据...")
    
    # 创建训练集
    print("\n处理训练集...")
    train_dataset = HandFootManipDataset(
        train=True,
        data_root_folder=args.data_root_folder,
        window=args.window,
        use_object_splits=args.use_object_splits
    )
    if os.path.exists(train_dataset.processed_data_path):
        train_dataset.window_data_dict = joblib.load(train_dataset.processed_data_path)
        # train_dataset.get_bps_from_window_data_dict()
        
    else:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, collate_fn=custom_collate)
        # 用于存放预处理结果的字典
        final_window_data_dict = {}
        idx_global = 0
        for sample in tqdm(train_loader, desc="预处理数据"):
            # sample 是一个字典，每个键对应的值为 [1, ...]，因为 batch_size=1
            sample = {key: sample[key][0] for key in sample}  # 去掉 batch 维度
            # temp_window_data_dict 中可能有多个窗口，比如键为 0,1,2,...
            for win_idx, win_data in sample.items():
                # 这里假设 win_idx 是窗口索引（整数或字符串），直接将其合并到最终字典中
                final_window_data_dict[idx_global] = win_data
                idx_global += 1
        joblib.dump(final_window_data_dict, train_dataset.processed_data_path)
        # extract min max mean std from data 
        train_dataset.window_data_dict = final_window_data_dict
        min_max_mean_std_jpos_data = train_dataset.extract_min_max_mean_std_from_data()
        joblib.dump(min_max_mean_std_jpos_data, train_dataset.min_max_mean_std_data_path)


    # 创建验证集
    print("\n处理验证集...")
    val_dataset = HandFootManipDataset(
        train=False,
        data_root_folder=args.data_root_folder,
        window=args.window,
        use_object_splits=args.use_object_splits
    )
    if os.path.exists(val_dataset.processed_data_path):
        val_dataset.window_data_dict = joblib.load(val_dataset.processed_data_path)
        # val_dataset.get_bps_from_window_data_dict()
    else:  
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, collate_fn=custom_collate)

        final_window_data_dict = {}
        idx_global = 0
        # 遍历所有样本（每个样本是一个窗口）
        for sample in tqdm(val_loader, desc="预处理数据"):
            # sample 是一个字典，每个键对应的值为 [1, ...]，因为 batch_size=1
            sample = {key: sample[key][0] for key in sample}  # 去掉 batch 维度
            # temp_window_data_dict 中可能有多个窗口，比如键为 0,1,2,...
            for win_idx, win_data in sample.items():
                # 这里假设 win_idx 是窗口索引（整数或字符串），直接将其合并到最终字典中
                final_window_data_dict[idx_global] = win_data
                idx_global += 1
        joblib.dump(final_window_data_dict, val_dataset.processed_data_path)
    print("\n数据预处理完成!")
