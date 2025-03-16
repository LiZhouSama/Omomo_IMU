# -*- coding: gb2312 -*-
import os
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_sequence_data(seq_data):
    """
    可视化原始序列数据
    
    参数:
        seq_data: 原始序列数据字典
    """
    print("\n=== 序列数据统计 ===")
    print(f"总序列数: {len(seq_data)}")
    
    # 随机选择一个序列进行可视化
    seq_idx = np.random.randint(len(seq_data))
    seq = seq_data[seq_idx]
    
    print(f"\n选择序列 {seq_idx} 的信息:")
    print("\n所有可用的数据字段及其形状:")
    for key in seq.keys():
        if isinstance(seq[key], (np.ndarray, torch.Tensor)):
            print(f"{key}: shape = {seq[key].shape}")
        else:
            print(f"{key}: type = {type(seq[key])}")
    
    # 可视化轨迹
    trans = seq['trans']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trans[:, 0], trans[:, 1], trans[:, 2], 'b-', label='移动轨迹')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"序列 {seq['seq_name']} 的移动轨迹")
    plt.legend()
    plt.show()

def visualize_window_data(window_data):
    """
    可视化处理后的窗口数据
    
    参数:
        window_data: 窗口数据字典
    """
    print("\n=== 窗口数据统计 ===")
    print(f"总窗口数: {len(window_data)}")
    
    # 随机选择一个窗口进行可视化
    window_idx = np.random.randint(len(window_data))
    window = window_data[window_idx]
    
    print(f"\n选择窗口 {window_idx} 的信息:")
    print("\n所有可用的数据字段及其形状:")
    for key in window.keys():
        if isinstance(window[key], (np.ndarray, torch.Tensor)):
            print(f"{key}: shape = {window[key].shape}")
        else:
            print(f"{key}: type = {type(window[key])}")
    
    # 可视化运动数据
    motion = window['motion']
    # 提取关节位置数据 (前72维是24个关节的3D位置)
    joint_positions = motion[:, :72].reshape(-1, 24, 3)
    
    # 绘制第一帧的骨骼姿态
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制关节点
    ax.scatter(joint_positions[0, :, 0], 
              joint_positions[0, :, 1], 
              joint_positions[0, :, 2], 
              c='b', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"窗口 {window_idx} 第一帧的骨骼姿态")
    plt.show()

def print_data_structure(data_dict, name="数据"):
    """
    打印数据字典的完整结构
    
    参数:
        data_dict: 数据字典
        name: 数据名称
    """
    print(f"\n=== {name}结构 ===")
    print(f"总条目数: {len(data_dict)}")
    
    # 随机选择一个条目
    idx = list(data_dict.keys())[0]
    item = data_dict[idx]
    
    print("\n数据字段:")
    for key in item.keys():
        if isinstance(item[key], (np.ndarray, torch.Tensor)):
            print(f"{key}: shape = {item[key].shape}")
        else:
            print(f"{key}: type = {type(item[key])}")

def main():
    # 设置数据路径
    data_root = "./data"  # 根据实际路径修改
    seq_path = os.path.join(data_root, "test_diffusion_manip_seq_joints24.p")
    window_path = os.path.join(data_root, "test_diffusion_manip_window_120_processed_joints24.p")
    
    # 加载数据
    print("正在加载序列数据...")
    seq_data = joblib.load(seq_path)
    print("正在加载窗口数据...")
    window_data = joblib.load(window_path)
    
    # 打印数据结构
    print_data_structure(seq_data, "序列数据")
    print_data_structure(window_data, "窗口数据")
    
    # 可视化数据
    visualize_sequence_data(seq_data)
    visualize_window_data(window_data)
    
    # 打印一些额外的统计信息
    print("\n=== 数据集统计信息 ===")
    
    # 统计不同物体的数量
    seq_objects = set()
    window_objects = set()
    
    for seq in seq_data.values():
        seq_objects.add(seq['seq_name'].split('_')[1])
    
    for window in window_data.values():
        window_objects.add(window['seq_name'].split('_')[1])
    
    print("\n序列数据中的物体类型:")
    print(sorted(list(seq_objects)))
    
    print("\n窗口数据中的物体类型:")
    print(sorted(list(window_objects)))

if __name__ == "__main__":
    window_data_dict = joblib.load('data/train_diffusion_manip_window_120_cano_joints24.p')
    print("序列长度:", len(window_data_dict))
    print("\n第0个序列的所有键值对:")
    for key, value in window_data_dict[0].items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(f"{key}: shape = {value.shape}")
        else:
            print(f"{key}: {value}")