import numpy as np
import json
import os
import math
import argparse

import bpy

if __name__ == "__main__":
    import sys
    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--")+1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment.')
    parser.add_argument('--folder', type=str, metavar='PATH',
                        help='path to specific folder which include folders containing .obj files',
                        default='')
    parser.add_argument('--out-folder', type=str, metavar='PATH',
                        help='path to output folder which include rendered img files',
                        default='')
    parser.add_argument('--scene', type=str, metavar='PATH',
                        help='path to specific .blend path for 3D scene',
                        default='')
    parser.add_argument('--vis-gt', type=str, default="False",
                        help='whether to visualize ground truth')
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    # Load the world
    WORLD_FILE = args.scene
    bpy.ops.wm.open_mainfile(filepath=WORLD_FILE)

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    scene_name = args.scene.split("/")[-1].replace("_scene.blend", "")
    print("scene name:{0}".format(scene_name))
   
    obj_folder = args.folder
    output_dir = args.out_folder
    print("obj_folder:{0}".format(obj_folder))
    print("output dir:{0}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare ply paths 
    ori_obj_files = os.listdir(obj_folder)
    # 确保文件按数字顺序排序
    ori_obj_files.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else -1)
    frame_files = {}
    for tmp_name in ori_obj_files:
        if ".ply" in tmp_name:
            frame_num = int(tmp_name.split('_')[0])
            if frame_num not in frame_files:
                frame_files[frame_num] = []
            frame_files[frame_num].append(tmp_name)

    frame_nums = sorted(frame_files.keys())
    for frame_idx, frame_num in enumerate(frame_nums):
        curr_files = frame_files[frame_num]
        # 确保每帧的文件按特定顺序排序：先物体，再真值，最后预测值
        curr_files.sort(key=lambda x: (
            "0" if "object" in x else
            "1" if "_gt" in x else
            "2" if "_pred" in x else "3"
        ))
        
        # 分别获取预测值、真值、物体和预测物体的文件名
        pred_file = next((f for f in curr_files if "_pred" in f and "object" not in f), None)
        gt_file = next((f for f in curr_files if "_gt" in f and "object" not in f), None)
        obj_file = next((f for f in curr_files if "object" in f and "pred" not in f), None)
        pred_obj_file = next((f for f in curr_files if "pred_object" in f), None)
        
        # 加载预测的人体网格
        if pred_file:
            pred_path = os.path.join(obj_folder, pred_file)
            bpy.ops.import_mesh.ply(filepath=pred_path)
            pred_obj = bpy.data.objects[str(pred_file.replace(".ply", ""))]
            
            # 设置预测模型材质（红色）
            pred_mat = bpy.data.materials.new(name="pred_material")
            pred_obj.data.materials.append(pred_mat)
            pred_mat.use_nodes = True
            pred_bsdf = pred_mat.node_tree.nodes['Principled BSDF']
            if pred_bsdf is not None:
                pred_bsdf.inputs[0].default_value = (0.8, 0.2, 0.2, 1)
            
            # 设置平滑着色
            for f in pred_obj.data.polygons:
                f.use_smooth = True

        # 加载真值的人体网格
        if gt_file and args.vis_gt.lower() == "true":
            gt_path = os.path.join(obj_folder, gt_file)
            bpy.ops.import_mesh.ply(filepath=gt_path)
            gt_obj = bpy.data.objects[str(gt_file.replace(".ply", ""))]
            
            # 设置真值模型材质（蓝色）
            gt_mat = bpy.data.materials.new(name="gt_material")
            gt_obj.data.materials.append(gt_mat)
            gt_mat.use_nodes = True
            gt_bsdf = gt_mat.node_tree.nodes['Principled BSDF']
            if gt_bsdf is not None:
                gt_bsdf.inputs[0].default_value = (0.2, 0.2, 0.8, 1)
            
            # 设置平滑着色
            for f in gt_obj.data.polygons:
                f.use_smooth = True

        # 加载真实物体网格
        if obj_file:
            obj_path = os.path.join(obj_folder, obj_file)
            bpy.ops.import_mesh.ply(filepath=obj_path)
            obj_object = bpy.data.objects[str(obj_file.replace(".ply", ""))]
            
            # 设置真实物体材质（蓝色，与真值人体相同）
            obj_mat = bpy.data.materials.new(name="object_material")
            obj_object.data.materials.append(obj_mat)
            obj_mat.use_nodes = True
            obj_bsdf = obj_mat.node_tree.nodes['Principled BSDF']
            if obj_bsdf is not None:
                obj_bsdf.inputs[0].default_value = (0.2, 0.2, 0.8, 1)
            
            # 设置平滑着色
            for f in obj_object.data.polygons:
                f.use_smooth = True

        # 加载预测的物体网格
        if pred_obj_file:
            pred_obj_path = os.path.join(obj_folder, pred_obj_file)
            bpy.ops.import_mesh.ply(filepath=pred_obj_path)
            pred_obj_object = bpy.data.objects[str(pred_obj_file.replace(".ply", ""))]
            
            # 设置预测物体材质（红色，与预测人体相同）
            pred_obj_mat = bpy.data.materials.new(name="pred_object_material")
            pred_obj_object.data.materials.append(pred_obj_mat)
            pred_obj_mat.use_nodes = True
            pred_obj_bsdf = pred_obj_mat.node_tree.nodes['Principled BSDF']
            if pred_obj_bsdf is not None:
                pred_obj_bsdf.inputs[0].default_value = (0.8, 0.2, 0.2, 1)
            
            # 设置平滑着色
            for f in pred_obj_object.data.polygons:
                f.use_smooth = True

        # 渲染当前帧
        bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, f"{frame_num:05d}.png")
        bpy.ops.render.render(write_still=True)

        # 清理场景中的对象和材质
        if pred_file:
            bpy.data.objects.remove(pred_obj, do_unlink=True)
        if gt_file and args.vis_gt.lower() == "true":
            bpy.data.objects.remove(gt_obj, do_unlink=True)
        if obj_file:
            bpy.data.objects.remove(obj_object, do_unlink=True)
        if pred_obj_file:
            bpy.data.objects.remove(pred_obj_object, do_unlink=True)

        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

    bpy.ops.wm.quit_blender()
