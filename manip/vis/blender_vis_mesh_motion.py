import os 
import subprocess 
import trimesh 
import imageio 
import numpy as np 

BLENDER_PATH = "blender-3.2.0-linux-x64/blender"
BLENDER_SCRIPTS_FOLDER = "manip/vis"

def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%05d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    # command = [
    #     'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%05d.png', output_vid_file,
    # ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

def images_to_video_w_imageio(img_folder, output_vid_file):
    img_files = os.listdir(img_folder)
    # 确保按数字顺序排序图片
    img_files.sort(key=lambda x: int(x.split('.')[0]))
    im_arr = []
    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        im = imageio.imread(img_path)
        im_arr.append(im)

    im_arr = np.asarray(im_arr)
    imageio.mimwrite(output_vid_file, im_arr, fps=30, quality=8) 

def run_blender_rendering_and_save2video(obj_folder_path, out_folder_path, out_vid_path, \
    scene_blend_path="", \
    vis_object=False, vis_human=True, vis_hand_and_object=False, vis_gt=True, \
    vis_handpose_and_object=False, hand_pose_path=None, mat_color="blue"):
    
    scene_blend_path = os.path.join(BLENDER_SCRIPTS_FOLDER, "floor_colorful_mat.blend")

    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    vid_folder = "/".join(out_vid_path.split("/")[:-1])
    if not os.path.exists(vid_folder):
        os.makedirs(vid_folder)

    if vis_object:
        if vis_human: # vis both human and object
            blender_py_path = os.path.join(BLENDER_SCRIPTS_FOLDER, "blender_vis_utils.py")
            cmd = [BLENDER_PATH, "-P", blender_py_path, "-b", "--", 
                   "--folder", obj_folder_path,
                   "--scene", scene_blend_path,
                   "--out-folder", out_folder_path]
            
            # 如果需要可视化真值，添加相应参数
            if vis_gt:
                cmd.extend(["--vis-gt", "True"])
            
            subprocess.call(" ".join(cmd), shell=True)
        else: # vis object only
            blender_py_path = os.path.join(BLENDER_SCRIPTS_FOLDER, "blender_vis_object_utils.py") 
            subprocess.call(BLENDER_PATH+" -P "+blender_py_path+" -b -- --folder "+obj_folder_path+" --scene "+scene_blend_path+" --out-folder "+out_folder_path, shell=True)  
    else: # Vis human only 
        blender_py_path = os.path.join(BLENDER_SCRIPTS_FOLDER, "blender_vis_human_utils.py") 
        subprocess.call(BLENDER_PATH+" -P "+blender_py_path+" -b -- --folder "+obj_folder_path+" --scene "+\
        scene_blend_path+" --out-folder "+out_folder_path+" --material-color "+mat_color, shell=True)    

    use_ffmpeg = False
    if use_ffmpeg:
        images_to_video(out_folder_path, out_vid_path)
    else:
        images_to_video_w_imageio(out_folder_path, out_vid_path)

def save_verts_faces_to_mesh_file(mesh_verts, mesh_faces, save_mesh_folder, save_gt=False):
    # mesh_verts: T X Nv X 3 
    # mesh_faces: Nf X 3 
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx],
                        faces=mesh_faces)
        if save_gt:
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+"_gt.obj")
        else:
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+".obj")
        mesh.export(curr_mesh_path)

def save_verts_faces_to_mesh_file_w_object(mesh_verts, mesh_faces, obj_verts, obj_faces, save_mesh_folder):
    # mesh_verts: T X Nv X 3 
    # mesh_faces: Nf X 3 
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx],
                        faces=mesh_faces)
        curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+".ply")
        mesh.export(curr_mesh_path)

        obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx],
                        faces=obj_faces)
        curr_obj_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+"_object.ply")
        obj_mesh.export(curr_obj_mesh_path)

def save_verts_faces_to_mesh_file_w_object_and_gt(pred_verts, pred_faces, 
                                                 gt_verts, gt_faces,
                                                 obj_verts, obj_faces, 
                                                 save_mesh_folder,
                                                 pred_color=None,
                                                 gt_color=None,
                                                 pred_obj_verts=None):
    """
    保存预测网格、真值网格和物体网格到指定文件夹中
    
    参数:
        pred_verts: (T, Nv, 3) 预测人体网格顶点
        pred_faces: (Nf, 3) 预测人体网格面片
        gt_verts: (T, Nv, 3) 真值人体网格顶点
        gt_faces: (Nf, 3) 真值人体网格面片
        obj_verts: (T, No, 3) 真实物体网格顶点
        obj_faces: (Nf_obj, 3) 物体网格面片
        save_mesh_folder: 保存文件的目标文件夹
        pred_color: 预测人体的颜色
        gt_color: 真值人体的颜色
        pred_obj_verts: (T, No, 3) 预测物体网格顶点，如果为None则不保存预测物体
    """
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)
    
    num_frames = pred_verts.shape[0]
    for idx in range(num_frames):
        
        
        # 保存真值人体网格（蓝色）
        gt_mesh = trimesh.Trimesh(vertices=gt_verts[idx], faces=gt_faces)
        if gt_color is not None:
            gt_mesh.visual.vertex_colors = [gt_color]*len(gt_verts[idx])
        gt_path = os.path.join(save_mesh_folder, f"{idx:05d}_gt.ply")
        gt_mesh.export(gt_path)
        
        # 保存真实物体网格（紫色）
        obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx], faces=obj_faces)
        if gt_color is not None:
            obj_mesh.visual.vertex_colors = [gt_color]*len(obj_verts[idx])
        obj_path = os.path.join(save_mesh_folder, f"{idx:05d}_object.ply")
        obj_mesh.export(obj_path)

        # 保存预测的人体网格（红色）
        pred_mesh = trimesh.Trimesh(vertices=pred_verts[idx], faces=pred_faces)
        if pred_color is not None:
            pred_mesh.visual.vertex_colors = [pred_color]*len(pred_verts[idx])
        pred_path = os.path.join(save_mesh_folder, f"{idx:05d}_pred.ply")
        pred_mesh.export(pred_path)
        # 如果有预测的物体网格，保存预测物体网格（黄色）
        if pred_obj_verts is not None:
            pred_obj_mesh = trimesh.Trimesh(vertices=pred_obj_verts[idx], faces=obj_faces)
            if pred_color is not None:
                pred_obj_mesh.visual.vertex_colors = [pred_color]*len(pred_obj_verts[idx])
            pred_obj_path = os.path.join(save_mesh_folder, f"{idx:05d}_pred_object.ply")
            pred_obj_mesh.export(pred_obj_path)

