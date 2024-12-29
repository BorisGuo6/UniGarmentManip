from copy import deepcopy
import os
import pickle 
import sys
curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(os.path.join(curpath,'garmentgym'))
sys.path.append(curpath+'/train')
from typing import List
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(os.path.join(curpath,'garmentgym'))
sys.path.append(curpath+'/train')
import torch
from train.model.basic_pn import basic_model
import argparse
from  garmentgym.garmentgym.base.record import task_info
from garmentgym.garmentgym.env.bimanual_fold import BimanualFoldEnv
import open3d as o3d
import torch.nn.functional as F
import pyflex
from typing import List
# from garmentgym.garmentgym.base.config import Task_result
from garmentgym.garmentgym.base.record import cross_Deform_info
import json



def pixel_to_world(pixel_coordinates, depth, camera_intrinsics, camera_extrinsics):
        # 将像素坐标点转换为相机坐标系
        camera_coordinates = np.dot(np.linalg.inv(camera_intrinsics), np.append(pixel_coordinates, 1.0))
        camera_coordinates *= depth

        # 将相机坐标系中的点转换为世界坐标系
        world_point = np.dot(np.linalg.inv(camera_extrinsics), np.append(camera_coordinates, 1.0))
        world_point[2]=-world_point[2]
        return world_point[:3]



def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]



def world_to_pixel(world_point, camera_intrinsics, camera_extrinsics):
        # 将世界坐标点转换为相机坐标系
        #u 是宽
        world_point[2]=-world_point[2]
        camera_point = np.dot(camera_extrinsics, np.append(world_point, 1.0))
        # 将相机坐标点转换为像素坐标系
        pixel_coordinates = np.dot(camera_intrinsics, camera_point[:3])
        pixel_coordinates /= pixel_coordinates[2]
        return pixel_coordinates[:2]

def world_to_pixel_valid(world_point,depth,camera_intrinsics,camera_extrinsics):
    # 将世界坐标点转换为相机坐标系
    #u 是宽
    world_point[2]=-world_point[2]
    camera_point = np.dot(camera_extrinsics, np.append(world_point, 1.0))
    # 将相机坐标点转换为像素坐标系
    pixel_coordinates = np.dot(camera_intrinsics, camera_point[:3])
    pixel_coordinates /= pixel_coordinates[2]


    x, y = pixel_coordinates[:2]
    depth=depth.reshape((depth.shape[0],depth.shape[1]))
    height, width = depth.shape

    # Generate coordinate matrices for all pixels in the depth map
    X, Y = np.meshgrid(np.arange(height), np.arange(width))

    # Calculate Euclidean distances from each pixel to the given coordinate
    distances = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

    # Mask depth map to exclude zero depth values
    nonzero_mask = depth != 0

    # Apply mask to distances and find the minimum distance
    min_distance = np.min(distances[nonzero_mask])

    # Generate a boolean mask for the nearest non-zero depth point
    nearest_mask = (distances == min_distance) & nonzero_mask

    # Get the coordinates of the nearest non-zero depth point
    nearest_coordinate = (X[nearest_mask][0], Y[nearest_mask][0])

    return np.array(nearest_coordinate)

    

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str,default="simple")
    parser.add_argument('--demonstration',type=str,default='./demonstration/trousers/fold/00051')#
    parser.add_argument('--current_cloth',type=str,default='/home/transfer/chenhn_data/cloth3d_eval/uniG/eval')
    parser.add_argument('--model_path',type=str,default='./checkpoint/trousers.pth')
    parser.add_argument('--mesh_id',type=str,default='0000')#4,
    parser.add_argument('--log_file', type=str,default="double_fold_from_flat_simple.pkl")
    parser.add_argument('--store_dir',type=str,default="fold_test")
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--json_id",type=str,default="1")

    args=parser.parse_args()
    demonstration=args.demonstration
    current_cloth=args.current_cloth
    model_path=args.model_path
    mesh_id=args.mesh_id
    store_dir=args.store_dir
    log_file=args.log_file
    device=args.device
    task_name=args.task_name
    # batch_input_path = "/home/transfer/chenhn/UniGarmentManip/task/fold/batch_input_PS.json"
    batch_input_path = "/home/transfer/chenhn/UniGarmentManip/task/fold/batch_input_cloth_eval_data_all_pyflex.json"
    with open(batch_input_path, 'r') as file:
        batch_input = json.load(file)
    current_cloth = batch_input[0]["cloth_root"]
    mesh_id = batch_input[int(args.json_id)]["cloth_name"]
    cloth_type = batch_input[int(args.json_id)]["cloth_type"]
    p = 0
    if cloth_type == "Pants":
        p = 0
        demonstration="./demonstration/trousers/fold/00051"
        model_path="./checkpoint/trousers.pth"
    elif cloth_type == "No-sleeve":
        p = 1
        demonstration="./demonstration/fold/simple_fold2/07483"
        model_path="./checkpoint/tops.pth"
    else:
        p = 0
        demonstration="./demonstration/fold/simple_fold2/07483"
        model_path="./checkpoint/tops.pth"

    print("---------------load model----------------")
    # load model
    model=basic_model(512).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device)["model_state_dict"])
    print('load model from {}'.format(model_path))
    model.eval()

    print("---------------load demonstration sequence----------------")
    # load demonstration sequence
    print(demonstration)
    info_sequence=list()
    ii=0
    for i in sorted(os.listdir(demonstration)):
        if i.endswith('.pkl'):
            with open(os.path.join(demonstration,i),'rb') as f:
                print("load {}".format(i))
                ii+=1
                if ii>p:  
                    data = pickle.load(f)
                    info_sequence.append(data)
                    print(data.action)
    print(info_sequence)
    if demonstration=='./demonstration/trousers/fold/00051':
        info_sequence[1],info_sequence[2] = info_sequence[2],info_sequence[1]
        info_sequence[0].cur_info, info_sequence[1].cur_info = info_sequence[2].cur_info, info_sequence[0].cur_info
    print(info_sequence)
    # info_sequence = [info_sequence[1],info_sequence[2]]
    print("---------------load flat cloth----------------")
    # load flat cloth
    env=BimanualFoldEnv(mesh_category_path=current_cloth,store_path=store_dir,id=mesh_id)
    for j in range(100):
        pyflex.step()
        pyflex.render()
    index = 3
    output_path = os.path.join(current_cloth, mesh_id)
    base_name = "mesh"
    while True:
        folder_name = os.path.join(output_path, f"{base_name}{index}")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
        break
        # index += 1
    particle_pos = pyflex.get_positions().reshape(-1,4)[:env.num_particles,:3]
    output_file = os.path.join(folder_name,"initial_pcd.txt")
    np.savetxt(output_file, particle_pos)
    output_file = os.path.join(folder_name,"faces.txt")
    np.savetxt(output_file, np.array(env.clothes.mesh.faces),fmt='%d')
    init_coverage=env.compute_coverage()
    initial_area_rect = env.calculate_rectangle_ratio(particle_pos)
    # output = []
    # pl=env.check_planeness()
    # output.append(pl)
    # print("check_planeness:",pl)

    action=[]


    for i in range(len(info_sequence)-1):
        print("step:",i)
        demo=info_sequence[i]
        cur_shape:task_info=env.get_cur_info()
        #-------------prepare pc--------------
        cur_pc_points=torch.tensor(cur_shape.cur_info.points).float()
        cur_pc_colors=torch.tensor(cur_shape.cur_info.colors).float()
        cur_pc=torch.cat([cur_pc_points,cur_pc_colors],dim=1)

        demo_pc=torch.tensor(demo.cur_info.points).float()
        demo_colors=torch.tensor(demo.cur_info.colors).float()
        demo_pc=torch.cat([demo_pc,demo_colors],dim=1)

        # down sample to 10000
        if len(demo_pc)>10000:
            demo_pc=demo_pc[torch.randperm(len(demo_pc))[:10000]]
        if len(cur_pc)>10000:
            cur_pc=cur_pc[torch.randperm(len(cur_pc))[:10000]]

        #-------------calculate query point--------------
        cur_action=info_sequence[i+1].action[-1]
        action_function=cur_action[0]
        action_points=cur_action[1]
        action_pcd=[]
        action_id=[]
        # print("ACTION",action_points)
        for point in action_points:
            point_pixel=world_to_pixel_valid(point,demo.cur_info.depth,demo.config.get_camera_matrix()[0],demo.config.get_camera_matrix()[1]).astype(np.int32)
            cam_matrix=demo.config.get_camera_matrix()[0]
            z=demo.cur_info.depth[point_pixel[1],point_pixel[0]]
            x=(point_pixel[0]-cam_matrix[0,2])*z/cam_matrix[0,0]
            y=(point_pixel[1]-cam_matrix[1,2])*z/cam_matrix[1,1]
            point_pcd=np.array([x,y,z])
            action_pcd.append(point_pcd)
            point_pcd=point_pcd.reshape(1,3)
            point_id=np.argmin(np.linalg.norm(demo_pc[:,:3]-point_pcd,axis=1))
            action_id.append(point_id)
        # print("ACTION",action_id)

        #-------------pass network--------------

        
        #通过网络
        demo_pc_ready=deepcopy(demo_pc)
        demo_pc_ready[:,2]=6-2*demo_pc_ready[:,2]
        cur_pc_ready=deepcopy(cur_pc)
        cur_pc_ready[:,2]=6-2*cur_pc_ready[:,2]

        demo_pc_ready=demo_pc_ready.cuda()
        cur_pc_ready=cur_pc_ready.cuda()

        demo_pc_ready=demo_pc_ready.unsqueeze(0)
        cur_pc_ready=cur_pc_ready.unsqueeze(0)
        demo_feature=model(demo_pc_ready)
        cur_feature=model(cur_pc_ready)
        demo_feature=F.normalize(demo_feature,dim=-1)
        cur_feature=F.normalize(cur_feature,dim=-1)
        demo_feature=demo_feature[0]
        cur_feature=cur_feature[0]

        #-------------find correspondence--------------
        cur_pc=cur_pc.numpy()
        action_world=[]
        cur_pcd=[]
        for id in action_id:
            cur_pcd_id=torch.argmax(torch.sum(demo_feature[id]*cur_feature,dim=1,keepdim=True))
            cur_pcd.append(cur_pcd_id)
            cur_matrix=cur_shape.config.get_camera_matrix()[0]
            action_rgbd=np.zeros((2))
            action_rgbd[0]=cur_pc[cur_pcd_id,0]*cur_matrix[0,0]/cur_pc[cur_pcd_id,2]+cur_matrix[0,2]
            action_rgbd[1]=cur_pc[cur_pcd_id,1]*cur_matrix[1,1]/cur_pc[cur_pcd_id,2]+cur_matrix[1,2]
            cur_world=pixel_to_world(action_rgbd,cur_pc[cur_pcd_id,2],cur_shape.config.get_camera_matrix()[0],cur_shape.config.get_camera_matrix()[1])
            action_world.append(cur_world)
        # print("ACTION",action_world)
        #-------------execute action--------------
        env.execute_action([action_function,action_world])
        particle_pos = pyflex.get_positions().reshape(-1,4)[:env.num_particles,:3]
        output_file = os.path.join(folder_name,"step_"+str(i)+"_pcd.txt")
        np.savetxt(output_file, particle_pos)
        # pl=env.check_planeness()
        # output.append(pl)
        # print("check_planeness:",pl)



    

    #-------------check success--------------
    result=env.check_success(type=task_name, init_coverage=init_coverage,initial_area_rect=initial_area_rect)
    particle_pos = pyflex.get_positions().reshape(-1,4)[:env.num_particles,:3]
    output_file = os.path.join(folder_name,"fin_pcd.txt")
    # for i,pl in enumerate(output):
    #     print("step",i,"_planeness:",pl)
    np.savetxt(output_file, particle_pos)
    # output_file = os.path.join(folder_name,"planeness.txt")
    # np.savetxt(output_file, np.array(output))
    # output_file = os.path.join(folder_name,"fin_faces.txt")
    # np.savetxt(output_file, np.array(env.clothes.mesh.faces))

    print("fold result:",result)

        




    

