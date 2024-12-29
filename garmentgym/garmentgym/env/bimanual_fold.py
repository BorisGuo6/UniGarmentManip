import argparse
import pickle
import random
import sys
import time
import os

import cv2
import tqdm
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import torch
curpath=os.getcwd()
sys.path.append(curpath)
sys.path.append(curpath+"/garmentgym")   

import open3d as o3d
from garmentgym.garmentgym.utils.init_env import init_env
import pyflex
from garmentgym.garmentgym.base.clothes_env import ClothesEnv
from garmentgym.garmentgym.base.clothes import Clothes
from copy import deepcopy
from garmentgym.garmentgym.clothes_hyper import hyper
from garmentgym.garmentgym.base.config import *
from garmentgym.garmentgym.utils.exceptions import MoveJointsException
from garmentgym.garmentgym.utils.flex_utils import center_object, wait_until_stable
from multiprocessing import Pool,Process
from garmentgym.garmentgym.utils.translate_utils import pixel_to_world, pixel_to_world_hard, world_to_pixel, world_to_pixel_hard
from garmentgym.garmentgym.utils.basic_utils import make_dir
task_config = {"task_config": {
    'observation_mode': 'cam_rgb',
    'action_mode': 'pickerpickplace',
    'num_picker': 2,
    'render': True,
    'headless': False,
    'horizon': 100,
    'action_repeat': 8,
    'render_mode': 'cloth',
}}

from garmentgym.garmentgym.base.record import task_info

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

def save_surface_visualization(vertices, faces, surface_faces, filename="./surface_faces.png"):
    """
    Save the visualization of the original mesh and the surface faces as an image.
    
    :param vertices: Array of vertices, shape (N, 3)
    :param faces: Array of all faces, shape (M, 3)
    :param surface_faces: Array of surface faces, shape (K, 3)
    :param filename: Name of the file to save the image as (default: "surface_faces.png")
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(131, projection='3d')
# Scatter plot of vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=1, label='Vertices')

    # Plot the entire mesh (optional, for reference)
    all_faces = Poly3DCollection(vertices[faces], alpha=0.1, facecolor='gray', edgecolor='gray')
    ax.add_collection3d(all_faces)

    # Highlight surface faces
    surface = Poly3DCollection(vertices[surface_faces], alpha=1, facecolor='blue')
    ax.add_collection3d(surface)

    
    # Set the view and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface Faces Visualization')
    ax.view_init(elev=30, azim=135)  # Adjust view angle

    # Set equal aspect ratio for better visualization
    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) / 2
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) / 2
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax = fig.add_subplot(132, projection='3d')
# Scatter plot of vertices
    # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=1, label='Vertices')

    # Plot the entire mesh (optional, for reference)
    all_faces = Poly3DCollection(vertices[faces], alpha=0.8, facecolor='gray', edgecolor='gray')
    ax.add_collection3d(all_faces)

    # Highlight surface faces
    surface = Poly3DCollection(vertices[surface_faces], alpha=0.3, facecolor='blue')
    ax.add_collection3d(surface)

    
    # Set the view and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface Faces Visualization')
    ax.view_init(elev=0, azim=90)  # Adjust view angle

    # Set equal aspect ratio for better visualization
    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) / 2
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) / 2
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax = fig.add_subplot(133, projection='3d')
# Scatter plot of vertices
    # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=1, label='Vertices')

    # Plot the entire mesh (optional, for reference)
    all_faces = Poly3DCollection(vertices[faces], alpha=0.8, facecolor='gray', edgecolor='gray')
    ax.add_collection3d(all_faces)

    # Highlight surface faces
    surface = Poly3DCollection(vertices[surface_faces], alpha=0.3, facecolor='blue')
    ax.add_collection3d(surface)

    
    # Set the view and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface Faces Visualization')
    ax.view_init(elev=0, azim=-90)  # Adjust view angle

    # Set equal aspect ratio for better visualization
    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) / 2
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) / 2
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    # Save the image
    # plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save with high resolution
    plt.close()
    print(f"Saved visualization to {filename}")

class BimanualFoldEnv(ClothesEnv):
    def __init__(self,mesh_category_path:str,gui=True,store_path="./",id=-1):
        self.config=Config(task_config)
        self.id=id
        self.clothes=Clothes(name="cloth"+str(id),config=self.config,mesh_category_path=mesh_category_path,id=id)
        super().__init__(mesh_category_path=mesh_category_path,config=self.config,clothes=self.clothes)
        self.store_path=store_path
        self.empty_scene(self.config)
        self.gui=gui
        self.gui=self.config.basic_config.gui
        center_object()
        self.action_tool.reset([0,0.1,0])
        pyflex.step()
        if gui:
            pyflex.render()
        
        self.info=task_info()
        self.action=[]
        self.info.add(config=self.config,clothes=self.clothes)
        self.info.init()
        
        self.grasp_states=[True,True]
        
        self.num_particles = self.clothes.mesh.num_particles    #我瞎改的
        self.particle_radius=0.00625

    def calculate_face_normals_similarity(self,vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)
        # 单位化法向量
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        normals[normals[:, 1] < 0] *= -1
        normal_mean = np.mean(normals, axis=0)
        normal_mean = normal_mean / np.linalg.norm(normal_mean)  # 单位化均值
        # 计算每个法向量与均值的相似度（点积）
        similarities = np.abs(np.dot(normals, normal_mean))
        return np.mean(similarities)

    def check_planeness(self):
        faces=self.clothes.mesh.faces
        cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
        cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
        vertices=np.array(cloth_pos)
        
        mesh = trimesh.Trimesh(vertices, faces, process=True)

        # Define ray origins (above each vertex, high on Z-axis)
        ray_origins = mesh.triangles_center + np.array([0, 1.0, 0])  # Offset above the triangles

        # Define ray directions (straight down)
        ray_directions = np.tile(np.array([0, -1, 0]), (len(ray_origins), 1))

        # Initialize the ray intersector
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

        # Perform ray intersection
        hit_faces = intersector.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            return_locations=True,
        )

        # Calculate hit distances for all intersections
        hit_distances = np.linalg.norm(hit_faces[2] - ray_origins[hit_faces[1]], axis=1)

        # Create an array to store the first face hit per ray
        first_hit_faces = np.full(len(ray_origins), -1, dtype=np.int32)

        # Loop through the rays and find the closest hit for each
        for ray_idx in range(len(ray_origins)):
            # Find all hits for the current ray
            ray_hits = np.where(hit_faces[1] == ray_idx)[0]
            
            if len(ray_hits) > 0:
                # Get distances for the current ray's hits
                ray_hit_distances = hit_distances[ray_hits]
                
                # Find the closest hit
                closest_hit_idx = ray_hits[np.argmin(ray_hit_distances)]
                
                # Store the closest hit face
                first_hit_faces[ray_idx] = hit_faces[0][closest_hit_idx]
        mask=np.unique(first_hit_faces)
        print(len(mask),len(faces))
        save_surface_visualization(vertices,faces,faces[mask])
        return self.calculate_face_normals_similarity(vertices,faces[mask])
    
        
        
    def move_sleeve(self):
        print("move sleeve")
        left_id=self.clothes.top_left
        right_id=self.clothes.top_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.2,0.2)
        next_left_pos[2]+=random.uniform(-0.2,0.4)
        # self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-0.2,0.2)
        next_right_pos[2]+=random.uniform(-0.3,0.3)
        # self.pick_and_place_primitive(cur_right_pos,next_right_pos)
        self.two_pick_and_place_primitive(cur_left_pos,next_left_pos,cur_right_pos,next_right_pos)
    def move_bottom(self):
        print("move bottom")
        left_id=self.clothes.bottom_left
        right_id=self.clothes.bottom_right
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.3,0.3)
        next_left_pos[2]+=random.uniform(-0.3,0.3)
        # self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-0.3,0.3)
        next_right_pos[2]+=random.uniform(-0.3,0.3)
        # self.pick_and_place_primitive(cur_right_pos,next_right_pos)
        self.two_pick_and_place_primitive(cur_left_pos,next_left_pos,cur_right_pos,next_right_pos)
    

    
    def move_middle(self):
        print("move middle")
        middle_id=self.clothes.middle_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_middle_pos=cur_pos[middle_id]
        next_middle_pos=deepcopy(cur_middle_pos)
        next_middle_pos[0]+=random.uniform(-0.5,0.5)
        next_middle_pos[2]+=random.uniform(-0.5,0.5)
        self.two_pick_and_place_primitive(cur_middle_pos,next_middle_pos)
       
    def move_left_right(self):
        print("move left right")
        left_id=self.clothes.left_point
        right_id=self.clothes.right_point
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        cur_left_pos=cur_pos[left_id]
        cur_right_pos=cur_pos[right_id]
        next_left_pos=deepcopy(cur_left_pos)
        next_left_pos[0]+=random.uniform(-0.5,1)
        next_left_pos[2]+=random.uniform(-0.7,0.7)
        # self.pick_and_place_primitive(cur_left_pos,next_left_pos)
        cur_right_pos=deepcopy(cur_right_pos)
        next_right_pos=deepcopy(cur_right_pos)
        next_right_pos[0]+=random.uniform(-1,0.5)
        next_right_pos[2]+=random.uniform(-0.7,0.7)
        # self.pick_and_place_primitive(cur_right_pos,next_right_pos)
        self.two_pick_and_place_primitive(cur_left_pos,next_left_pos,cur_right_pos,next_right_pos)
        
        
    def record_info(self):
        self.info.update(self.action)
        make_dir(os.path.join(self.store_path,str(self.id)))
        self.curr_store_path=os.path.join(self.store_path,str(self.id),str(len(self.action))+".pkl")
        with open(self.curr_store_path,"wb") as f:
            pickle.dump(self.info,f)
    
    def get_cur_info(self):
        self.info.update(self.action)
        return self.info




    def two_pick_change_nodown(
        self, p1, p2,p3 ,p4,p5,p6,lift_height=0.15):
        # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1.copy(), p2.copy(),p3.copy()
        pick_pos2, mid_pos2,place_pos2 = p4.copy(), p5.copy(),p6.copy()
        pick_pos1[1] -= 0.04
        place_pos1[1] += 0.03 + 0.05
        mid_pos1[1] += 0.03 + 0.05
        pick_pos2[1] -= 0.04
        place_pos2[1] += 0.03 + 0.05
        mid_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.two_movep([prepick_pos1,prepick_pos2], speed=8e-2)
        self.two_movep([pick_pos1,pick_pos2], speed=6e-2)
        self.set_grasp(True)
        self.two_movep([prepick_pos1,prepick_pos2], speed=1e-2)
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)
        self.two_movep([preplace_pos1,preplace_pos2], speed=1e-2)
        self.two_movep([place_pos1,place_pos2], speed=2e-2)
        self.set_grasp(False)
        self.two_movep([preplace_pos1,preplace_pos2], speed=8e-2)
        self.two_hide_end_effectors()







    def two_pick_and_place_primitive(self, p1_s, p1_e, p2_s,p2_e,lift_height=0.15,down_height=0.03):
    # prepare primitive params
        pick_pos1, place_pos1 = p1_s.copy(), p1_e.copy()
        pick_pos2, place_pos2 = p2_s.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        place_pos1[1] += 0.2
        pick_pos2[1] += down_height
        place_pos2[1] += 0.2

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, prepick_pos2], speed=8e-2)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=3e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=8e-3)  # 修改此处
        self.two_movep([preplace_pos1, preplace_pos2], speed=8e-3)  # 修改此处
        self.two_movep([place_pos1, place_pos2], speed=8e-3)  # 修改此处
        self.set_grasp([False, False])
        self.two_movep([preplace_pos1, preplace_pos2], speed=8e-2)  # 修改此处
        self.two_hide_end_effectors()
    
    
    
    def two_pick_and_down(self, p1_s,p1_m ,p1_e, p2_s,p2_m,p2_e,lift_height=0.15,down_height=0.03):
    # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1_s.copy(),p1_m.copy(), p1_e.copy()
        pick_pos2, mid_pos2,place_pos2 = p2_s.copy(),p2_m.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        mid_pos1[1]+=down_height+0.04
        place_pos1[1] += 0.03 + 0.05
        pick_pos2[1] += 0.03
        mid_pos2[1]+=0.03
        place_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1]=lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, prepick_pos2], speed=8e-2)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=6e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)  # 修改此处
        self.two_movep([mid_pos1,mid_pos2], speed=1e-2)  # 修改此处
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)  # 修改此处
        
        self.two_movep([preplace_pos1, preplace_pos2], speed=1e-2)  # 修改此处
        self.two_movep([place_pos1, place_pos2], speed=1e-2)  # 修改此处
        self.set_grasp([False, False])
        self.two_movep([preplace_pos1, preplace_pos2], speed=8e-2)  # 修改此处
        self.two_hide_end_effectors()
    
    
    def get_current_covered_area(self,cloth_particle_num, cloth_particle_radius: float = 0.00625):
        """
        Calculate the covered area by taking max x,y cood and min x,y 
        coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        pos = pyflex.get_positions()
        pos = np.reshape(pos, [-1, 4])[:cloth_particle_num]
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = self.vectorized_range1(slotted_x_low, slotted_x_high)
        listy = self.vectorized_range1(slotted_y_low, slotted_y_high)
        listxx, listyy = self.vectorized_meshgrid1(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1
        return np.sum(grid) * span[0] * span[1]

            
    def vectorized_range1(self,start, end):
        """  Return an array of NxD, iterating from the start to the end"""
        N = int(np.max(end - start)) + 1
        idxes = np.floor(np.arange(N) * (end - start)
                        [:, None] / N + start[:, None]).astype('int')
        return idxes

    def vectorized_meshgrid1(self,vec_x, vec_y):
        """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)"""
        N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
        vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
        vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
        return vec_x, vec_y
    
    
    
    def hide_end_effectors(self):
        self.movep([[0.5, 0.5, -1]], speed=5e-2)
        
    def two_hide_end_effectors(self):
        self.set_grasp([False,False])
        self.two_movep([[0.5, 3, -1],[0.5,3,-1]], speed=5e-2)

    def set_grasp(self, grasp):
        if type(grasp) == bool:
            self.grasp_states = [grasp] * len(self.grasp_states)
        elif len(grasp) == len(self.grasp_states):
            self.grasp_states = grasp
        else:
            raise Exception()
             
    
    def step_fn(gui=True):
        pyflex.step()
        if gui:
            pyflex.render()
    def show_position(self):
        self.action_tool.shape_move(np.array([0.9,0,0.9]))
    
    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 2
        target_pos=pos
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            deltas = [(targ - curr)
                      for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and\
                    (min_steps is None or step > min_steps):
                return
            action = []
            for targ, curr, delta, dist, gs in zip(
                    target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr+delta*speed), float(gs)])
            if self.gui:
                pyflex.render()
            action=np.array(action)
            self.action_tool.step(action)
        raise MoveJointsException
    
    def two_pick_change_nodown(
        self, p1, p2,p3 ,p4,p5,p6,lift_height=0.15):
        # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1.copy(), p2.copy(),p3.copy()
        pick_pos2, mid_pos2,place_pos2 = p4.copy(), p5.copy(),p6.copy()
        pick_pos1[1] -= 0.04
        place_pos1[1] += 0.03 + 0.05
        mid_pos1[1] += 0.03 + 0.05
        pick_pos2[1] -= 0.04
        place_pos2[1] += 0.03 + 0.05
        mid_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.two_movep([prepick_pos1,prepick_pos2], speed=8e-2)
        self.two_movep([pick_pos1,pick_pos2], speed=6e-2)
        self.set_grasp(True)
        self.two_movep([prepick_pos1,prepick_pos2], speed=1e-2)
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)
        self.two_movep([preplace_pos1,preplace_pos2], speed=1e-2)
        self.two_movep([place_pos1,place_pos2], speed=2e-2)
        self.set_grasp(False)
        self.two_movep([preplace_pos1,preplace_pos2], speed=8e-2)
        self.two_hide_end_effectors()

    def two_nodown_one_by_one(
        self, p1, p2,p3 ,p4,p5,p6,lift_height=0.15):
        # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1.copy(), p2.copy(),p3.copy()
        pick_pos2, mid_pos2,place_pos2 = p4.copy(), p5.copy(),p6.copy()
        pick_pos1[1] -= 0.04
        place_pos1[1] += 0.03 + 0.05
        mid_pos1[1] += 0.03 + 0.05
        pick_pos2[1] -= 0.04
        place_pos2[1] += 0.03 + 0.05
        mid_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.two_movep([prepick_pos1,prepick_pos2], speed=8e-2)
        self.two_movep([pick_pos1,pick_pos2], speed=1e-2)
        self.set_grasp(True)
        self.two_movep([prepick_pos1,prepick_pos2], speed=1e-2)
        self.two_movep([premid_pos1,prepick_pos2], speed=1e-2)
        self.two_movep([preplace_pos1,premid_pos2], speed=1e-2)
        self.two_movep([place_pos1,preplace_pos2], speed=1e-2)
        self.two_movep([place_pos1,place_pos2], speed=1e-2)
        self.set_grasp(False)
        self.two_movep([preplace_pos1,preplace_pos2], speed=1e-2)
        self.two_hide_end_effectors()
        
    def two_one_by_one(self, p1_s, p1_e, p2_s,p2_e,lift_height=0.15,down_height=0.03):
    # prepare primitive params
        pick_pos1, place_pos1 = p1_s.copy(), p1_e.copy()
        pick_pos2, place_pos2 = p2_s.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        place_pos1[1] += 0.2
        pick_pos2[1] += down_height
        place_pos2[1] += 0.2

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, prepick_pos2], speed=10e-1)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=4e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([preplace_pos1,prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([preplace_pos1,prepick_pos2], speed=1e-2) 
        self.set_grasp([False,True])
        self.two_movep([prepick_pos1,preplace_pos2], speed=1e-2) 
        self.two_movep([prepick_pos1, place_pos2], speed=1e-2)  # 修改此处
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, preplace_pos2], speed=5e-1)  # 修改此处
        self.two_hide_end_effectors()

    
    
    
    def two_pick_and_down(self, p1_s,p1_m ,p1_e, p2_s,p2_m,p2_e,lift_height=0.15,down_height=0.03):
    # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1_s.copy(),p1_m.copy(), p1_e.copy()
        pick_pos2, mid_pos2,place_pos2 = p2_s.copy(),p2_m.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        mid_pos1[1]+=down_height+0.04
        place_pos1[1] += 0.03 + 0.05
        pick_pos2[1] += 0.03
        mid_pos2[1]+=0.03
        place_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1]=lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, prepick_pos2], speed=8e-2)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=6e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)  # 修改此处
        self.two_movep([mid_pos1,mid_pos2], speed=1e-2)  # 修改此处
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)  # 修改此处
        
        self.two_movep([preplace_pos1, preplace_pos2], speed=1e-2)  # 修改此处
        self.two_movep([place_pos1, place_pos2], speed=1e-2)  # 修改此处
        self.set_grasp([False, False])
        self.two_movep([preplace_pos1, preplace_pos2], speed=8e-2)  # 修改此处
        self.two_hide_end_effectors()
    
    # def two_movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 0.08
        target_pos = np.array(pos)
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            deltas = [(targ - curr)
                      for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and\
                    (min_steps is None or step > min_steps):
                return
            action = []
            for targ, curr, delta, dist, gs in zip(
                    target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr+delta*speed), float(gs)])
            action = np.array(action)
            self.action_tool.step(action)


        raise MoveJointsException
    def two_movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 0.08
        target_pos = np.array(pos)
        pick = False
        for gs in self.grasp_states:
            pick = pick or float(gs)
        for step in range(limit):
            curr_pos,pts = self.action_tool._get_pos()
            deltas = [(targ - curr)
                      for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and\
                    (min_steps is None or step > min_steps):
                return
            action = []
            
            for targ, curr, delta, dist, gs in zip(
                    target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr+delta*speed), float(gs)])
                
            action = np.array(action)
            self.action_tool.step(action)
            # if not pick and step > 30:
            #     maxy = np.max(pts[:,1])
            #     # print("HERE",maxy , np.max(curr_pos[:,1]))
            #     if step == 31:
            #         print("HERE",maxy , np.max(curr_pos[:,1]))
            #     if maxy > np.max(curr_pos[:,1]):
            #         assert(0)


        raise MoveJointsException
    
    
    
    
    def hide_end_effectors(self):
        self.movep([[0.5, 0.5, -1]], speed=5e-2)
        
    # def two_hide_end_effectors(self):
    #     self.set_colors([False,False])
    #     self.two_movep([[0.5, 0.5, -1],[0.5,0.5,-1]], speed=5e-2)
    
    def execute_action(self,action):
        function=action[0]
        args=action[1]
        if function=="two_pick_and_place_primitive":
            self.two_pick_and_place_primitive(*args)
        elif function=="two_pick_and_down":
            self.two_pick_and_down(*args)
        elif function=="two_one_by_one":
            self.two_one_by_one(*args)
        elif function=="two_nodown_one_by_one":
            self.two_nodown_one_by_one(*args)
        elif function=="two_pick_change_nodown":
            self.two_pick_change_nodown(*args)
        else:
            raise Exception("No such function")
            
    def wait_until_stable(self,max_steps=300,
                      tolerance=1e-2,
                      gui=False,
                      step_sim_fn=lambda: pyflex.step()):
        for _ in range(max_steps):
            particle_velocity = pyflex.get_velocities()
            if np.abs(particle_velocity).max() < tolerance:
                return True
            step_sim_fn()
            if gui:
                pyflex.render()
        return False
    
    
    def updown(self):
        left_shoulder_id=self.clothes.left_shoulder
        right_shoulder_id=self.clothes.right_shoulder
        cur_pos=np.array(pyflex.get_positions()).reshape(-1,4)[:,:3]
        left_pos=cur_pos[left_shoulder_id]
        right_pos=cur_pos[right_shoulder_id]
        next_left_pos=deepcopy(left_pos)
        next_right_pos=deepcopy(right_pos)
        next_left_pos[1]+=1
        next_right_pos[1]+=1
        #next_left_pos[2]+=random.uniform(0.5,1)
        #next_right_pos[2]+=random.uniform(0.5,1)
        self.two_pick_and_place_primitive(left_pos,next_left_pos,right_pos,next_right_pos,0.8)
    
    
    def compute_coverage(self):
        return self.get_current_covered_area(self.num_particles, self.particle_radius)
    def dbscan(self,points):
        db = DBSCAN(eps=0.3, min_samples=10).fit(torch.from_numpy(points))
        labels = db.labels_
        return points[labels != -1]
    def calculate_rectangle_ratio(self,point_cloud):
        # 将3D点云投影到XY平面
        # point_cloud = self.dbscan(point_cloud)
        points_2d = point_cloud[:, [0,2]]
        
        # 计算点云的凸包
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
        
        # 使用OpenCV计算最小外接矩形
        rect = cv2.minAreaRect(hull_points)
        box = cv2.boxPoints(rect)
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])
        area_rect = width * height
        
        return area_rect
    def check_success(self,type:str,init_coverage,initial_area_rect):
        initial_area=init_coverage
        init_mask=self.clothes.init_cloth_mask
        if type=="funnel":

            rate_boundary=0.7
            shoulder_boundary=0.35
            sleeve_boundary=0.4
            rate_boundary_upper=0.25
            


            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-bottom_left)
            right_sleeve_distance=np.linalg.norm(top_right-bottom_right)
            left_shoulder_distance=np.linalg.norm(bottom_left-left_shoulder)
            right_shoulder_distance=np.linalg.norm(bottom_right-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("left_shoulder_distance=",left_shoulder_distance)
            print("right_shoulder_distance=",right_shoulder_distance)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and left_shoulder_distance<shoulder_boundary and right_shoulder_distance<shoulder_boundary \
            and left_sleeve_distance<sleeve_boundary and right_sleeve_distance<sleeve_boundary:
                return True
            else:
                return False
            
        elif type=="simple":

            rate_boundary=0.5
            shoulder_boundary=0.3
            sleeve_boundary=0.5
            rate_boundary_upper=0.25
            

            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            area_rect = self.calculate_rectangle_ratio(cloth_pos)
            print("initial&fin_area_rect_rate=", area_rect/initial_area_rect)
            print("rect_ratio=",final_area/area_rect)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-right_shoulder)
            right_sleeve_distance=np.linalg.norm(top_right-left_shoulder)
            left_shoulder_distance=np.linalg.norm(bottom_left-left_shoulder)
            right_shoulder_distance=np.linalg.norm(bottom_right-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("left_shoulder_distance=",left_shoulder_distance)
            print("right_shoulder_distance=",right_shoulder_distance)
            
            #sleeve_boundary=np.linalg.norm(top_left-top_right)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and left_shoulder_distance<shoulder_boundary and right_shoulder_distance<shoulder_boundary \
            and left_sleeve_distance<=sleeve_boundary and right_sleeve_distance<=sleeve_boundary:
                return True
            else:
                return False
        
        
        elif type=="left_right":

            rate_boundary=0.7
            shoulder_boundary=0.3
            bottom_boundary=0.3
            sleeve_boundary=0.4
            rate_boundary_upper=0.25
            

            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-bottom_left)
            right_sleeve_distance=np.linalg.norm(top_right-bottom_right)
            bottom_distance=np.linalg.norm(bottom_left-bottom_right)
            shoulder_distance=np.linalg.norm(left_shoulder-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("bottom_distance=",bottom_distance)
            print("shoulder_distance=",shoulder_distance)
            
            sleeve_boundary=np.linalg.norm(left_shoulder-bottom_left)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and shoulder_distance<shoulder_boundary and bottom_distance<bottom_boundary \
            and left_sleeve_distance<sleeve_boundary and right_sleeve_distance<sleeve_boundary:
                return True
            else:
                return False
            
        
        elif type=="jinteng":

            rate_boundary=0.5
            shoulder_boundary=0.3
            sleeve_boundary=0.35
            rate_boundary_upper=0.25
            


            self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            right_shoulder=cloth_pos[self.clothes.right_shoulder][:3].copy()
            left_shoulder=cloth_pos[self.clothes.left_shoulder][:3].copy()
            
            left_sleeve_distance=np.linalg.norm(top_left-bottom_left)
            right_sleeve_distance=np.linalg.norm(top_right-bottom_right)
            left_shoulder_distance=np.linalg.norm(bottom_left-left_shoulder)
            right_shoulder_distance=np.linalg.norm(bottom_right-right_shoulder)
            print("left_sleeve_distance=",left_sleeve_distance)
            print("right_sleeve_distance=",right_sleeve_distance)
            print("left_shoulder_distance=",left_shoulder_distance)
            print("right_shoulder_distance=",right_shoulder_distance)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and left_shoulder_distance<shoulder_boundary and right_shoulder_distance<shoulder_boundary \
            and left_sleeve_distance<sleeve_boundary and right_sleeve_distance<sleeve_boundary:
                return True
            else:
                return False
        
        elif type=='trousers_fold':
            rate_boundary=0.6
            top_boundary=0.6
            bottom_boundary=0.35
            updown_boundary=0.6
            rate_boundary_upper=0.2
            


            #self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            
            
            undown_distance1=np.linalg.norm(top_left-bottom_left)
            updown_distance2=np.linalg.norm(top_right-bottom_right)
            bottom_distance=np.linalg.norm(bottom_left-bottom_right)
            top_distance=np.linalg.norm(top_right-top_left)
            print("undown_distance1=",undown_distance1)
            print("updown_distance2=",updown_distance2)
            print("bottom_distance=",bottom_distance)
            print("top_distance=",top_distance)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and updown_distance2<updown_boundary and undown_distance1<updown_boundary \
            and bottom_distance<bottom_boundary and top_distance<top_boundary:
                return True
            else:
                return False
            
        elif type=='dress_fold':
            rate_boundary=0.6
            top_boundary=0.63
            bottom_boundary=0.4
            updown_boundary=0.63
            rate_boundary_upper=0.2
            


            #self.wait_until_stable()
            
            cur_pos=pyflex.get_positions().reshape(-1,4)[:,:3]
            cloth_pos=cur_pos[:self.clothes.mesh.num_particles]
            cloth_pos=np.array(cloth_pos)
            
            final_area=self.compute_coverage()
            print("final_area=",final_area)
            
            rate=final_area/initial_area
            print("rate=",rate)

            
            bottom_left=cloth_pos[self.clothes.bottom_left][:3].copy()
            bottom_right=cloth_pos[self.clothes.bottom_right][:3].copy()
            top_left=cloth_pos[self.clothes.top_left][:3].copy()
            top_right=cloth_pos[self.clothes.top_right][:3].copy()
            
            
            undown_distance1=np.linalg.norm(top_left-bottom_left)
            updown_distance2=np.linalg.norm(top_right-bottom_right)
            bottom_distance=np.linalg.norm(bottom_left-bottom_right)
            top_distance=np.linalg.norm(top_right-top_left)
            print("undown_distance1=",undown_distance1)
            print("updown_distance2=",updown_distance2)
            print("bottom_distance=",bottom_distance)
            print("top_distance=",top_distance)

            if rate>rate_boundary_upper and rate<rate_boundary \
            and updown_distance2<updown_boundary and undown_distance1<updown_boundary \
            and bottom_distance<bottom_boundary and top_distance<top_boundary:
                return True
            else:
                return False

            if rate>rate_boundary_upper and rate<rate_boundary \
            and left_shoulder_distance<shoulder_boundary and right_shoulder_distance<shoulder_boundary \
            and left_sleeve_distance<sleeve_boundary and right_sleeve_distance<sleeve_boundary:
                return True
            else:
                return False
        

    
    
        
        
    
    
        
    

    
    
    
    