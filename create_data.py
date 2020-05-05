import os
import numpy as np
from datetime import datetime

import torch

from pytorch3d.io import save_obj
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes

import utils
import deformation


def create_data(folder_path='meshes/', nb_of_pointclouds=50, nb_of_points=5000, sphere_level=4, normalize_data=True):
    device = torch.device("cuda:0")
    
    data_path = os.path.join(os.getcwd(), folder_path)
    src_mesh = ico_sphere(sphere_level, device)
    
    for filename in os.listdir(data_path):
        print(f"{datetime.now()} Starting:{filename}")
        file_path = os.path.join(data_path, filename)
        cur_mesh = utils.load_mesh(file_path)
        cur_deform_verts = deformation.get_deform_verts(cur_mesh, nb_of_points, sphere_level)
        data_verts = np.expand_dims(cur_deform_verts.detach().cpu().numpy(), axis=0)
        data_input = None
        data_output = None
        for _ in range(nb_of_pointclouds):
            data_a = sample_points_from_meshes(cur_mesh, nb_of_points).squeeze().cpu().numpy()
            if normalize_data:
                data_a = data_a - np.mean(data_a, axis=0)
                data_a = data_a / np.max(data_a, axis=0)
                data_a_sort_indices = np.argsort(np.linalg.norm(data_a, axis=1))
                data_a = data_a[data_a_sort_indices]
            data_a = np.expand_dims(data_a, axis=0)
            data_input = data_a if data_input is None else np.concatenate((data_input, data_a))
            data_output = data_verts if data_output is None else np.concatenate((data_output, data_verts))
        np.save(f'data/{os.path.splitext(filename)[0]}_input.npy', data_input)
        np.save(f'data/{os.path.splitext(filename)[0]}_output.npy', data_output)
        deformed_mesh = src_mesh.offset_verts(cur_deform_verts)
        final_verts, final_faces = deformed_mesh.get_mesh_verts_faces(0)
        final_obj = os.path.join('deformed_meshes/', f'{os.path.splitext(filename)[0]}_deformed.obj')
        save_obj(final_obj, final_verts, final_faces)
        print(f"{datetime.now()} Finished:{filename}, Point Cloud Shape:{data_input.shape} Deform Verts Shape:{data_output.shape}")
