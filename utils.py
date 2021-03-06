import os
import random
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import OpenGLPerspectiveCameras, look_at_view_transform, RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams, PhongShader, PointLights
from pytorch3d.ops import sample_points_from_meshes


def load_mesh(filename):
    device = torch.device("cuda:0")

    trg_obj = os.path.join(filename)
    
    verts, faces, aux = load_obj(trg_obj)

    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    
    return Meshes(verts=[verts], faces=[faces_idx])


def render_obj(verts, faces, distance, elevation, azimuth):
    device = torch.device("cuda:0")
    
    verts_rgb = torch.ones_like(verts)[None]
    textures = Textures(verts_rgb=verts_rgb.to(device))

    cur_mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)

    cameras = OpenGLPerspectiveCameras(device=device)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1, bin_size=0)

    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=PhongShader(device=device, lights=lights))

    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    
    return phong_renderer(meshes_world=cur_mesh, R=R, T=T).cpu().numpy()


def get_model_names(folder_path, randomize=True):
    data_path = os.path.join(os.getcwd(), folder_path)
    model_names = []
    for filename in os.listdir(data_path):
        if "output" not in filename:
            model_names.append(os.path.splitext(filename)[0][:-6])
    if randomize:
        random.shuffle(model_names)
    return model_names


def read_data(model_names, folder_path, points_x=5000, points_y=2562):
    training_x = []
    training_y = []
    for i in range(int(len(model_names) * 0.8)):
        cur_model = model_names[i]
        cur_x = np.load(f"{folder_path}/{cur_model}_input.npy")
        cur_y = np.load(f"{folder_path}/{cur_model}_output.npy")
        training_x.append(cur_x)
        training_y.append(cur_y)
    training_x = np.reshape(np.array(training_x), (-1, points_x, 3))
    training_y = np.reshape(np.array(training_y), (-1, points_y, 3))
    test_x = []
    test_y = []
    for i in range(len(model_names) - int(len(model_names) * 0.8)):
        cur_model = model_names[int(len(model_names) * 0.8)+i]
        cur_x = np.load(f"{folder_path}/{cur_model}_input.npy")
        cur_y = np.load(f"{folder_path}/{cur_model}_output.npy")
        test_x.append(cur_x)
        test_y.append(cur_y)
    test_x = np.reshape(np.array(test_x), (-1, points_x, 3))
    test_y = np.reshape(np.array(test_y), (-1, points_y, 3))
    return training_x, training_y, test_x, test_y


def animate_pointcloud(mesh, anim_file, points_to_sample, restore_anim=True, is_mesh=True):
    if os.path.isfile(anim_file) and restore_anim:
        return anim_file
    frames = []
    for plot_i in range(24):
        if is_mesh:
            points = sample_points_from_meshes(mesh, points_to_sample)
            x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
        else:
            x, y, z = mesh.unbind(1)
        fig = plt.figure(figsize=(5, 5))
        canvas = FigureCanvas(fig)
        ax = Axes3D(fig)
        ax.scatter3D(x, z, -y)
        ax.view_init(elev=190, azim=360*(plot_i/24))
        plt.axis('off')
        plt.close()
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        frames.append(np.frombuffer(s, np.uint8).reshape((height, width, 4)))
    imageio.mimsave(anim_file, frames, 'GIF', fps=8)
    return anim_file
