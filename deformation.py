import torch
from datetime import datetime
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency


def get_deform_verts(target_mesh, points_to_sample=5000, sphere_level=4):
    device = torch.device("cuda:0")
    
    src_mesh = ico_sphere(sphere_level, device)

    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

    learning_rate = 0.01
    num_iter = 500
    w_chamfer = 1.0 
    w_edge = 0.05
    w_normal = 0.0005
    w_laplacian = 0.005
    
    optimizer = torch.optim.Adam([deform_verts], lr=learning_rate, betas=(0.5, 0.999))

    for _ in range(num_iter):
        optimizer.zero_grad()

        new_src_mesh = src_mesh.offset_verts(deform_verts)

        sample_trg = sample_points_from_meshes(target_mesh, points_to_sample)
        sample_src = sample_points_from_meshes(new_src_mesh, points_to_sample)

        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
        loss_edge = mesh_edge_loss(new_src_mesh)
        loss_normal = mesh_normal_consistency(new_src_mesh)
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

        loss.backward()
        optimizer.step()
    print(f"{datetime.now()} Loss Chamfer:{loss_chamfer * w_chamfer}, Loss Edge:{loss_edge * w_edge}, Loss Normal:{loss_normal * w_normal}, Loss Laplacian:{loss_laplacian * w_laplacian}")
        
    return deform_verts
