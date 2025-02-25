"""
Module for computing image consistency between two meshes.

It is the product of 2D IoU and cosine similarity between the normal maps of the two meshes,
rendered from multiple viewpoints situated on the vertices of a cuboid.

See MeshUDF, Guillard et al., 2022 ECCV for more details.
"""

import warnings

import numpy as np
import trimesh
import torch
import torch.nn as nn
try:
    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        PointLights,
        AmbientLights,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        HardPhongShader,
        SoftSilhouetteShader,
        TexturesVertex,
    )
    from pytorch3d.structures import Meshes
    renderer_available = True
except ImportError:
    warnings.warn("PyTorch3D not found, image-consistency will not work.")
    renderer_available = False

from .utils import get_device
from .visualization import convert_meshes_pytorch3d


_device = get_device()


def get_projection_torch3D(az, el, distance, focal_length=35, img_w=256, img_h=256, sensor_size_mm = 32., RCAM=False):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    # Calculate intrinsic matrix.
    f_u = focal_length * img_w  / sensor_size_mm
    f_v = focal_length * img_h  / sensor_size_mm
    u_0 = img_w / 2
    v_0 = img_h / 2
    K = np.matrix(((f_u, 0, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    # Edo's convention
    #sa = np.sin(np.radians(az+90))
    #ca = np.cos(np.radians(az+90))
    R_azimuth = np.transpose(np.matrix(((ca, 0, sa),
                                          (0, 1, 0),
                                          (-sa, 0, ca))))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_elevation = np.transpose(np.matrix(((1, 0, 0),
                                          (0, ce, -se),
                                          (0, se, ce))))
    # fix up camera
    se = np.sin(np.radians(90))
    ce = np.cos(np.radians(90))
    if RCAM:
        R_cam = np.transpose(np.matrix(((ce, -se, 0),
                                            (se, ce, 0),
                                            (0, 0, 1))))
    else:
        R_cam = np.transpose(np.matrix(((1, 0, 0),
                                        (0, 1, 0),
                                        (0, 0, 1))))
    T_world2cam = np.transpose(np.matrix((0,
                                           0,
                                           distance)))
    RT = np.hstack((R_cam@R_azimuth@R_elevation, T_world2cam))

    return RT, K



class SoftThreshold(torch.autograd.Function):
    @staticmethod
    #def forward(ctx, input, threshold = 0.55, factor = 10.0):
    #def forward(ctx, input, threshold = 0.85, factor = 10.0):
    #def forward(ctx, input, threshold = 0.45, factor = 10.0):
    #def forward(ctx, input, threshold = 0.15, factor = 10.0):
    #def forward(ctx, input, threshold = 0.45, factor = 10.0):
    #def forward(ctx, input, threshold = -1.5, factor = 10.0):  # very high thresh, was useful for fitting prosketch
    def forward(ctx, input, threshold = 0.45, factor = 10.0):   # == generated data
        with torch.enable_grad():
            output = torch.sigmoid(factor*(input-threshold))
            ctx.save_for_backward(input, output)
        # binary thresholding
        return (input>threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        input.retain_grad()
        output.backward(grad_output, retain_graph=True)
        return input.grad


class Renderer(nn.Module):
    def __init__(self, silhouette_renderer, depth_renderer, image_renderer, max_depth = 5,  modality = "sobel", H=256, W=256, device=_device):
        super().__init__()
        self.silhouette_renderer = silhouette_renderer
        self.depth_renderer = depth_renderer
        self.image_renderer = image_renderer

        self.max_depth = max_depth

        self.threshold = SoftThreshold()

        # sobel filters
        with torch.no_grad():
            # if torch.cuda.is_available():
            #     self.device = torch.device("cuda:0")
            #     torch.cuda.set_device(self.device)
            # else:
            #     self.device = torch.device("cpu")
            self.device = device
            if torch.cuda.is_available() and self.device != torch.device("cpu"):
                torch.cuda.set_device(self.device)

            k_filter = 3
            #print("MODALITY")
            #print(modality)

            if modality == "sobel":
                filter = self.get_sobel_kernel(k_filter)
            elif modality == "scharr":
                filter = self.get_scharr_kernel(k_filter)
            elif modality == "prewitt":
                filter = self.get_prewitt_kernel(k_filter)
            elif modality == "gaussian":
                filter = self.get_gaussian_kernel(k_filter)

            self.filter_x = torch.nn.Conv2d(in_channels=1,
                                            out_channels=1,
                                            kernel_size=k_filter,
                                            padding=0,
                                            bias=False)
            self.filter_x.weight[:] = torch.tensor(filter, requires_grad = False)
            self.filter_x = self.filter_x.to(self.device)

            self.filter_y = torch.nn.Conv2d(in_channels=1,
                                            out_channels=1,
                                            kernel_size=k_filter,
                                            padding=0,
                                            bias=False)
            self.filter_y.weight[:] = torch.tensor(filter.T, requires_grad = False)
            self.filter_y = self.filter_y.to(self.device)
            # Finite difference in double precision, for improved numerical stability
            self.filter_x.double()
            self.filter_y.double()

        # Pixel coordinates
        self.X, self.Y = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
        self.X = (2*(0.5 + self.X.unsqueeze(0).unsqueeze(-1))/H - 1).float().to(device)
        self.Y = (2*(0.5 + self.Y.unsqueeze(0).unsqueeze(-1))/W - 1).float().to(device)

    def get_sobel_kernel(self, k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range, indexing='xy')
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D

    def get_scharr_kernel(self,k=3):
        assert k == 3
        # get range
        scharr_2D = np.array([[-3, 0, 3],[-10, 0, 10], [-3, 0, 3]]).astype(float)/10.0
        return scharr_2D

    def get_prewitt_kernel(self,k=3):
        assert k == 3
        # get range
        prewitt_2D = np.array([[-1, 0, 1],[-1, 0, 1], [-1, 0, 1]]).astype(float)
        return prewitt_2D

    def get_gaussian_kernel(self,k=3):
        assert k == 3
        # get range
        gauss_2D = np.array([[-0.2274, 0, 0.2274],[-0.2840, 0, 0.2840], [-0.2274, 0, 0.2274]]).astype(float)/0.2840
        return gauss_2D

    def depth_2_normal(self, depth, depth_unvalid, cameras, device=_device):
        B, H, W, C = depth.shape

        grad_out = torch.zeros(B, H, W, 3).to(device)
        # Pixel coordinates
        xy_depth = torch.cat([self.X.repeat(B,1,1,1), self.Y.repeat(B,1,1,1), depth], 3).reshape(B,-1, 3)
        xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)

        # compute tangent vectors
        XYZ_camera = xyz_unproj.reshape(B, H, W, 3)
        vx = XYZ_camera[:,1:-1,2:,:]-XYZ_camera[:,1:-1,1:-1,:]
        vy = XYZ_camera[:,2:,1:-1,:]-XYZ_camera[:,1:-1,1:-1,:]

        # finally compute cross product
        normal = torch.cross(vx.reshape(-1, 3),vy.reshape(-1, 3))
        normal_norm = normal.norm(p=2, dim=1, keepdim=True)
        normal_normalized = normal.div(normal_norm)
        # reshape to image
        normal_out = normal_normalized.reshape(B, H-2, W-2, 3)
        grad_out[:,1:-1,1:-1,:] = (0.5 - 0.5*normal_out)

        # zero out +Inf
        grad_out[depth_unvalid] = 0.0

        return grad_out

    def buffer_2_contour(self, buffer):
        # set the steps tensors
        B, C, H, W = buffer.shape
        grad = torch.zeros((B, 1, H, W)).to(self.device)
        padded_buffer = torch.nn.functional.pad(buffer, (1,1,1,1), mode='reflect')
        for c in range(C):
            grad_x = self.filter_x(padded_buffer[:, c:c+1])
            grad_y = self.filter_y(padded_buffer[:, c:c+1])
            grad_tensor = torch.stack((grad_x, grad_y),-1)
            grad_magnitude = torch.norm(grad_tensor, p =2, dim = -1)
            grad = grad + grad_magnitude

        #return 1.0 - self.threshold.apply((torch.clamp(grad,0,1)))
        #return 1.0 - (torch.clamp(grad,0,1))
        return self.threshold.apply(1.0 - (torch.clamp(grad,0,1)))

    def buffer_2_contour_float64(self, buffer):
        # set the steps tensors
        B, C, H, W = buffer.shape
        grad = torch.zeros((B, 1, H, W)).to(self.device).double()
        padded_buffer = torch.nn.functional.pad(buffer, (1,1,1,1), mode='reflect')
        for c in range(C):
            grad_x = self.filter_x(padded_buffer[:, c:c+1])
            grad_y = self.filter_y(padded_buffer[:, c:c+1])
            grad_tensor = torch.stack((grad_x, grad_y),-1)
            grad_magnitude = torch.norm(grad_tensor, p =2, dim = -1)
            grad = grad + grad_magnitude

        return self.threshold.apply(1.0 - (torch.clamp(grad,0,1)))
        #return self.threshold.apply(1.0 - grad)


    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        # take care of soft silhouette
        silhouette_ref = self.silhouette_renderer(meshes_world=meshes_world, **kwargs)
        silhouette_out = silhouette_ref[..., 3]

        # now get depth out
        depth_ref = self.depth_renderer(meshes_world=meshes_world, **kwargs)
        depth_ref = depth_ref.zbuf[...,0].unsqueeze(-1)
        depth_unvalid = depth_ref<0
        depth_ref[depth_unvalid] = self.max_depth
        depth_out = depth_ref[..., 0]

        # post process depth to get normals, contours
        normals_out = self.depth_2_normal(depth_ref, depth_unvalid.squeeze(-1), kwargs['cameras']).permute(0,3,1,2)
        #contours_out = self.buffer_2_contour(torch.cat((normals_out, depth_ref.permute(0,3,1,2)), 1))
        contours_out = self.buffer_2_contour_float64(
                                torch.cat(( normals_out,
                                            depth_ref.permute(0,3,1,2))
                                    , 1).double()
                                ).float()

        # finally textured image for visualization
        image_ref = self.image_renderer(meshes_world=meshes_world, **kwargs)
        image_out = image_ref[..., 0:3].permute(0,3,1,2)

        return image_out, contours_out, normals_out, depth_out, silhouette_out



def get_renderer_cameras_lights(R_cuda, t_cuda, device, image_size, fov=60):
  lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
#   lights = AmbientLights(device=device, ambient_color=((1., 1., 1.),))
  cameras = FoVPerspectiveCameras(device=device, znear=0.001, zfar=3500, aspect_ratio=1.0, fov=fov, R=R_cuda, T=t_cuda)
  sigma = 1e-5
  raster_settings = RasterizationSettings(
      image_size=image_size,
      #blur_radius=0.,
      blur_radius=0.000001, # Avoids bumps in the depth map > grainy sketches
      faces_per_pixel=1,
  )
  raster_settings_soft = RasterizationSettings(
      image_size=image_size,
      blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
      #blur_radius=0.,
      faces_per_pixel=25,
      #faces_per_pixel=1
  )
  # silhouette renderer
  silhouette_renderer = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=cameras,
          raster_settings=raster_settings_soft
      ),
      shader=SoftSilhouetteShader()
  )
  # depth renderer
  depth_renderer = MeshRasterizer(
      cameras=cameras,
      raster_settings=raster_settings
  )
  # image renderer
  image_renderer = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=cameras,
          raster_settings=raster_settings
      ),
      shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
  )
  # assemble in single rendering function
  renderer_pytorch3D = Renderer(silhouette_renderer, depth_renderer, image_renderer, H=image_size, W=image_size)
  return cameras, renderer_pytorch3D, lights


def render_normalmap(mesh_torch, renderer, camera, light, device=_device):
    verts_shape = mesh_torch.verts_packed().shape
    sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, requires_grad=False).to(device)
    mesh_torch.textures = TexturesVertex(verts_features=sphere_verts_rgb)
    gt_image, _, gt_normal, _, _ =\
            renderer(mesh_torch, cameras=camera, lights=light)
    return (gt_normal[0].cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8), gt_image


def compute_image_consistency(n1, n2):
  # Erode both masks and take intersection
  n1_mask = np.sum(np.abs(n1), axis = 2) > 0
  n2_mask = np.sum(np.abs(n2), axis = 2) > 0
  inner_mask = (n1_mask * n2_mask).astype(bool)
  # Compute iou
  iou = np.sum((n1_mask * n2_mask) > 0) / np.sum((n1_mask + n2_mask) > 0)
  # Compute dot product
  n1_vecs = 2*n1[inner_mask]-1
  n1_vecs = n1_vecs / np.linalg.norm(n1_vecs, axis=1)[:,None]
  n2_vecs = 2*n2[inner_mask]-1
  n2_vecs = n2_vecs / np.linalg.norm(n2_vecs, axis=1)[:,None]
  normal_consistency = np.mean(np.sum(n1_vecs*n2_vecs, 1))
  # Return the product
  return iou * normal_consistency


if renderer_available:

    def _convert_mesh(mesh):
        if isinstance(mesh, trimesh.Trimesh):
            if not mesh.is_empty:
                return convert_meshes_pytorch3d([mesh])
            else:
                return Meshes([], [])
        # Hopefully, it is already a Pytorch3D mesh
        return mesh
    
    def get_normalmap(mesh, EL, AZ, DISTANCE=2., IMG_SIZE=256, device=_device):
        """Get the normalmap of a mesh from a specific viewpoint."""
        # Convert Trimesh to Pytorch3D mesh
        mesh_torch = _convert_mesh(mesh)
        # Create camera matrix
        RT, _ = get_projection_torch3D(-AZ -90, EL, DISTANCE, img_w=IMG_SIZE, img_h=IMG_SIZE, focal_length=50)
        RT = torch.tensor(RT).float().to(device).unsqueeze(0)
        R_cuda = RT[:, 0:3, 0:3].float().to(device).clone().detach()
        t_cuda = RT[:, 0:3, 3].float().to(device).clone().detach()
        # Create the renderer
        camera, renderer_pytorch3D, light = get_renderer_cameras_lights(R_cuda, t_cuda, device, IMG_SIZE)
        # Render the normal map
        normal, _ = render_normalmap(mesh_torch, renderer_pytorch3D, camera, light)
        return normal
    
    def image_consistency(gt_mesh_torch, pred_mesh_torch, DISTANCE=2., IMG_SIZE=256,
                          ELs=[-15, 15], AZs=[45, 135, 225, 315], device=_device):
        # Convert Trimesh to Pytorch3D mesh
        gt_mesh_torch = _convert_mesh(gt_mesh_torch)
        pred_mesh_torch = _convert_mesh(pred_mesh_torch)

        if gt_mesh_torch.isempty() and pred_mesh_torch.isempty():
            return 1.0
        elif gt_mesh_torch.isempty() or pred_mesh_torch.isempty():
            return 0.0

        img_consistencies = []
        for EL in ELs:
            for AZ in AZs:
                # Create camera matrix
                RT, _ = get_projection_torch3D(-AZ -90, EL, DISTANCE, img_w=IMG_SIZE, img_h=IMG_SIZE, focal_length=50)
                RT = torch.tensor(RT).float().to(device).unsqueeze(0)
                R_cuda = RT[:, 0:3, 0:3].float().to(device).clone().detach()
                t_cuda = RT[:, 0:3, 3].float().to(device).clone().detach()
                # Create the renderer
                camera, renderer_pytorch3D, light = get_renderer_cameras_lights(R_cuda, t_cuda, R_cuda.device, IMG_SIZE)
                # Render gt and pred
                gt_normal, _ = render_normalmap(gt_mesh_torch, renderer_pytorch3D, camera, light)
                pred_normal, _ = render_normalmap(pred_mesh_torch, renderer_pytorch3D, camera, light)
                # Compute and aggregate image consistency from this vp
                img_consistencies.append(compute_image_consistency(gt_normal, pred_normal))
        return np.mean(img_consistencies)
    
else:

    def image_consistency(gt_mesh_torch, pred_mesh_torch, DISTANCE=2., IMG_SIZE=256,
                          ELs=[-15, 15], AZs=[45, 135, 225, 315], device=_device):
        return np.nan
    