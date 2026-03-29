from dataclasses import dataclass, field
from collections import defaultdict
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from tgs.utils.typing import *
from tgs.utils.base import BaseModule
from tgs.utils.ops import trunc_exp
from tgs.models.networks import MLP
from tgs.utils.ops import scale_tensor
from tgs.models.verts_refinement import vert_valid, vert_pos_refinement
from tgs.models.inter_attn import inter_attn
from tgs.models.self_attn import SelfAttn

from einops import rearrange, reduce
import trimesh

inverse_sigmoid = lambda x: np.log(x / (1 - x))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix_refine(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear) # z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y


class Camera:
    def __init__(self, w2c, intrinsic, FoVx, FoVy, height, width, znear, zfar, trans=np.array([0.0, 0.0, 0.0]), scale=1.0) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.world_view_transform = w2c.transpose(0, 1)


        self.zfar = 1000.0
        self.znear = 0.01

        # self.znear, self.zfar = znear, zfar
        # print("znaer zfar")
        # print([znear,zfar])

        self.trans = trans
        self.scale = scale

        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(w2c.device)
        self.projection_matrix = getProjectionMatrix_refine(intrinsic, self.height, self.width, self.znear, self.zfar).transpose(0, 1).to(w2c.device)
        
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def from_w2c(w2c, intrinsic, height, width, znear, zfar):
        FoVx, FoVy = intrinsic_to_fov(intrinsic, w=torch.tensor(width, device=w2c.device), h=torch.tensor(height, device=w2c.device))
        return Camera(w2c=w2c, intrinsic=intrinsic, FoVx=FoVx, FoVy=FoVy, height=height, width=width, znear=znear, zfar=zfar)

class GaussianModel(NamedTuple):
    xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]*features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        color_data = np.zeros_like(xyz)+1

        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        f_dc = features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3).detach().cpu().numpy())
        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))

        # color = PlyElement.describe(np.array(color_data,dtype=[......]),'color')
        el = PlyElement.describe(elements, 'vertex')
        # PlyData([el, color]).write(path)
        PlyData([el]).write(path)


class GSLayer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 128
        feature_channels: dict = field(default_factory=dict)
        xyz_offset: bool = True
        restrict_offset: bool = False
        use_rgb: bool = False
        clip_scaling: Optional[float] = None
        init_scaling: float = -5.0
        init_density: float = 0.1
        time_weight: bool = False
        time_reg: bool = False
        pose_reg: bool = False

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        self.out_layers = nn.ModuleList()
        for key, out_ch in self.cfg.feature_channels.items():
            if key == "shs" and self.cfg.use_rgb:
                out_ch = 3
                # self.shade_layer = nn.Linear(self.cfg.in_channels, 1)
            layer = nn.Linear(self.cfg.in_channels, out_ch)
            # initialize
            if not (key == "shs" and self.cfg.use_rgb):
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, self.cfg.init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                nn.init.constant_(layer.bias, inverse_sigmoid(self.cfg.init_density))

            self.out_layers.append(layer)

        self.out_layers_shade = nn.ModuleList()
        for key, out_ch in self.cfg.feature_channels.items():
            if key == "shs" and self.cfg.use_rgb:
                out_ch = 3
                self.shade_layer = nn.Linear(self.cfg.in_channels, 1)
            layer = nn.Linear(self.cfg.in_channels, out_ch)
            # initialize
            if not (key == "shs" and self.cfg.use_rgb):
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, self.cfg.init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                nn.init.constant_(layer.bias, inverse_sigmoid(self.cfg.init_density))

            self.out_layers_shade.append(layer)

        self.time_weight_layer = nn.Linear(256, 1)
        nn.init.constant_(self.time_weight_layer.weight, 0)
        nn.init.constant_(self.time_weight_layer.bias, 0)
        self.out_layers_time = nn.ModuleList()
        for key, out_ch in self.cfg.feature_channels.items():
            if key == "shs" and self.cfg.use_rgb:
                out_ch = 3
            layer = nn.Linear(out_ch + 256, out_ch)
            # initialize
            if not (key == "shs" and self.cfg.use_rgb):
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, self.cfg.init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                nn.init.constant_(layer.bias, inverse_sigmoid(self.cfg.init_density))
            self.out_layers_time.append(layer)

    def forward(self, x, x_shade, x_time, pts, x_s):
        ret = {}
        v_time_sum = 0
        if self.cfg.time_reg:
            rand_value = random.random()
            time_weight_weight = 0 if rand_value <= 0.3 else 1
        if self.cfg.pose_reg:
            rand_value = random.random()
            pose_weight = 0 if rand_value <= 0.3 else 1
            print("pose_weight")
            print(pose_weight)
        for k, layer, layer_shade, layer_time in zip(self.cfg.feature_channels.keys(), self.out_layers, self.out_layers_shade, self.out_layers_time):
            v = layer(x)
            v_shade = layer_shade(x_shade)
            if self.cfg.pose_reg:
                v_shade = v_shade*pose_weight
            v = v + v_shade
            # print("v_time")
            # print(v.shape)
            if x_time is not None:
                # print(x_time.shape)
                v_time = layer_time(torch.cat([v,x_time.repeat(v.shape[0],1)],dim=-1))
                time_weight = self.time_weight_layer(x_time)
                time_weight = (torch.sigmoid(time_weight)-0.5)*2
                print("self.time_weight")
                if self.cfg.time_reg:
                    # rand_value = random.random()
                    # time_weight_weight = 0 if rand_value <= 0.3 else 1
                    time_weight = time_weight*time_weight_weight
                if not self.cfg.time_weight:
                    time_weight = time_weight*0
                print(time_weight)
                v = v + time_weight*v_time
                v_time_sum = v_time_sum + v_time.mean()
            else:
                time_weight = torch.FloatTensor(np.zeros((1,)).astype(np.float32))
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                if self.cfg.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.cfg.clip_scaling)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                if self.cfg.use_rgb:
                    v = torch.sigmoid(v)
                    if x_s is not None:
                        v_s = self.shade_layer(x_s)
                        v_s = torch.sigmoid(v_s)
                        # v_s = v_s*0+1
                v = torch.reshape(v, (v.shape[0], -1, 3))
                if x_s is not None:
                    v_s = torch.reshape(v_s, (v_s.shape[0], -1, 1))
                    v=v*v_s
                    # v=(v*0+1)*v_s
                else:
                    v_s = None
            elif k == "xyz":
                if self.cfg.restrict_offset:
                    max_step = 1.2 / 32
                    v = (torch.sigmoid(v) - 0.5) * max_step
                v = v + pts if self.cfg.xyz_offset else pts
                v_pts = v
            ret[k] = v
        return GaussianModel(**ret), v_s, v_pts, ret, time_weight, v_time_sum

class GS3DRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        mlp_network_config: Optional[dict] = None
        gs_out: dict = field(default_factory=dict)
        sh_degree: int = 3
        scaling_modifier: float = 1.0
        random_background: bool = False
        radius: float = 1.0
        radius_texture: float = 1.0
        feature_reduction: str = "concat"
        projection_feature_dim: int = 773
        background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (1.0, 1.0, 1.0)
        )

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        if self.cfg.feature_reduction == "mean":
            mlp_in = 80
        elif self.cfg.feature_reduction == "concat":
            mlp_in = 80 * 3
        else:
            raise NotImplementedError
        mlp_in = 608
        mlp_in_t = 256
        mlp_in_s = 864
        if self.cfg.mlp_network_config is not None:
            self.mlp_net = MLP(mlp_in, self.cfg.gs_out.in_channels, **self.cfg.mlp_network_config)
            self.mlp_net_shade = MLP(mlp_in, self.cfg.gs_out.in_channels, **self.cfg.mlp_network_config)
            self.mlp_net_time = MLP(mlp_in_t, self.cfg.gs_out.in_channels, **self.cfg.mlp_network_config)
            self.mlp_net_s = MLP(mlp_in_s, self.cfg.gs_out.in_channels, **self.cfg.mlp_network_config)
        else:
            self.cfg.gs_out.in_channels = mlp_in
        self.gs_net = GSLayer(self.cfg.gs_out)

    def forward_gs(self, x_tex, x_shade, x_time, p, x_s):
        if self.cfg.mlp_network_config is not None:
            x_tex = self.mlp_net(x_tex)
            x_shade = self.mlp_net_shade(x_shade)
            # x_time = self.mlp_net_shade(x_time)
            if x_s is not None:
                x_s = self.mlp_net_s(x_s)
        return self.gs_net(x_tex, x_shade, x_time, p, x_s)

    def forward_single_view(self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
        color_w = None,
        color_b = None,
        ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        bg_color = background_color
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.cfg.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.cfg.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.cfg.use_rgb:
            colors_precomp = gs.shs.squeeze(1)
            if color_w is not None:
                colors_precomp = colors_precomp*color_w.view(-1,16,3)[:,0,:] + color_w.view(-1,16,3)[:,1,:] -1
                # colors_precomp = colors_precomp*color_w.view(-1,16,3)[:,2,:] + color_w.view(-1,16,3)[:,3,:] -1

            if color_b is not None:
                colors_precomp = colors_precomp+ color_b.view(-1,16,3)[:,0,:]
        else:
            shs = gs.shs
            if color_w is not None:
                # print("shs")
                # print(shs.shape)
                # print(color_b.view(-1,16,3).shape)
                # shs = shs*color_w.view(-1,16,3) + color_b.view(-1,16,3)
                # shs = shs*color_w.view(-1,16,3) + color_b.view(-1,16,3)
                shs = shs*color_w.view(-1,16,3)
            if color_b is not None:
                shs = shs*color_w.view(-1,16,3) + color_b.view(-1,16,3)

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            rendered_image, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
        
        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),
            "comp_rgb_bg": bg_color
        }
        
        if ret_mask:
            mask_bg_color = torch.zeros(3, dtype=torch.float32, device=self.device)
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.height),
                image_width=int(viewpoint_camera.width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=mask_bg_color,
                scale_modifier=self.cfg.scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform.float(),
                sh_degree=0,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=False
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                rendered_mask, radii = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    # shs = ,
                    colors_precomp = torch.ones_like(means3D),
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                ret["comp_mask"] = rendered_mask.permute(1, 2, 0)

        return ret
    
    def query_triplane(
        self,
        positions: Float[Tensor, "*B N 3"],
        triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
    ) -> Dict[str, Tensor]:
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]
        # print("triplane.shape")
        # print(triplanes.shape)
        # print("position")
        # print(positions.max()) #0.8049
        # print(positions.min()) #-0.8926
        # print(positions[...,0].max()) #-0.6366
        # print(positions[...,0].min()) #-0.8444
        # print(positions[...,1].max()) #-0.6414
        # print(positions[...,1].min()) #-0.8679
        # print(positions[...,2].max()) #0.8546
        # print(positions[...,2].min()) #0.5718
        # print(positions.shape)
        positions=positions-positions.mean(-2).unsqueeze(-2)
        # print("position0")
        # print(positions[...,0].max()) #-0.6366
        # print(positions[...,0].min()) #-0.8444
        # print(positions[...,1].max()) #-0.6414
        # print(positions[...,1].min()) #-0.8679
        # print(positions[...,2].max()) #0.8546
        # print(positions[...,2].min()) #0.5718
        # positions = scale_tensor(positions, (-0.20, 0.20), (-1, 1))
        positions = scale_tensor(positions, (-self.cfg.radius, self.cfg.radius), (-1, 1))
        # print("position1")
        # print(positions[...,0].max()) #-0.6366 0.4560
        # print(positions[...,0].min()) #-0.8444 -0.7492
        # print(positions[...,1].max()) #-0.6414 0.6287
        # print(positions[...,1].min()) #-0.8679 -0.6993
        # print(positions[...,2].max()) #0.8546 0.9557
        # print(positions[...,2].min()) #0.5718 -0.6879
        
        indices2D: Float[Tensor, "B 3 N 2"] = torch.stack(
                (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
                dim=-3,
            )
        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3),
            rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3),
            align_corners=True,
            mode="bilinear",
        )

        if self.cfg.feature_reduction == "concat":
            out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
        elif self.cfg.feature_reduction == "mean":
            out = reduce(out, "(B Np) Cp () N -> B N Cp", Np=3, reduction="mean")
        else:
            raise NotImplementedError
        
        if not batched:
            out = out.squeeze(0)

        return out

    def query_triplane_texture(
        self,
        positions: Float[Tensor, "*B N 2"],
        triplanes: Float[Tensor, "*B 1 Cp Hp Wp"],
    ) -> Dict[str, Tensor]:
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]
        # print("triplane.shape")
        # print(triplanes.shape)
        # print("position")
        # print(positions.max()) #0.8049
        # print(positions.min()) #-0.8926
        # print(positions[...,0].max()) #-0.6366
        # print(positions[...,0].min()) #-0.8444
        # print(positions[...,1].max()) #-0.6414
        # print(positions[...,1].min()) #-0.8679
        positions = scale_tensor(positions, (-self.cfg.radius_texture, self.cfg.radius_texture), (-1, 1))
        # print("position1")
        # print(positions[...,0].max()) #-0.6366 0.4560
        # print(positions[...,0].min()) #-0.8444 -0.7492
        # print(positions[...,1].max()) #-0.6414 0.6287
        # print(positions[...,1].min()) #-0.8679 -0.6993
 
        indices2D: Float[Tensor, "B N 2"] = positions[:, :, None]

        # print(positions.shape)
        # print(triplanes.shape)

        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            triplanes.squeeze(1),
            indices2D,
            align_corners=True,
            mode="bilinear",
        )

        # print(out.shape) #[8, 80, 24674, 1]
        out = out.view(*out.shape[:2], -1).permute(0, 2, 1)
        # print(out.shape) #[8, 24674, 80]
        # if self.cfg.feature_reduction == "concat":
        #     out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=1)
        # elif self.cfg.feature_reduction == "mean":
        #     out = reduce(out, "(B Np) Cp () N -> B N Cp", Np=3, reduction="mean")
        # else:
        #     raise NotImplementedError
        
        if not batched:
            out = out.squeeze(0)

        return out

    def forward_single_batch(
        self,
        gs_hidden_features_tex: Float[Tensor, "Np Cp"],
        gs_hidden_features_shade: Float[Tensor, "Np Cp"],
        query_points: Float[Tensor, "Np 3"],
        w2cs: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int,
        znear,
        zfar,
        background_color: Optional[Float[Tensor, "3"]],
        shade_features = None,
        time_features = None,
        color_w = None,
        color_b = None
    ):

        gs, v_s, v_pts, gs_att, time_weight, v_time_sum = self.forward_gs(gs_hidden_features_tex, gs_hidden_features_shade, time_features, query_points, shade_features)
        out_list = []

        if color_b != None:
            # color_b = torch.cat([color_b[...,:1024],color_b[...,:1024]],dim=-1)
            # color_b = torch.cat([color_b[...,1024:],color_b[...,1024:]],dim=-1)
            color_b = self.query_triplane_texture(vert_uv, color_b.unsqueeze(0).unsqueeze(0)).squeeze(0)
            print("color")
            print(color_w.shape)
            print(color_b.shape)
        for w2c, intrinsic in zip(w2cs, intrinsics):
            out_list.append(self.forward_single_view(
                                gs, 
                                Camera.from_w2c(w2c = w2c, intrinsic = intrinsic, height = height, width = width, znear = znear, zfar = zfar),
                                background_color,
                                color_w = color_w,
                                color_b = color_b


                            ))
        
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        try:
            out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        except:
            pass
        out["3dgs"] = gs
        out["v_s"] = v_s
        out["v_pts"] = v_pts
        out["gs_att"] = gs_att
        out["time_weight"] = time_weight
        out["v_time_sum"] = v_time_sum
        # print("my!!!!!!!!")
        return out

    def forward(self, 
        scene_codes_texture_list,
        scene_codes_pose_list,
        vert_uv,
        query_points: Float[Tensor, "B Np 3"],
        w2c: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        znear=0.71, 
        zfar=1.42,
        query_points_tar=None,
        additional_features_tex: Optional[Float[Tensor, "B C H W"]] = None,
        additional_features_pose: Optional[Float[Tensor, "B C H W"]] = None,
        shade_features=None,
        time_features=None,
        background_color: Optional[Float[Tensor, "B 3"]] = None,
        intrinsic_input: Float[Tensor, "B Nv 4 4"] = None,
        w2c_input: Float[Tensor, "B Nv 4 4"] = None,
        texture_rgb = None,
        face = None,
        gs_hidden_features: Float[Tensor, "B Np Cp"] = None,
        mink_idxs_inter = None,
        gs_hidden_features_tex = None,
        gs_hidden_features_pose = None,
        color_w = None,
        color_b = None,


        **kwargs):

        batch_size = scene_codes_texture_list[0].shape[0]

        out_list = []
        out_list_input = []

        # gs_hidden_features = self.query_triplane(query_points, gs_hidden_features)
        # print(gs_hidden_features.shape)
        if gs_hidden_features_tex is None:
            gs_hidden_features_texture_list = []
            for scene_codes_texture in scene_codes_texture_list:
                gs_hidden_features_texture = self.query_triplane_texture(vert_uv, scene_codes_texture)
                # print("gs_hidden_features_texture")
                # print(gs_hidden_features_texture.shape)
                gs_hidden_features_texture_list.append(gs_hidden_features_texture)

            gs_hidden_features_pose_list = []
            for scene_codes_pose in scene_codes_pose_list:
                gs_hidden_features_pose = self.query_triplane_texture(vert_uv, scene_codes_pose)
                # print("gs_hidden_features_texture")
                # print(gs_hidden_features_texture.shape)
                gs_hidden_features_pose_list.append(gs_hidden_features_pose)

            gs_hidden_features_texture = torch.cat(gs_hidden_features_texture_list,dim=-1)
            gs_hidden_features_pose = torch.cat(gs_hidden_features_pose_list,dim=-1)

            if additional_features_tex is not None:
                gs_hidden_features_tex = torch.cat([gs_hidden_features_texture, additional_features_tex], dim=-1)
            if additional_features_pose is not None:
                gs_hidden_features_pose = torch.cat([gs_hidden_features_pose, additional_features_pose], dim=-1)
            if shade_features is not None:
                shade_features = torch.cat([gs_hidden_features_pose, shade_features], dim=-1)

        # if query_points_tar is not None:
        #     query_points = query_points_tar

        # for b in range(batch_size):
        #     out_list.append(self.forward_single_batch(
        #         gs_hidden_features=gs_hidden_features[b],
        #         query_points=query_points[b],
        #         w2cs=w2c[b],
        #         intrinsics=intrinsic[b],
        #         height=height, 
        #         width=width,
        #         znear=znear,
        #         zfar=zfar,
        #         background_color=background_color[b] if background_color is not None else None),
        #         )

        for b in range(batch_size):
            if shade_features is not None:
                shade_features_b = shade_features[b]
            else:
                shade_features_b = shade_features
                time_features_b = time_features
            if time_features is not None:
                time_features_b = time_features[b]
            else:
                time_features_b = time_features
            out_list_input.append(self.forward_single_batch(
                gs_hidden_features_tex=gs_hidden_features_tex[b],
                gs_hidden_features_shade=gs_hidden_features_pose[b],
                shade_features=shade_features_b,
                time_features=time_features_b,
                query_points=query_points[b],
                w2cs=w2c_input[b],
                intrinsics=intrinsic_input[b],
                height=height, 
                width=width,
                znear=znear,
                zfar=zfar,
                color_w = color_w,
                color_b = color_b,


                background_color=background_color[b] if background_color is not None else None),
                )
        
        # print(out_list)

        out = defaultdict(list)
        out_input = defaultdict(list)

        # for out_ in out_list:
        #     for k, v in out_.items():
        #         out[k].append(v)
        # for k, v in out.items():
        #     if isinstance(v[0], torch.Tensor):
        #         out[k] = torch.stack(v, dim=0)
        #     else:
        #         out[k] = v

        for out_ in out_list_input:
            for k, v in out_.items():
                out_input[k+'_input'].append(v)
        for k, v in out_input.items():
            if isinstance(v[0], torch.Tensor):
                out_input[k] = torch.stack(v, dim=0)
            else:
                out_input[k] = v

        for k, v in out_input.items():
            out[k] = v

        print(out.keys())
        return out
        