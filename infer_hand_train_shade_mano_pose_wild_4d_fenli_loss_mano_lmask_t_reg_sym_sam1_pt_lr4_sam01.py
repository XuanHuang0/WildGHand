import torch
from dataclasses import dataclass, field
from einops import rearrange
import os
from torch.utils.data import DataLoader
import tgs
from tgs.models.image_feature import ImageFeature
from tgs.utils.saving import SaverMixin
from tgs.utils.config import parse_structured
from tgs.utils.ops import points_projection_my, points_projection
from tgs.utils.misc import load_module_weights
from tgs.utils.typing import *
from tgs.utils.ops import scale_tensor
from utils.render_vis import render_img

import torch.nn.functional as F
import torch.nn as nn
from model_utils_mask_lr import compute_error, VGGLoss, LaplacianReg

import cv2
import numpy as np
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import Trainer, loggers
import copy
import evaluator
# from .DINAR.discriminators.style_gan_v2 import Discriminator
import yaml
import config
from spatial import SpatialEncoder
from livehand.input_encoder import read_mano_uv_obj, save_obj_for_debugging, get_uvd
import trimesh
import time
import math
from tgs.models.verts_refinement import additional_features_fc

import smplx
from smplx.manohd.subdivide import sub_mano
from smplx.utils import vertex_normals
from smplx.lbs import get_normal_coord_system

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

class CostTime(object):
    def __init__(self):
        self.t = 0
        self.t_list=[]

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'cost time:{time.perf_counter() - self.t:.8f} s')
        # self.t_list.append(time.perf_counter() - self.t)
        # print(mean(self.t_list))

def concat_meshes(mesh_list):
    '''manually concat meshes'''
    cur_vert_number = 0
    cur_face_number = 0
    verts_list = []
    faces_list = []
    for idx, m in enumerate(mesh_list):
        verts_list.append(m.vertices)
        faces_list.append(m.faces + cur_vert_number)
        cur_vert_number += len(m.vertices)

    combined_mesh = trimesh.Trimesh(np.concatenate(verts_list),
        np.concatenate(faces_list), process=False
    )
    return combined_mesh

class Soft_threshold(nn.Module):
    def __init__(self, threshold_init=0.5):
        super(Soft_threshold, self).__init__()
        # 使用可学习的阈值，初始化为0.0
        self.threshold = nn.Parameter(torch.tensor(threshold_init))

    def forward_big(self, x):
        logits = x - self.threshold
        return F.gumbel_softmax(logits, tau=1.0, hard=False)
    
    def forward_small(self, x):
        logits = self.threshold - x
        return F.gumbel_softmax(logits, tau=1.0, hard=False)

class Soft_threshold_relu(nn.Module):
    def __init__(self, threshold_init=0.5):
        super(Soft_threshold_relu, self).__init__()
        # 使用可学习的阈值，初始化为0.5
        self.threshold = nn.Parameter(torch.tensor(threshold_init))

    def forward_big(self, x):
        # 使用 ReLU 来模拟 x > self.threshold 的效果
        result = F.relu(x - self.threshold)  # 如果 x > threshold，结果为正值
        return result

    def forward_small(self, x):
        # 使用 ReLU 来模拟 x > self.threshold 的效果
        result = F.relu(self.threshold-x)  # 如果 x > threshold，结果为正值
        return result


class TGS(torch.nn.Module, SaverMixin):
    @dataclass
    class Config:
        radius_texture: float = 1.0
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        pose_embedder_cls: str = ""
        pose_embedder: dict = field(default_factory=dict)

        time_embedder_cls: str = ""
        time_embedder: dict = field(default_factory=dict)

        image_feature: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_shade_cls: str = ""
        tokenizer_shade: dict = field(default_factory=dict)

        tokenizer_texture_cls: str = ""
        tokenizer_texture: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        backbone_shade_cls: str = ""
        backbone_shade: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        post_processor_texture_cls: str = ""
        post_processor_texture: dict = field(default_factory=dict)

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

        pointcloud_generator_cls: str = ""
        pointcloud_generator: dict = field(default_factory=dict)

        pointcloud_encoder_shade_cls: str = ""
        pointcloud_encoder_shade: dict = field(default_factory=dict)

        pointcloud_encoder_texture_cls: str = ""
        pointcloud_encoder_texture: dict = field(default_factory=dict)

        smpl_cfg: dict = field(default_factory=dict)
        deform_network: dict = field(default_factory=dict)

        add_time_bias: bool = False
        add_id_bias: bool = False

        sp_encoder_time: int = 4

    
    cfg: Config

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=False)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None

        self.image_tokenizer = tgs.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )

        assert self.cfg.camera_embedder_cls == 'tgs.models.networks.MLP'
        weights = self.cfg.camera_embedder.pop("weights") if "weights" in self.cfg.camera_embedder else None
        self.camera_embedder = tgs.find(self.cfg.camera_embedder_cls)(**self.cfg.camera_embedder)
        if weights:
            from tgs.utils.misc import load_module_weights
            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.camera_embedder.load_state_dict(state_dict)

        assert self.cfg.pose_embedder_cls == 'tgs.models.networks.MLP'
        weights = self.cfg.pose_embedder.pop("weights") if "weights" in self.cfg.pose_embedder else None
        self.pose_embedder = tgs.find(self.cfg.pose_embedder_cls)(**self.cfg.pose_embedder)
        if weights:
            from tgs.utils.misc import load_module_weights
            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.pose_embedder.load_state_dict(state_dict)

        assert self.cfg.time_embedder_cls == 'tgs.models.networks.MLP'
        weights = self.cfg.time_embedder.pop("weights") if "weights" in self.cfg.time_embedder else None
        self.time_embedder = tgs.find(self.cfg.time_embedder_cls)(**self.cfg.time_embedder)
        if weights:
            from tgs.utils.misc import load_module_weights
            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.time_embedder.load_state_dict(state_dict)

        self.image_feature = ImageFeature(self.cfg.image_feature)

        self.tokenizer_shade = tgs.find(self.cfg.tokenizer_shade_cls)(self.cfg.tokenizer_shade)
        self.tokenizer_texture = tgs.find(self.cfg.tokenizer_texture_cls)(self.cfg.tokenizer_texture)

        self.backbone = tgs.find(self.cfg.backbone_cls)(self.cfg.backbone)

        self.backbone_shade = tgs.find(self.cfg.backbone_shade_cls)(self.cfg.backbone_shade)


        self.post_processor = tgs.find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.post_processor_texture = tgs.find(self.cfg.post_processor_texture_cls)(
            self.cfg.post_processor_texture
        )

        self.post_processor_pose = tgs.find(self.cfg.post_processor_texture_cls)(
            self.cfg.post_processor_texture
        )

        self.renderer = tgs.find(self.cfg.renderer_cls)(self.cfg.renderer)

        # pointcloud generator
        self.pointcloud_generator = tgs.find(self.cfg.pointcloud_generator_cls)(self.cfg.pointcloud_generator)

        self.point_encoder_shade = tgs.find(self.cfg.pointcloud_encoder_shade_cls)(self.cfg.pointcloud_encoder_shade)
        self.point_encoder_texture = tgs.find(self.cfg.pointcloud_encoder_texture_cls)(self.cfg.pointcloud_encoder_texture)

        self.identity_code_book = torch.nn.Parameter(torch.clamp(torch.normal(mean=0.0, std=0.02, size=(27, 1, 33, 64, 128)), -1, 1))
        self.identity_code_one_shot = torch.nn.Parameter(torch.clamp(torch.randn(size=(10, 1, 33, 64, 128)), -1, 1))

        # self.x_threshold = torch.nn.Parameter(torch.ones(size=(1,))*-1000.0)
        # self.y_threshold = torch.nn.Parameter(torch.ones(size=(1,))*-1000.0)
        # self.z_threshold = torch.nn.Parameter(torch.ones(size=(1,))*-1000.0)

        # self.x_threshold = Soft_threshold_relu(threshold_init=0.5)
        # self.y_threshold = Soft_threshold_relu(threshold_init=0.3)
        # self.z_threshold = Soft_threshold_relu(threshold_init=0.5)


        self.sp_encoder = SpatialEncoder(sp_level = 4)
        self.sp_encoder_time = SpatialEncoder(sp_level = self.cfg.sp_encoder_time)

        self.additional_features_fc_shade_ = additional_features_fc(788,128)
        self.additional_features_fc_tex_ = additional_features_fc(53,128)
        # self.additional_features_fc = additional_features_fc(851,512)
        self.template_offsets_fc_ = additional_features_fc(628,3)
        self.id_fc_ = additional_features_fc(270336,256)
        self.pose_offsets_fc_ = additional_features_fc(65638,102)
        self.shape_offsets_fc_ = additional_features_fc(32788,20)
        self.tokens_texture_fc_ = additional_features_fc(2048,64)
        self.tokens_pose_fc_ = additional_features_fc(2048,64)
        self.shade_features_fc_ = additional_features_fc(818,256)
        self.pose_offsets_time_fc = additional_features_fc(614,102)
        self.id_time_fc_ = additional_features_fc(512,256)





        vanerf_path="/home/huangx/vanerf"
        # mano layer
        smplx_path = vanerf_path+'/smplx/models/'
        mano_layer = {'right': smplx.create(smplx_path,'mano', ncomps=45,  use_pca=False, is_rhand=True, flat_hand_mean=False), 'left': smplx.create(smplx_path,'mano', ncomps=45,  use_pca=False, is_rhand=False, flat_hand_mean=False)}
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
            print('Fix shapedirs bug of MANO')
            mano_layer['left'].shapedirs[:,0,:] *= -1

        # manohd
        # smpl_body = smplx.create(**cfg.smpl_cfg).requires_grad_(False)
        # if not cfg.smpl_cfg.is_rhand:
        #     smpl_body.shapedirs[:,0,:] *= -1
        if self.cfg.smpl_cfg['manohd']>0:
            print('MANO-HD in Model')
            mano_layer['right'] ,_ ,_ = sub_mano(mano_layer['right'], self.cfg.smpl_cfg['manohd'])
            # mano_layer['right'] ,_ ,_ = sub_mano(mano_layer['right'], self.cfg.smpl_cfg['manohd'])
            # lbs_weights = torch.load(self.cfg.smpl_cfg['lbs_weights'], map_location='cpu')
            # mano_layer['right'].lbs_weights = lbs_weights

            mano_layer['left'] ,_ ,_ = sub_mano(mano_layer['left'], self.cfg.smpl_cfg['manohd'])
            # mano_layer['left'] ,_ ,_ = sub_mano(mano_layer['left'], self.cfg.smpl_cfg['manohd'])
            # lbs_weights = torch.load(self.cfg.smpl_cfg['lbs_weights'], map_location='cpu')
            # mano_layer['left'].lbs_weights = lbs_weights
        
        self.mano_layer_right = mano_layer['right']
        self.mano_layer_left = mano_layer['left']

        # deform
        # self.pointcloud_generator = tgs.find(self.cfg.pointcloud_generator_cls)(self.cfg.pointcloud_generator)
        if not self.cfg.smpl_cfg['ignore_deform']:
            self.center_template_right = torch.mm(self.mano_layer_right.J_regressor, self.mano_layer_right.v_template)[4]
            self.deformer_right = tgs.find(self.cfg.deform_network.module)(verts=self.mano_layer_right.v_template[None], **self.cfg.deform_network)

            self.center_template_left = torch.mm(self.mano_layer_left.J_regressor, self.mano_layer_left.v_template)[4]
            self.deformer_left = tgs.find(self.cfg.deform_network.module)(verts=self.mano_layer_left.v_template[None], **self.cfg.deform_network)

        # load checkpoint
        if self.cfg.weights is not None:
            self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)

        t_mano_pose = torch.FloatTensor(np.zeros((1,2,48)).astype(np.float32))
        t_trans_left = torch.FloatTensor(np.ones((1,3)).astype(np.float32))*0.5
        t_trans_right = torch.FloatTensor(np.zeros((1,3)).astype(np.float32))
        t_mano_shape = torch.FloatTensor(np.zeros((1,2,10)).astype(np.float32))
        t_root_pose = t_mano_pose[:,:,:3]
        t_hand_pose = t_mano_pose[:,:,3:]
        t_shaped_verts_right = None
        t_shaped_verts_left = None

        t_smpl_output_r = self.mano_layer_right(
            t_mano_shape[:,0,:], 
            t_root_pose[:,0,:], 
            t_hand_pose[:,0,:], 
            transl=t_trans_right,
            return_verts=True, 
            return_full_pose=True,
            shaped_verts=t_shaped_verts_right,
        )

        t_smpl_output_l = self.mano_layer_left(
            t_mano_shape[:,1,:], 
            t_root_pose[:,1,:], 
            t_hand_pose[:,1,:], 
            transl=t_trans_left,
            return_verts=True, 
            return_full_pose=True,
            shaped_verts=t_shaped_verts_left,
        )

        self.t_pointclouds = torch.cat([t_smpl_output_r.vertices, t_smpl_output_r.vertices],dim=1)
        face_r = self.mano_layer_right.faces
        face_l = self.mano_layer_left.faces

        print(self.t_pointclouds.shape)

        mesh_v_r = trimesh.Trimesh(vertices=t_smpl_output_r.vertices[0].clone().detach().cpu().numpy(), faces=face_r, process=False)
        # mesh_v_r.export('/home/huangx/TriplaneGaussian_online/mesh_wild/mesh_texture'+str(frame_idx[0])+'r.obj')
        mesh_v_l = trimesh.Trimesh(vertices=t_smpl_output_r.vertices[0].clone().detach().cpu().numpy(), faces=face_r, process=False)
        # mesh_v_l.export('/home/huangx/TriplaneGaussian_online/mesh_wild/mesh_texture'+str(frame_idx[0])+'l.obj')
        mesh_v = concat_meshes([mesh_v_r, mesh_v_l])

        vertex_num_upsampled = self.t_pointclouds.shape[1]
        self.vertex_num_single = t_smpl_output_r.vertices.shape[1]

        print(vertex_num_upsampled)
        self.lap_reg = LaplacianReg(vertex_num_upsampled, mesh_v.faces)

    
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
       
        # print("position")
        # print(positions.max())
        # print(positions.min())
        # print(positions[...,0].max()) #-0.6366
        # print(positions[...,0].min()) #-0.8444
        # print(positions[...,1].max()) #-0.6414
        # print(positions[...,1].min()) #-0.8679
        # print(positions[...,2].max()) #0.8546
        # print(positions[...,2].min()) #0.5718
        
        positions=positions-positions.mean(-2).unsqueeze(-2)

        # print("position")
        # print(positions.max())
        # print(positions.min())
        # print(positions[...,0].max()) #-0.6366
        # print(positions[...,0].min()) #-0.8444
        # print(positions[...,1].max()) #-0.6414
        # print(positions[...,1].min()) #-0.8679
        # print(positions[...,2].max()) #0.8546
        # print(positions[...,2].min()) #0.5718

        positions = scale_tensor(positions, (-self.cfg.radius_texture, self.cfg.radius_texture), (-1, 1))
        
        # print(self.cfg.radius_texture)
        # print("position")
        # print(positions.max())
        # print(positions.min())
        # print(positions[...,0].max()) #-0.6366
        # print(positions[...,0].min()) #-0.8444
        # print(positions[...,1].max()) #-0.6414
        # print(positions[...,1].min()) #-0.8679
        # print(positions[...,2].max()) #0.8546
        # print(positions[...,2].min()) #0.5718

        
        indices2D: Float[Tensor, "B N 2"] = positions[:, :, None]

        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            triplanes.squeeze(1),
            indices2D,
            align_corners=True,
            mode="bilinear",
        )

        # print(out.shape) #[8, 80, 24674, 1]
        out = out.view(*out.shape[:2], -1).permute(0, 2, 1)
        # print(out.shape) #[8, 24674, 80]
        if not batched:
            out = out.squeeze(0)

        return out

    def _forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # generate point cloud
        # out = self.pointcloud_generator(batch)
        # pointclouds = out["points"]
        # print("points")
        # print(pointclouds.shape) [8, 16384, 3]
        out={}
        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        mano_pose = batch['ret']['targets']['input_mano_pose'] #[1, 2, 48]
        mano_shape = batch['ret']['targets']['input_mano_shape'] #[1, 2, 10]
        mano_trans = batch['ret']['targets']['input_mano_trans'] #[1, 2, 3]

        tar_cam = batch['ret']['tar_cam']

        root_pose = mano_pose[:,:,:3]
        hand_pose = mano_pose[:,:,3:]
 
        # t_mano_pose = torch.FloatTensor(np.zeros((1,2,48)).astype(np.float32)).to(mano_pose.device)
        # t_trans_left = torch.FloatTensor(np.ones((1,3)).astype(np.float32)).to(mano_pose.device)*0.5
        # t_trans_right = torch.FloatTensor(np.zeros((1,3)).astype(np.float32)).to(mano_pose.device)
        # t_mano_shape = torch.FloatTensor(np.zeros((1,2,10)).astype(np.float32)).to(mano_pose.device)
        # t_root_pose = t_mano_pose[:,:,:3]
        # t_hand_pose = t_mano_pose[:,:,3:]
        # t_shaped_verts_right = None
        # t_shaped_verts_left = None

        # t_smpl_output_r = self.mano_layer_right(
        #     t_mano_shape[:,0,:], 
        #     t_root_pose[:,0,:], 
        #     t_hand_pose[:,0,:], 
        #     transl=t_trans_right,
        #     return_verts=True, 
        #     return_full_pose=True,
        #     shaped_verts=t_shaped_verts_right,
        # )

        # t_smpl_output_l = self.mano_layer_left(
        #     t_mano_shape[:,1,:], 
        #     t_root_pose[:,1,:], 
        #     t_hand_pose[:,1,:], 
        #     transl=t_trans_left,
        #     return_verts=True, 
        #     return_full_pose=True,
        #     shaped_verts=t_shaped_verts_left,
        # )

        # t_pointclouds = torch.cat([t_smpl_output_r.vertices, t_smpl_output_l.vertices],dim=1)

        t_pointclouds = self.t_pointclouds.to(mano_pose.device)

        n_points = t_pointclouds.shape[1]

        targets = batch['ret']['targets']
        vert3d_uv_or=targets['vert_uv_or']
        face_uv_or=targets['face_uv_or'].long()[0]
        face_uv_xy_or=targets['face_uv_xy_or'][0]

        capture_id = batch['ret']['human_idx']
        # print(self.identity_code_book.shape)
        # identity_code = self.identity_code_book[capture_id,:]
        identity_code = self.identity_code_one_shot[capture_id,:]

        # identity_code = identity_code.unsqueeze(1)
        print("id")
        print(capture_id)
        # print(identity_code.shape) 

        # v_template = torch.cat([self.mano_layer_right.v_template,self.mano_layer_left.v_template],dim=0)

        # print(vert3d_uv_or[0].max())
        # print(vert3d_uv_or[0].min())
        # print(t_pointclouds.max())
        # print(t_pointclouds.min())
        
        verts_uv,verts_d=[],[]
        for b in range(batch_size):
            vert_uv, vert_d, intermediates_vert = get_uvd(t_pointclouds[0], vert3d_uv_or[0], face_uv_or, face_uv_xy_or)
            verts_uv.append(vert_uv)
            verts_d.append(vert_d)
        vert_uv=torch.cat(verts_uv,0)
        vert_d=torch.cat(verts_d,0)

        # normalize it to [-1, 1]
        vert_uv[:,:self.vertex_num_single, 0] = 2.0 * (vert_uv[:,:self.vertex_num_single, 0] /1) - 1.0
        vert_uv[:,self.vertex_num_single:, 0] = 2.0 * (vert_uv[:,self.vertex_num_single:, 0] /1)
        vert_uv[..., 1] = 2.0 * (vert_uv[..., 1] /0.5) - 1.0

        print("vert_uv")
        print(vert_uv[...,0].max()) #-0.6366
        print(vert_uv[...,0].min()) #-0.8444
        print(vert_uv[...,1].max()) #-0.6414
        print(vert_uv[...,1].min()) #-0.8679
        
        # print(torch.cat([vert_uv,identity_code], dim=-1).shape)
        vert_uv_sp = self.sp_encoder(vert_uv)
        # print(vert_uv_sp.shape)
        # print('id_code')
        # print(identity_code.shape)
        identity_code_vert = self.query_triplane_texture(vert_uv, identity_code)
        # print(identity_code_vert.shape)

        face_r = self.mano_layer_right.faces
        face_l = self.mano_layer_left.faces

        # [1, 1, 33, 64, 128]
        id_feature = self.id_fc_(identity_code.view(identity_code.shape[0], -1))
        # print(id_feature.shape)

        frame_idx = batch['ret']['frame_index']

        # Camera modulation
        camera_extri = batch["c2w_cond"].view(*batch["c2w_cond"].shape[:-2], -1)
        camera_intri = batch["intrinsic_normed_cond"].view(*batch["intrinsic_normed_cond"].shape[:-2], -1)
        camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)
        # print(camera_feats.shape) #[2, 1, 25]
        camera_feats = self.camera_embedder(camera_feats)
        # print(camera_feats.shape) #[2, 1, 768]

        mano_pose = torch.cat([root_pose, hand_pose], dim=-1)
        pose = mano_pose.view(batch_size,1,-1)
        shape = mano_shape.view(batch_size,1,-1)

        pose_feats = self.pose_embedder(pose)

        c2w_cond = batch["c2w_cond"].squeeze(1)
        w2c_cond = batch["w2c_cond"].squeeze(1)
        intrinsic_cond = batch["intrinsic_cond"].squeeze(1)
        face = batch['ret']['targets']['face_world']


        point_cond_embeddings_texture = self.point_encoder_texture(torch.cat([vert_uv, vert_uv_sp, identity_code_vert], dim=-1))
        tokens_texture: Float[Tensor, "B Ct Nt"] = self.tokenizer_texture(batch_size, cond_embeddings=point_cond_embeddings_texture)
        tokens_texture = self.backbone(tokens_texture)

        # point_cond_embeddings_pose = self.point_encoder_shade(torch.cat([vert_uv, vert_uv_sp, pointclouds, pointclouds_sp, pose_feats.repeat(1,n_points,1)], dim=-1))
        point_cond_embeddings_pose = self.point_encoder_shade(torch.cat([vert_uv, vert_uv_sp, pose_feats.repeat(1,n_points,1)], dim=-1))
        tokens_pose: Float[Tensor, "B Ct Nt"] = self.tokenizer_shade(batch_size, cond_embeddings=point_cond_embeddings_pose)
        tokens_pose = self.backbone_shade(tokens_pose)

        print("tokens_texture")
        print(tokens_texture.shape) #[2, 512, 2048]
        print(tokens_pose.shape) #[2, 512, 2048]

        scene_codes_texture = self.tokenizer_texture.detokenize(tokens_texture)
        scene_codes_pose = self.tokenizer_texture.detokenize(tokens_pose)

        out['scene_codes_texture'] = scene_codes_texture

        scene_codes_texture_list = self.post_processor_texture(scene_codes_texture)
        scene_codes_pose_list = self.post_processor_pose(scene_codes_pose)

        # print("scene_codes_texture")
        # print(scene_codes_texture.shape) #torch.Size([1, 2, 512, 32, 32])


        additional_features_tex = torch.cat([vert_uv, vert_uv_sp, identity_code_vert], dim=-1)
        print(additional_features_tex.shape) #[1, 98562, 512]
        additional_features_tex = self.additional_features_fc_tex_(additional_features_tex) #
        print(additional_features_tex.shape) #[1, 98562, 512]

        additional_features_pose = torch.cat([vert_uv, vert_uv_sp, pose_feats.repeat(1,n_points,1)], dim=-1)
        # print(additional_features_shade.shape) #[1, 98562, 512]
        additional_features_pose = self.additional_features_fc_shade_(additional_features_pose) #
        # print(additional_features_shade.shape) #[1, 98562, 512]
        # print(gs_hidden_features.shape)

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

        additional_features_offsets = torch.cat([vert_uv, vert_uv_sp, gs_hidden_features_tex], dim=-1)
        # print('additional_features_offsets')
        # print(additional_features_offsets.shape)
        template_offsets = self.template_offsets_fc_(additional_features_offsets) #
        clamp = float(0.005*(self.mano_layer_right.v_template.max()-self.mano_layer_right.v_template.min()).detach())
        template_offsets = torch.clamp(template_offsets, -1*clamp, clamp)
        # print(template_offsets.shape) #[1, 98562, 512]

        normals_right = vertex_normals(self.mano_layer_right.v_template[None], torch.from_numpy(face_r[None]).to(self.mano_layer_right.v_template.device))
        # print('normals_right')
        # print(normals_right.shape)
        normal_coord_sys_right = get_normal_coord_system(normals_right.view(-1, 3)).view(1, self.mano_layer_right.v_template.shape[0], 3, 3)
        # offsets = torch.matmul(normal_coord_sys.permute(0, 1, 3, 2), offsets.unsqueeze(-1)).squeeze(-1) 

        if not self.cfg.smpl_cfg['ignore_deform']:
            offsets_right = template_offsets[:,:int(template_offsets.shape[1]/2),:]
            offsets_right = torch.matmul(normal_coord_sys_right.permute(0, 1, 3, 2), offsets_right.unsqueeze(-1)).squeeze(-1) 
            shaped_verts_right = self.mano_layer_right.v_template[None].clone().detach() + offsets_right
            center_right = torch.mm(self.mano_layer_right.J_regressor, shaped_verts_right[0])[4]
            shaped_verts_right = shaped_verts_right - center_right[None, None] + self.center_template_right[None, None].to(shaped_verts_right.device)

            out["verts_template"] = torch.cat([self.mano_layer_right.v_template[None].clone().detach(),self.mano_layer_right.v_template[None].clone().detach()],dim=-2)
            out["shaped_verts"] = torch.cat([shaped_verts_right,shaped_verts_right],dim=-2)

            # offsets_left = template_offsets[:,int(template_offsets.shape[1]/2):,:]
            # offsets_left = torch.matmul(normal_coord_sys_right.permute(0, 1, 3, 2), offsets_left.unsqueeze(-1)).squeeze(-1)          
            # shaped_verts_left = self.mano_layer_right.v_template[None] + offsets_left
            # center_left = torch.mm(self.mano_layer_right.J_regressor, shaped_verts_left[0])[4]
            # shaped_verts_left = shaped_verts_left - center_left[None, None] + self.center_template_right[None, None].to(shaped_verts_left.device)
        else:
            shaped_verts_right = None
            shaped_verts_left = None

        tokens_texture_feat = self.tokens_texture_fc_(tokens_texture)
        tokens_pose_feat = self.tokens_pose_fc_(tokens_pose)

        # shape_features_offsets = torch.cat([mano_shape.view(mano_shape.shape[0],-1), tokens_texture_feat.view(mano_shape.shape[0],-1)], dim=-1)
        # # print('additional_features_offsets')
        # # print(additional_features_offsets.shape)[2, 512, 2048]
        # shape_offsets = self.shape_offsets_fc_(shape_features_offsets) 
        # clamp = float(0.005*(mano_shape.max()-mano_shape.min()).clone().detach())
        # shape_offsets = torch.clamp(shape_offsets, -1*clamp, clamp)
        # # print(shape_offsets.shape)
        # mano_shape = mano_shape + shape_offsets.reshape(shape_offsets.shape[0],2,-1)

        pose_features_offsets = torch.cat([mano_pose.view(mano_shape.shape[0],-1), mano_trans.view(mano_shape.shape[0],-1), tokens_texture_feat.view(mano_shape.shape[0],-1), tokens_pose_feat.view(mano_shape.shape[0],-1)], dim=-1)
        pose_offsets = self.pose_offsets_fc_(pose_features_offsets)
        pose_offsets =  pose_offsets.reshape(pose_offsets.shape[0],2,-1)

        clamp_r = float(0.0005*(root_pose.max()-root_pose.min()).clone().detach())
        clamp_h = float(0.0005*(hand_pose.max()-hand_pose.min()).clone().detach())
        clamp_t = float(0.0005*(mano_trans.max()-mano_trans.min()).clone().detach())
        clamp=min([clamp_r, clamp_h, clamp_t])
        pose_offsets = torch.clamp(pose_offsets, -1*clamp, clamp)

        root_pose = root_pose + pose_offsets[:,:,:3]
        hand_pose = hand_pose + pose_offsets[:,:,3:48]
        mano_trans = mano_trans + pose_offsets[:,:,48:]
        mano_pose = torch.cat([root_pose,hand_pose],dim=-1)

        if self.cfg.add_time_bias:
            total_frame = batch["total_frame"]
            time_interval = 1 / total_frame
            frame_id = batch["index"]
            frame_t =  (frame_id*time_interval.squeeze(-1)).view(batch_size,-1)
            frame_t_sp = self.sp_encoder_time(frame_t.unsqueeze(1))
            frame_t_sp = self.time_embedder(frame_t_sp)

            total_params1 = sum(p.numel() for p in self.sp_encoder_time.parameters())
            total_params2 = sum(p.numel() for p in self.time_embedder.parameters())
            print('total_params1', total_params1, 'total_params2', total_params2)


            pose_features_offsets_time = torch.cat([mano_pose.view(mano_shape.shape[0],-1), mano_trans.view(mano_shape.shape[0],-1), id_feature, frame_t_sp.squeeze(1)], dim=-1)
            print(pose_features_offsets_time.shape) #torch.Size([1, 367]) torch.Size([2, 376])

            pose_offsets_time = self.pose_offsets_time_fc(pose_features_offsets_time)
            pose_offsets_time =  pose_offsets_time.reshape(pose_offsets_time.shape[0],2,-1)

            clamp_r = float(0.0005*(root_pose.max()-root_pose.min()).clone().detach())
            clamp_h = float(0.0005*(hand_pose.max()-hand_pose.min()).clone().detach())
            clamp_t = float(0.0005*(mano_trans.max()-mano_trans.min()).clone().detach())
            clamp=min([clamp_r, clamp_h, clamp_t])
            pose_offsets_time = torch.clamp(pose_offsets_time, -1*clamp, clamp)

            root_pose = root_pose + pose_offsets_time[:,:,:3]
            hand_pose = hand_pose + pose_offsets_time[:,:,3:48]
            mano_trans = mano_trans + pose_offsets_time[:,:,48:]

        # pose_offsets = torch.cat([vert_uv, vert_uv_sp, identity_code_vert], dim=-1)
        
        smpl_output_r = self.mano_layer_right(
            mano_shape[:,0,:], 
            root_pose[:,0,:], 
            hand_pose[:,0,:], 
            # transl=mano_trans[:,0,:],
            return_verts=True, 
            return_full_pose=True,
            shaped_verts=shaped_verts_right,
        )

        smpl_output_l = self.mano_layer_right(
            mano_shape[:,1,:], 
            root_pose[:,1,:], 
            hand_pose[:,1,:], 
            # transl=mano_trans[:,1,:],
            return_verts=True, 
            return_full_pose=True,
            shaped_verts=shaped_verts_right,
        )

        pred_rel = targets['root_rel']

        joints_right, joints_left = torch.bmm(self.mano_layer_right.J_regressor[None].repeat(batch_size,1,1), smpl_output_r.vertices), torch.bmm(self.mano_layer_left.J_regressor[None].repeat(batch_size,1,1), smpl_output_l.vertices)
        verts_right, verts_left = smpl_output_r.vertices, smpl_output_l.vertices
        root_right, root_left = joints_right[:, 4:5, :].clone(), joints_left[:, 4:5, :].clone()
        verts_w_right = verts_right - root_right
        verts_w_left = verts_left - root_left
        joints_w_right = joints_right - root_right
        joints_w_left = joints_left - root_left

        verts_w_left[:, :, 0] = -verts_w_left[:, :, 0]
        joints_w_left[:, :, 0] = -joints_w_left[:, :, 0]
     
        verts_w_left = verts_w_left + pred_rel
        joints_w_left = joints_w_left + pred_rel

        pointclouds = torch.cat([verts_w_right + mano_trans[:,0:1,:], verts_w_left + mano_trans[:,1:2,:]],dim=1)

        
        # mesh_v_r = trimesh.Trimesh(vertices=(verts_w_right + mano_trans[:,0:1,:])[0].clone().detach().cpu().numpy(), faces=face_r, process=False)
        # mesh_v_r.export('/home/huangx/TriplaneGaussian_online/mesh_wild/mesh_texture'+str(frame_idx[0])+'r.obj')
        # mesh_v_l = trimesh.Trimesh(vertices=(verts_w_left + mano_trans[:,1:2,:])[0].clone().detach().cpu().numpy(), faces=face_l, process=False)
        # mesh_v_l.export('/home/huangx/TriplaneGaussian_online/mesh_wild/mesh_texture'+str(frame_idx[0])+'l.obj')
        # mesh_v = concat_meshes([mesh_v_r, mesh_v_l])
        # mesh_v.export('/home/huangx/TriplaneGaussian_online/mesh_wild/mesh_texture'+str(frame_idx[0])+'.obj')

        mano_msks=[]
        for b in range(batch_size):
            mesh_v_r = trimesh.Trimesh(vertices=(verts_w_right + mano_trans[:,0:1,:])[b].clone().detach().cpu().numpy(), faces=face_r, process=False)
            # mesh_v_r.export('/home/huangx/TriplaneGaussian_online/mesh_wild/mesh_texture'+str(frame_idx[0])+'r.obj')
            mesh_v_l = trimesh.Trimesh(vertices=(verts_w_left + mano_trans[:,1:2,:])[b].clone().detach().cpu().numpy(), faces=face_r, process=False)
            # mesh_v_l.export('/home/huangx/TriplaneGaussian_online/mesh_wild/mesh_texture'+str(frame_idx[0])+'l.obj')
            mesh_v = concat_meshes([mesh_v_r, mesh_v_l])
            # mesh_v.export('/home/huangx/TriplaneGaussian_online/mesh_wild/mesh_texture'+str(frame_idx[0])+'.obj')
            mano_msk=render_img(torch.from_numpy(mesh_v.vertices).float(), torch.from_numpy(mesh_v.faces), tar_cam['input_R'][b:b+1,...].float(), tar_cam['input_T'][b:b+1,...].float(), tar_cam['input_focal'][b:b+1,0].float(), tar_cam['input_focal'][b:b+1,1].float(), tar_cam['input_princpt'][b:b+1,0].float(), tar_cam['input_princpt'][b:b+1,1].float(), device=pointclouds.device)
            mano_msks.append(mano_msk.unsqueeze(0))
        mano_msk=torch.cat(mano_msks,0)
        print("mask")
        print(mano_msk.shape)
        

        pointclouds_sp = self.sp_encoder(pointclouds)

        out["points"] = pointclouds

        out["mano_msk"] = mano_msk
        out["mano_shape"] = mano_shape
        # print(pointclouds.shape) #[8, 24674, 3] [4, 49488, 3] [8, 12448, 3]

        shade_features = torch.cat([vert_uv, vert_uv_sp, pointclouds, pointclouds_sp, pose_feats.repeat(1,n_points,1)], dim=-1)
        # print(shade_features.shape) #[1, 98562, 512]
        shade_features = self.shade_features_fc_(shade_features) #
        # print(shade_features.shape) #[1, 98562, 512]
        if shade_features is not None:
            shade_features = torch.cat([gs_hidden_features_pose, shade_features], dim=-1)
        
        if self.cfg.add_id_bias:
            print("frame_t_sp.shape")
            print(frame_t_sp.shape)
            print(id_feature.shape)
            frame_t_sp = torch.cat([frame_t_sp.squeeze(1), id_feature], dim=-1)
            print(frame_t_sp.shape)
            frame_t_sp = self.id_time_fc_(frame_t_sp)

        if self.cfg.add_time_bias:
            rend_out = self.renderer(scene_codes_texture_list=scene_codes_texture_list,
                                    scene_codes_pose_list=scene_codes_pose_list,
                                    vert_uv=vert_uv,
                                    shade_features=shade_features,
                                    time_features=frame_t_sp,
                                    gs_hidden_features_tex = gs_hidden_features_tex,
                                    gs_hidden_features_pose = gs_hidden_features_pose,
                                    # texture_rgb=texture_rgb,
                                    # face=face,
                                    query_points=pointclouds,
                                    query_points_tar=pointclouds,
                                    # additional_features=additional_features,
                                    height=256,
                                    width=256,
                                    intrinsic_input = batch["intrinsic_cond"],
                                    w2c_input = batch["w2c_cond"],
                                    **batch)
        else:
            rend_out = self.renderer(scene_codes_texture_list=scene_codes_texture_list,
                                    scene_codes_pose_list=scene_codes_pose_list,
                                    vert_uv=vert_uv,
                                    shade_features=shade_features,
                                    gs_hidden_features_tex = gs_hidden_features_tex,
                                    gs_hidden_features_pose = gs_hidden_features_pose,
                                    # time_feature=frame_t_sp,
                                    # texture_rgb=texture_rgb,
                                    # face=face,
                                    query_points=pointclouds,
                                    query_points_tar=pointclouds,
                                    # additional_features=additional_features,
                                    height=256,
                                    width=256,
                                    intrinsic_input = batch["intrinsic_cond"],
                                    w2c_input = batch["w2c_cond"],
                                    **batch)
        # batch_size = batch["index"].shape[0]
        # for b in range(batch_size):
        #     if batch["view_index"][b, 0] == 0:
        #         rend_out["3dgs"][b].save_ply(self.get_save_path(f"3dgs/{batch['instance_id'][b]}.ply"))
        #         print("save_ply")

        # rend_out["3dgs_input"][0].save_ply("./overview/"+index+"rgb_input_or.ply")
        # print("save_ply")

        # input_img = batch['rgb_cond'].squeeze(1).permute(0, 3, 1, 2)[0].permute(1, 2, 0).detach().cpu().numpy()*255
        # rgb_pred = rend_out["comp_rgb_input"].squeeze(1).permute(0, 3, 1, 2)[0].permute(1, 2, 0).detach().cpu().numpy()*255
        # cv2.imwrite("./overview/"+index+"rgb_input.jpg", input_img[...,[2,1,0]])
        # cv2.imwrite("./overview/"+index+"rgb_pred_input_or.jpg", rgb_pred[...,[2,1,0]])

        return {**out, **rend_out}
    
    def forward(self, batch):
        out = self._forward(batch)
        batch_size = batch["index"].shape[0]
        for b in range(batch_size):
            if batch["view_index"][b, 0] == 0:
                out["3dgs"][b].save_ply(self.get_save_path(f"3dgs/{batch['instance_id'][b]}.ply"))

            for index, render_image in enumerate(out["comp_rgb"][b]):
                view_index = batch["view_index"][b, index]
                self.save_image_grid(
                    f"video/{batch['instance_id'][b]}/{view_index}.png",
                    [
                        {
                            "type": "rgb",
                            "img": render_image,
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                )
        

class HandLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, cfg: dict, cfg_model: dict, TGS_cfg: dict):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.kwargs = self.cfg['models']['VANeRF']
        self.tgs_cfg = copy.deepcopy(TGS_cfg)
        self.cfg_model = cfg_model
        self.idx = 0
        self.expname = cfg['expname']
        self.save_dir = f'{cfg["out_dir"]}/{cfg["expname"]}'
        self.save_hyperparameters()
        self.dataset = Dataset
        self.video_dirname = 'video'
        self.images_dirname = 'images'
        self.test_dst_name = cfg['test_dst_name']
        # self.nkpt_r,self.nkpt_l=778,778
        self.nkpt_r,self.nkpt_l=21,21
        self.model = TGS(self.tgs_cfg.system)
        self.model.set_save_dir("outputs")
        # self.discriminator=Discriminator(image_size=cfg['models']['Discriminator']['params']['image_size'],activation_layer=cfg['models']['Discriminator']['params']['activation_layer'], channel_multiplier=cfg['models']['Discriminator']['params']['channel_multiplier'])
        self.vgg_loss = VGGLoss()

        sam_checkpoint = "/home/huangx/TriplaneGaussian_online/EXPERIMENTS/arxive/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        # self.x_threshold = Soft_threshold_relu(threshold_init=2.0)
        # self.y_threshold = Soft_threshold_relu(threshold_init=0.3)
        # self.z_threshold = Soft_threshold_relu(threshold_init=0.5)

        self.x_threshold = Soft_threshold_relu(threshold_init=cfg.get("x_threshold", 2.0))
        self.y_threshold = Soft_threshold_relu(threshold_init=cfg.get("y_threshold", 0.3))
        self.z_threshold = Soft_threshold_relu(threshold_init=cfg.get("z_threshold", 0.5))


        self.evaluator = evaluator.Evaluator()
        self.pretrained_path = cfg.get("pretrained_path", '/home/huangx/TriplaneGaussian_online/EXPERIMENTS/online_shade_mano_pose_all_fenli_mano/ckpts/0.ckpt')
        print('!!!!!!!!!!!!!!!!!!!!!!')
        print(self.pretrained_path)
        if self.pretrained_path != "None":
            pretrained_model = torch.load(self.pretrained_path)
            # print(pretrained_model['state_dict'].keys())
            self.load_state_dict(pretrained_model['state_dict'], strict=False)
            if "fine_tune" in cfg:
                print("trainable!!!!!!!!!!!!!!!!!!!!!!!")
                for name, param in self.model.named_parameters():
                    # if 'map_bias' in name or 'color_w' in name or 'color_b' in name or 'opacity_b' in name or 'identity_code_one_shot' in name or ('out_layers' in name and '4' in name):
                    for ft in cfg["fine_tune"]:
                        if ft in name:
                        # if 'offsets' in name or 'time' in name or 'map_bias' in name or 'color_w' in name or 'color_b' in name or 'opacity_b' in name or 'identity_code_one_shot' in name:
                            print(name)
                            continue
                    param.requires_grad = False
            else:
                print("all trainable")
        # self.load_state_dict(pretrained_model['state_dict'])
        # self.pretrained_path = '/home/huangx/TriplaneGaussian-main/EXPERIMENTS/tgs_new_64_9w_texture_identity_map_valid_inter_attn_points_sp_c0_train_all/ckpts/6e_all.ckpt'
        # pretrained_model = torch.load(self.pretrained_path)
        # # print(pretrained_model['state_dict'].keys())
        # self.load_state_dict(pretrained_model['state_dict'], strict=False)

    def configure_optimizers(self):
        opt_g=torch.optim.Adam(self.model.parameters(), lr=self.cfg['training'].get('lr', 1e-4))

        opt_g = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr':self.cfg['training'].get('lr', 1e-4)},
            {'params': self.y_threshold.parameters(), 'lr':self.cfg['training'].get('lr_y', 1e-6)},
            {'params': self.z_threshold.parameters(), 'lr':self.cfg['training'].get('lr_z', 1e-6)},
            {'params': self.x_threshold.parameters(), 'lr':self.cfg['training'].get('lr_x', 1e-6)},

        ])

        # 设置warm up的轮次为100次
        warm_up_iter = 100
        T_max = 100	# 周期
        lr_max = self.cfg['training'].get('lr_max', 1)	# 最大值
        lr_min = self.cfg['training'].get('lr_min', 1e-2)	# 最小值

        # # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
        # lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
        #         (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_iter-warm_up_iter)/(T_max-warm_up_iter+0.000001)*math.pi)))/0.1

        # step_scheduler = torch.optim.lr_scheduler.StepLR(opt_g, step_size=2, gamma=0.8)

        step_scheduler = torch.optim.lr_scheduler.StepLR(opt_g, step_size=self.cfg['training'].get('step_size', 2), gamma=self.cfg['training'].get('gamma', 0.8))


        scheduler = {'scheduler':step_scheduler}

        # optim_dict_g = {'optimizer': opt_g, 'lr_scheduler': scheduler}
        return [opt_g], [scheduler]
        # return torch.optim.Adam(self.model.parameters(), lr=self.cfg['training'].get('lr', 1e-5))

    # def configure_optimizers(self):
    #     # return torch.optim.Adam(self.model.parameters(), lr=self.cfg['training'].get('lr', 1e-5))
    #     opt_g=torch.optim.Adam(self.model.parameters(), lr=self.cfg['training'].get('lr', 1e-5))
    #     opt_d=torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg['training'].get('lr', 1e-5))
    #     StepLR_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=[2,5,10,20,35], gamma=0.5)
    #     StepLR_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, milestones=[2,5,10,20,35], gamma=0.5)
    #     optim_dict_g = {'optimizer': opt_g, 'lr_scheduler': StepLR_g}
    #     optim_dict_d = {'optimizer': opt_d, 'lr_scheduler': StepLR_d}
    #     return [optim_dict_g,optim_dict_d]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, closure=None, **kwargs):
        # 检查和处理 NaN 梯度

        for name, param in self.named_parameters():
            if "threshold" in name or "_fc" in name or "identity_code" in name:
                try:
                    print(name, param.grad.max())
                    print(name, param.requires_grad)
                except:
                    print(name,"grad,none!!!!!!!!!!!")

        for param in self.parameters():
            if param.grad is not None:
                nan_mask = torch.isnan(param.grad)
                inf_idxs = torch.isinf(param.grad)
                # if nan_mask.any():
                param.grad[nan_mask] = 0  # 将 NaN 值置为 0
                param.grad[inf_idxs] = 0 
                if nan_mask.any():
                    print("nan_mask")
                    print(nan_mask)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # 执行梯度计算
        if closure is not None:
            closure()
        
        # 执行优化器步进
        optimizer.step()

    @classmethod
    def from_config(cls, cfg_hand, cfg_model, cfg_tgs):
        return cls(cfg_hand, cfg_model, cfg_tgs)

    def train_dataloader(self, batch_size=None):
        train_dataset = self.dataset.from_config(self.cfg['dataset'], 'train', self.cfg)
        return torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            # shuffle=False,
            num_workers=self.cfg['training'].get('train_num_workers', 0),
            batch_size=self.cfg['training'].get('train_batch_size', 1) if batch_size is None else batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self, batch_size=None):
        val_dataset = self.dataset.from_config(self.cfg['dataset'], 'val', self.cfg)
        return torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            num_workers=self.cfg['training'].get('val_num_workers', 0),
            batch_size=self.cfg['training'].get('val_batch_size', 1) if batch_size is None else batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self, batch_size=None):
        test_dataset = self.dataset.from_config(self.cfg['dataset'], 'test', self.cfg)
        return torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            num_workers=self.cfg['training'].get('val_num_workers', 0),
            batch_size=self.cfg['training'].get('val_batch_size', 1) if batch_size is None else batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def save_ckpt(self, **kwargs):
        pass

    def test_epoch_end(self, outputs):
        
        results = {key: torch.stack([x[key] for x in outputs]).mean() for key in outputs[0].keys()}
        results = {key: float(val.item()) if torch.is_tensor(val) else float(val) for key, val in results.items()}

        

        path = os.path.join(
            self.save_dir,  f'test_{self.test_dst_name}_{self.current_epoch}_{self.global_step}.yml')

        print(results)
        with open(path, 'w') as f:
            yaml.dump(results, f)

        print('Results saved in', path)
        print(results)

    @staticmethod
    def collate_fn(items):
        """ Modified form of :func:`torch.utils.data.dataloader.default_collate` that will strip samples from
        the batch if they are ``None``.
        """
        try:
            items = [item for item in items if item is not None]
            return torch.utils.data.dataloader.default_collate(items) if len(items) > 0 else None
        except Exception as e:
            return None
    
    def load_ckpt(self, ckpt_path):
        assert os.path.exists(ckpt_path), f'Checkpoint ({ckpt_path}) does not exists!'
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["state_dict"])
        return ckpt['epoch'], ckpt['global_step']

    @staticmethod
    def compute_test_metric(rendered_img, gt_img, mask=None, max_val=1.0):
        """
        Args:
            rendered_img (torch.tensor): (3, H, W) or (B, 3, H, W) [0, 1.0]
            gt_img (torch.tensor): (3, H, W) or (B, 3, H, W) [0, 1.0]
            mask (torch.tensor: torch.bool): (3, H, W) or (B, 3, H, W) [0, 1.0]
        """
        assert rendered_img.shape == gt_img.shape
        if len(rendered_img.shape) == 3:
            rendered_img = rendered_img.unsqueeze(0)
            gt_img = gt_img.unsqueeze(0)
        mask = mask.view(1, *mask.shape[-2:]) if mask is not None else mask

        # B,3,H,W
        ssim = K.metrics.ssim(rendered_img, gt_img, window_size=7, max_val=max_val)
        ssim = ssim.permute(0, 2, 3, 1)[mask] if mask is not None else ssim
        ssim = ssim.mean()

        if mask is not None:
            rendered_img = rendered_img.permute(0, 2, 3, 1)[mask]
            gt_img = gt_img.permute(0, 2, 3, 1)[mask]

        return {
            f'psnr': K.metrics.psnr(rendered_img, gt_img, max_val=max_val),
            f'ssim': ssim,
        }

    def save_test_image(self, batch, rendered_img, gt_img, mask=None, face_mask=None):
        """
        Args:
            rendered_img (torch.tensor): (3, H, W) [0, 1.0]
        """
        index = batch['index']
        sub_id = index['frame'][0]
        tar_cam_id = index['tar_cam_id'][0]
        # prepare directory
        dst_dir = os.path.join(
            self.save_dir,
            f'{self.images_dirname}_{self.test_dst_name}',  # _{self.current_epoch}_{self.global_step}
            sub_id)
        cond_mkdir(dst_dir)

        # save images
        if rendered_img is not None:
            rendered_img = tensor_to_image(rendered_img)  # H,W,3
            rendered_img = (rendered_img*255.).astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.pred.png')
            cv2.imwrite(path, rendered_img[:, :, ::-1])

        if gt_img is not None:
            gt_img = tensor_to_image(gt_img)
            gt_img = (gt_img*255.).astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.gt.png')
            cv2.imwrite(path, gt_img[:, :, ::-1])
        
        if mask is not None:
            mask = (mask*255.).squeeze().unsqueeze(-1).repeat(1, 1,  3)
            gt_img = mask.detach().cpu().numpy().astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.mask.png')
            cv2.imwrite(path, gt_img[:, :, ::-1])

        if face_mask is not None:
            face_mask = (face_mask*255.).squeeze().unsqueeze(-1).repeat(1, 1,  3)
            gt_img = face_mask.detach().cpu().numpy().astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.face_mask.png')
            cv2.imwrite(path, gt_img[:, :, ::-1])

    def training_step(self, batch, batch_idx):
        out = self.model._forward(batch)
        # print(out.keys())
        lambdas = self.kwargs.get("lambdas", {})
        expname = self.cfg.get('expname', 'ex_wild')
        os.makedirs(os.path.dirname("./vis/"+expname+"/"), exist_ok=True)

        if "comp_rgb_input" in out:
            out['input_img'] = batch['rgb_cond'].clone().squeeze(1).permute(0, 3, 1, 2)
            rbg_input = batch['rgb_cond'].clone()
            msk_input = batch['mask'].clone()
            rbg_input[msk_input <= 0.1] = 0
            rbg_masked = rbg_input
            # print(batch['rgb_cond'].shape)
            # print(batch['mask'].shape)
            out['input_msk'] = batch['mask'].squeeze(1).permute(0, 3, 1, 2)
            out["tex_cal_fine_input"] = out["comp_rgb_input"].squeeze(1).permute(0, 3, 1, 2).clone()
            out["alpha_fine_input"] = out['comp_mask_input'].float().mean(-1).unsqueeze(-1).squeeze(1).permute(0, 3, 1, 2)
            out["tar_alpha_input"] = out['mano_msk'].float().unsqueeze(1)
            out["tar_alpha_input_sam"] = batch['mask'].squeeze(1).permute(0, 3, 1, 2)[:,0,:,:]

            res1 = (out["tex_cal_fine_input"] - out['input_img']).abs()
            # res1[out["tar_alpha_input"].repeat(1,3,1,1)<=0.1]=0

            out['input_img'][out['input_msk']<0.5] = 0
            time_weight = out["time_weight_input"]

            frame_idx = int(batch['ret']['frame_index'][0])

            # out['input_img'][out["tar_alpha_input"].repeat(1,3,1,1).clone().detach()<0.5] = 0

            if batch['bbox_mask'] is not None:
                print(batch['bbox_mask'].shape)
                bbox_mask = batch['bbox_mask'].unsqueeze(1)
                bbox_mask_ = batch['bbox_mask'][0].clone().detach().cpu().numpy()*255
                cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_bbox"+".jpg", bbox_mask_)
                rgb_pred = out["tex_cal_fine_input"][0].permute(1, 2, 0).clone().detach().cpu().numpy()*255
                cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_pred_input_.jpg", rgb_pred[...,[2,1,0]])
                out['input_msk'][bbox_mask.repeat(1,3,1,1) == 0] = 0
                out["alpha_fine_input"][bbox_mask == 0] = 0
                out["tex_cal_fine_input"][bbox_mask.repeat(1,3,1,1) == 0] = 0
                out['input_img'][bbox_mask.repeat(1,3,1,1) == 0] = 0

            # print(out["alpha_fine_input"].shape)
            # print(out["tar_alpha_input"].shape)
            # print(rbg_masked.shape)
            # print(out["tex_cal_fine_input"].shape)
            # print(out["tar_alpha_input_sam"].shape)
            # print(out["alpha_fine_input"].max())
            # print(out["tar_alpha_input"].max())

            
            res = (out["tex_cal_fine_input"] - rbg_masked.squeeze(1).permute(0, 3, 1, 2)).abs()
            # res[out["tar_alpha_input"].repeat(1,3,1,1)<=0.1]=0

            # res1 = (out["tex_cal_fine_input"] - out['input_img']).abs()
            # res1[out["tar_alpha_input"].repeat(1,3,1,1)<=0.1]=0

            min_val = res.min()
            max_val = res.max()
            res = (res - min_val) / (max_val - min_val)

            min_val = res1.min()
            max_val = res1.max()
            res1 = (res1 - min_val) / (max_val - min_val)

        
            batch_size, c, h, w= out['input_img'].shape
            dis_mask_list = []
            for b in range(batch_size):
                dis_mask = torch.zeros(h,w).to(time_weight.device).requires_grad_(True)
                image = batch['rgb_cond'].squeeze(1)[b].clone().detach().cpu().numpy()*255
                masks_auto = self.mask_generator.generate(image.astype("uint8"))
                res_mask = res1[b].clone().mean(0)
                # print("res_mask.mean()") #0.1879 0.0510
                res_mask_mean = res_mask.sum()/out['input_msk'][b].mean(0).sum()
                # print(res_mask_mean)
                dis_masks = []
                for mask_auto in masks_auto:
                    # res_mask = res1[b].clone().mean(0)
                    # print("res_mask.mean()") #0.1879 0.0510
                    # res_mask_mean = res_mask.sum()/out['input_msk'][b].mean(0).sum()
                    # print(res_mask_mean)
                    sam_mask = torch.from_numpy(mask_auto['segmentation']).clone()
                    sam_mask_all = torch.from_numpy(mask_auto['segmentation']).clone()
                    # print(out["tar_alpha_input"].shape)
                    sam_mask [ out["tar_alpha_input"][b].squeeze(0) < 0.1 ] = 0
                    rate_sam = sam_mask.sum() / out["tar_alpha_input"][b].sum()
                    # if rate_sam < self.model.y_threshold:
                    #     continue
                    res_mask = res1[b].clone().mean(0)
                    res_mask[sam_mask<0.5] = 0
                    if sam_mask.sum()>0:
                        rate = res_mask.sum()/sam_mask.sum()
                    else:
                        continue
                    # print(rate)
                    sam_mask_all = (sam_mask_all).float().to(time_weight.device)
                    z_threshold = self.z_threshold
                    y_threshold = self.y_threshold
                    # sam_mask_all = sam_mask_all*(rate > z_threshold)*(rate_sam > y_threshold)
                    sam_mask_all = 100*sam_mask_all*(self.z_threshold.forward_small(rate))*(self.y_threshold.forward_big(rate_sam))
                    sam_mask_all = sam_mask_all * (1/1+1000*(int(time_weight[b].clone().detach().abs())))
                    if sam_mask_all.max()>0:
                        print(time_weight[b])
                        print("sam_mask_all",sam_mask_all.max())
                        print("rate",rate,"rate_sam",rate_sam, "z", self.z_threshold.threshold, "y", self.y_threshold.threshold)
                    # sam_mask_all = (sam_mask_all<0.5)
                    dis_mask= dis_mask + sam_mask_all
                    # dis_masks.append(sam_mask)
                    sam_mask_ = sam_mask_all.clone().detach().cpu().numpy()*255    
                    dis_mask_ = dis_mask.clone().detach().cpu().numpy()*255     
                    res_mask_ = res_mask.clone().detach().cpu().numpy()*255 
                    # if rate<self.model.z_threshold.threshold and rate_sam>self.model.y_threshold.threshold:
                    #     cv2.imwrite("./vis/"+expname+"/rgb"+str(batch_idx)+"_pred_dis_mask_"+str(rate)+str(rate_sam)+".jpg", sam_mask_ )
                    #     cv2.imwrite("./vis/"+expname+"/rgb"+str(batch_idx)+"_pred_dis_mask_all"+str(rate)+str(rate_sam)+".jpg", dis_mask_)
                    #     cv2.imwrite("./vis/"+expname+"/rgb"+str(batch_idx)+"_pred_res_mask_all"+str(rate)+str(rate_sam)+".jpg", res_mask_)

                # print(len(dis_masks))
                dis_mask_list.append(dis_mask.unsqueeze(0))
            if len(dis_mask_list) > 0 and self.cfg.get('dis_mask', True):     
                dis_mask = torch.cat(dis_mask_list, dim = 0)
                out['dis_mask'] = dis_mask.float().unsqueeze(1).repeat(1,3,1,1)
                out['dis_mask'][out['input_msk']<0.5] = 0
                # if res_mask_mean > self.model.x_threshold:
                #     out['dis_mask'][out['input_msk']<0.5] = 0
                # out['dis_mask'][out['input_msk']<0.5] = out['dis_mask'][out['input_msk']<0.5]*(res_mask_mean < self.model.x_threshold)
                print('res_mask_mean',res_mask_mean,'x',self.x_threshold.threshold)
                print(out['alpha_fine_input'].shape)
                print(out['dis_mask'].shape)
                out['dis_mask'][out['alpha_fine_input'].repeat(1,3,1,1)<0.95] = out['dis_mask'][out['alpha_fine_input'].repeat(1,3,1,1)<0.95]+(self.x_threshold.forward_small(res_mask_mean))
                
                out['dis_mask'] = out['dis_mask']*0 + 1
                out['dis_mask'][out['input_msk']<0.5] = 0

                dis_mask_ = out['dis_mask'][0][0].clone().detach().cpu().numpy()
                dis_mask_  = (dis_mask_ - dis_mask_.min()) / (dis_mask_.max() - dis_mask_.min())*255

                dis_img = batch['rgb_cond'].squeeze(1).clone()
                dis_img[dis_mask.unsqueeze(-1).repeat(1,1,1,3)<0.5]=0
                dis_img = dis_img[0].clone().detach().cpu().numpy()*255
                # print(dis_img.max())
                # print(dis_img.shape)
                cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_pred_dis_mask_all"+".jpg", dis_mask_)
                cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_pred_dis_img_all"+".jpg", dis_img[...,[2,1,0]])


                rgb_pred = out["tex_cal_fine_input"][0].permute(1, 2, 0).clone().detach().cpu().numpy()*255
                cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_pred_input.jpg", rgb_pred[...,[2,1,0]])

                rgb_res = res[0].permute(1, 2, 0).clone().detach().cpu().numpy()*255
                cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_pred_res.jpg", rgb_res[...,[2,1,0]])

                rgb_res1 = res1[0].permute(1, 2, 0).clone().detach().cpu().numpy()*255
                cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_pred_res1.jpg", rgb_res1[...,[2,1,0]])



            # rgb_pred = out["tex_cal_fine_input"][0].permute(1, 2, 0).detach().cpu().numpy()*255
            # cv2.imwrite("./vis/"+expname+"/rgb"+str(batch_idx)+"_pred_input_masked.jpg", rgb_pred[...,[2,1,0]])
        # print(out['alpha_fine'].shape)        
        # print(out["tar_alpha"].shape)
        # print("render result")
        # print(out["comp_rgb"].max())
        # print(out["comp_rgb"].min())
        # print("gt")
        # print(out['tar_img'].max())
        # print(out['tar_img'].min())

        # rgb_pred = out["tex_cal_fine"][0].permute(1, 2, 0).detach().cpu().numpy()*255
        # print(rgb_pred.shape)
        # input_imgs = input_imgs.detach().cpu().numpy()  # V, H, W, 1
        # rgb_gt = out['tar_img'][0].permute(1, 2, 0).detach().cpu().numpy()*255
        # print(rgb_gt.shape)

        

        rgb_input = batch['rgb_cond'].squeeze(1)[0].clone().detach().cpu().numpy()*255
        rbg_masked_img = rbg_masked.squeeze(1)[0].clone().detach().cpu().numpy()*255
        # mask_input = batch['mask'].squeeze(1)[0].detach().cpu().numpy()*255
        cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_input.jpg", rgb_input[...,[2,1,0]])
        cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_msked.jpg", rbg_masked_img[...,[2,1,0]])

        rbg_input = batch['rgb_cond'].clone().squeeze(1)
        print(out['input_msk'].shape)
        # out['input_msk'][out["tar_alpha_input"].repeat(1,3,1,1)<0.1] = 0 
        rbg_input[out['input_msk'].permute(0, 2, 3, 1) <= 0.1] = 0
        rbg_masked = rbg_input
        rbg_masked_img = rbg_masked.squeeze(1)[0].clone().detach().cpu().numpy()*255
        cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_msked_mano.jpg", rbg_masked_img[...,[2,1,0]])

        if 'dis_mask' in out:
            rbg_input = batch['rgb_cond'].clone().squeeze(1)
            out['input_msk'][out['dis_mask']<0.1] = 0 
            rbg_input[out['input_msk'].permute(0, 2, 3, 1) <= 0.1] = 0
            rbg_masked = rbg_input
            rbg_masked_img = rbg_masked.squeeze(1)[0].clone().detach().cpu().numpy()*255
            cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_msked_mano_dis.jpg", rbg_masked_img[...,[2,1,0]])
        

        input_msk = out['input_msk'].permute(0, 2, 3, 1)[0].clone().detach().cpu().numpy()*255
        cv2.imwrite("./vis/"+expname+"/rgb"+str(frame_idx)+"_input_msk.jpg", input_msk[...,[2,1,0]])
        # print(out['tar_img'].shape)
        loss, err_dict = compute_error(out_nerf=out, vggloss=self.vgg_loss, lambdas=lambdas, lap_reg = self.model.lap_reg)

        print(err_dict)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss contains NaN or Inf!!!!!!!")

        return {'loss':loss}    

    # def validation_step(self, batch, batch_idx):
    #     prefix = 'val/'
    #     # tr_batch = self.decode_batch(batch)
    #     # out_dict = self.model(**tr_batch)
        
    #     # nerf_level = max(0, int(math.log(tr_batch['im'].shape[-2], 2)) - 5)
    #     # out_nerf = self.render_full_nerf_image(tr_batch, nerf_level)
    #     # renderings = self._arrange_nerf_images(out_nerf, self.znear, self.zfar)  # H, Wt, 3
    #     # src_imgs = self._arrange_src_images(tr_batch['im'], renderings.shape[0])  # H,Ws,3
    #     # gt_img = out_nerf['tar_img'].permute(1, 2, 0).cpu().numpy()
    #     # input_densepose=out_nerf['input_densepose'].permute(1, 2, 0).cpu().numpy()
    #     # tar_densepose=out_nerf['tar_densepose'].permute(1, 2, 0).cpu().numpy()
    #     # vis_img_gt = out_nerf['vis_img'].squeeze(0).expand(3,-1,-1).permute(1, 2, 0).cpu().numpy()
    #     # fake_vis_pred = out_nerf['fake_vis_pred'].squeeze(0).expand(3,-1,-1).permute(1, 2, 0).cpu().numpy()
    #     # real_vis_pred = out_nerf['real_vis_pred'].squeeze(0).expand(3,-1,-1).permute(1, 2, 0).cpu().numpy()
    #     # msk = tr_batch['tar_img_mask'].squeeze(0).expand(3,-1,-1).permute(1, 2, 0).cpu().numpy()
    #     # fake_vis_pred[~msk]=1.
    #     # real_vis_pred[~msk]=1.

    #     # log_img = torch.from_numpy(np.concatenate((src_imgs, gt_img, renderings, input_densepose, tar_densepose, msk, vis_img_gt, real_vis_pred, fake_vis_pred), axis=-2))
    #     # self.logger.experiment.add_image(f'{prefix}renderings', log_img.permute(2, 0, 1), self.global_step)

    #     # out_losses = out_dict['err_dict']
    #     # log = {f'{prefix}{key}': val for key, val in out_losses.items() if torch.is_tensor(val)}
    #     # log['val_total_loss'] = out_dict['loss']

    #     out = self.model._forward(batch)
    #     lambdas = self.kwargs.get("lambdas", {})
    #     # out['tar_img'] = batch['tar_img'].squeeze(1).permute(0, 3, 1, 2)
    #     # out["tex_cal_fine"] = out["comp_rgb"].squeeze(1).permute(0, 3, 1, 2)
    #     # # print(out['comp_mask'].shape)        
    #     # # print(batch['tar_msk_all'].shape)
    #     # out["alpha_fine"] = out['comp_mask'].float().mean(-1).unsqueeze(-1).squeeze(1).permute(0, 3, 1, 2)
    #     # out["tar_alpha"] = batch['tar_msk_all'].float().squeeze(1).permute(0, 3, 1, 2)

    #     # rgb_pred = out["tex_cal_fine"][0].permute(1, 2, 0).detach().cpu().numpy()*255
    #     # rgb_gt = out['tar_img'][0].permute(1, 2, 0).detach().cpu().numpy()*255
    #     rgb_input = batch['rgb_cond'].squeeze(1)[0].detach().cpu().numpy()*255

    #     msk_pred = out["alpha_fine"][0].permute(1, 2, 0).detach().cpu().numpy()*255
    #     msk_gt = out["tar_alpha"][0].permute(1, 2, 0).detach().cpu().numpy()*255

    #     # cv2.imwrite("/home/huangx/TriplaneGaussian-main/TriplaneGaussian-main/ex_cj_up_val/rgb_pred"+str(batch_idx)+".jpg", rgb_pred[...,[2,1,0]])
    #     # cv2.imwrite("/home/huangx/TriplaneGaussian-main/TriplaneGaussian-main/ex_cj_up_val/rgb_gt"+str(batch_idx)+".jpg", rgb_gt[...,[2,1,0]])
    #     # cv2.imwrite("/home/huangx/TriplaneGaussian-main/TriplaneGaussian-main/ex_cj_up_val/rgb_input"+str(batch_idx)+".jpg", rgb_input[...,[2,1,0]])
    #     # cv2.imwrite("/home/huangx/TriplaneGaussian-main/TriplaneGaussian-main/ex_cj_up_val/msk_pred"+str(batch_idx)+".jpg", msk_pred)
    #     # cv2.imwrite("/home/huangx/TriplaneGaussian-main/TriplaneGaussian-main/ex_cj_up_val/msk_gt"+str(batch_idx)+".jpg", msk_gt)
        
    #     # print(out['tar_img'].shape)
    #     loss, err_dict = compute_error(out_nerf=out, vggloss=self.vgg_loss, lambdas=lambdas)

    #     log_img = torch.from_numpy(np.concatenate((rgb_input, rgb_gt, rgb_pred), axis=-2))
    #     self.logger.experiment.add_image(f'{prefix}renderings', log_img.permute(2, 0, 1), self.global_step)

    #     out_losses = err_dict
    #     log = {f'{prefix}{key}': val for key, val in out_losses.items() if torch.is_tensor(val)}
    #     log['val_total_loss'] = loss

    #     return log

    def test_step(self, batch, batch_nb):
        self.evaluator.result_dir = os.path.join(
            self.save_dir,
            f'{self.images_dirname}_{self.test_dst_name}')

        with CostTime():
            out = self.model._forward(batch)
        out['tar_img'] = batch['rgb_cond'].squeeze(1).permute(0, 3, 1, 2).clone()
        out["tar_alpha_input"] = out['mano_msk'].float().unsqueeze(1)
        out['input_msk'] = batch['mask'].squeeze(1).permute(0, 3, 1, 2).clone()
        out['tar_img'][out['input_msk']<0.5] = 0
        out['tar_img'][out["tar_alpha_input"].repeat(1,3,1,1).detach()<0.5] = 0
        out["tex_cal_fine"] = out["comp_rgb_input"].squeeze(1).permute(0, 3, 1, 2).clone()
        # out['tex_cal_fine'][out['input_msk']<0.5] = 0
        out['tex_cal_fine'][out["tar_alpha_input"].repeat(1,3,1,1).detach()<0.5] = 0
        rendered_image = out["tex_cal_fine"]
        # print(rendered_image.max())
        human_idx = str(int(batch['ret']['human_idx'][0]))
        frame_index = str(int(batch['ret']['frame_index'][0]))
        view_index = str(int(batch['ret']['cam_ind'][0]))
        # print('Processing:', human_idx, frame_index, view_index)
        bbox_mask = batch['bbox_mask'].unsqueeze(1)
        out["tex_cal_fine"][bbox_mask.repeat(1,3,1,1) == 0] =0
        rendered_image = out["tex_cal_fine"]
        scores = self.evaluator.compute_score(
            rendered_image,
            out['tar_img'],
            input_imgs=batch['rgb_cond'].squeeze(1),
            mask_at_box=batch['ret']['mask_at_box'],
            human_idx=human_idx,
            frame_index=frame_index,
            view_index=view_index,
            # ka_xy=out_nerf['ka_xy'], 
            # vert_vis=out_nerf['vert_vis'],
            # vert_xy=out_nerf['vert_xy']
            # fake_vis_pred=fake_vis_pred,
            # real_vis_pred=real_vis_pred,
            # vis_img_gt=vis_img_gt,
            # msk=msk
        )
        scores = {key: torch.tensor(val) for key, val in scores.items()}

        for key, val in scores.items():
            if torch.isinf(val):
                return None
        out['tar_img'] = batch['rgb_cond'].squeeze(1).permute(0, 3, 1, 2).clone()
        out["tex_cal_fine"] = out["comp_rgb_input"].squeeze(1).permute(0, 3, 1, 2).clone()
        # out["tex_cal_fine"][bbox_mask.repeat(1,3,1,1) == 0] =0
        rendered_image = out["tex_cal_fine"]

        human_dir = os.path.join(self.evaluator.result_dir, human_idx)
        render_dir = os.path.join(human_dir, 'render')
        tw_dir = os.path.join(human_dir, 'time_weight')

        os.system(f'mkdir -p {render_dir}')
        os.system(f'mkdir -p {tw_dir}')

        # save images
        psnr = str(float(scores['psnr']))
        rgb_pred = rendered_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        input_imgs = batch['rgb_cond'].squeeze(1).detach().cpu().numpy()  # V, H, W, 1
        rgb_gt = out['tar_img'].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        cv2.imwrite(os.path.join(render_dir, f'frame{frame_index}_view{view_index}_gt.png'), (rgb_gt[..., [2, 1, 0]]*255))
        cv2.imwrite(os.path.join(render_dir, f'frame{frame_index}_view{view_index}_pred_{psnr}_lr4.png'), (rgb_pred[..., [2, 1, 0]]*255))

        out["tex_cal_fine"][bbox_mask.repeat(1,3,1,1) == 0] =0
        rendered_image = out["tex_cal_fine"]
        rgb_pred = rendered_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        cv2.imwrite(os.path.join(render_dir, f'frame{frame_index}_view{view_index}_pred_{psnr}_lr4_bbox.png'), (rgb_pred[..., [2, 1, 0]]*255))

        time_weight = float(out["time_weight_input"][0])
        cv2.imwrite(os.path.join(tw_dir, f'frame{frame_index}_view{view_index}_gt_{time_weight}.png'), (rgb_gt[..., [2, 1, 0]]*255))
        # cv2.imwrite(os.path.join(render_dir, f'frame{frame_index}_view{view_index}_input.png'), input_imgs[0])

        return scores

    # def validation_epoch_end(self, outputs):
    #     for key in outputs[0].keys():
    #         self.log(key, torch.stack([x[key] for x in outputs]).mean(), prog_bar=False)
    #     return

if __name__ == "__main__":
    import argparse
    import subprocess
    from tgs.utils.config import ExperimentConfig, load_config
    from tgs.data import CustomImageOrbitDataset
    from dataset_identity_mano_wild_4d_sam1 import Dataset, load_cfg

    from tgs.utils.misc import todevice, get_device

    parser = argparse.ArgumentParser("Triplane Gaussian Splatting")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--config_hand", default="vanerf_triplane.json", help="path to config file")
    parser.add_argument("--model_ckpt")
    parser.add_argument("--num_gpus", default=1)
    parser.add_argument("--out", default="outputs", help="path to output folder")
    parser.add_argument("--cam_dist", default=1.9, type=float, help="distance between camera center and scene center")
    parser.add_argument(
        "--run_val", action='store_true',
    )
    parser.add_argument(
        "--repose", action='store_true',
    )
    parser.add_argument(
        "--in_the_wild", action='store_true',
    )
    parser.add_argument("--image_preprocess", action="store_true", help="whether to segment the input image by rembg and SAM")
    args, extras = parser.parse_known_args()

    device = get_device()

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)
    from huggingface_hub import hf_hub_download
    

    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    
    # load configuration
    # parser = config.create_parser()
    # args = parser.parse_args(None)
    cfg_hand = config.load_cfg(args.config_hand)

    cfg_hand['expname'] = cfg_hand.get('expname', 'default')
    config.save_config(os.path.join(cfg_hand['out_dir'], cfg_hand['expname']), cfg_hand)

    # create model
    model = HandLightningModule.from_config(cfg_hand, cfg_hand.get('method', None), cfg)
    
    val_key = cfg_hand["training"].get("model_selection_metric", 'val_PSNR')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{cfg_hand["out_dir"]}/{cfg_hand["expname"]}/ckpts/',
        filename='model-{epoch:04d}-{%s:.4f}' % val_key,
        verbose=True,
        monitor=val_key,
        mode=cfg_hand["training"].get("model_selection_mode", 'max'),
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=True,
    )
    last_ckpt = os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")
    if not os.path.exists(last_ckpt):
        last_ckpt = None
    if args.model_ckpt is not None:  # overwrite last ckpt if specified model path
        last_ckpt = args.model_ckpt

    resume_from_checkpoint = cfg_hand.get('resume_from_checkpoint', last_ckpt)

    # create trainer
    logger = loggers.TestTubeLogger(
        save_dir=cfg_hand["out_dir"],
        name=cfg_hand['expname'],
        debug=False,
        create_git_tag=False
    )
    trainer = Trainer(
        max_epochs=cfg_hand["training"]["max_epochs"],
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=1)],
        resume_from_checkpoint=resume_from_checkpoint,
        logger=logger,
        gpus=args.num_gpus,
        num_sanity_val_steps=0,
        benchmark=True,
        detect_anomaly=True,
        gradient_clip_val=1.0,
        terminate_on_nan=False,
        accumulate_grad_batches=cfg_hand["training"].get("accumulate_grad_batches", 1),
        precision=32,
        # fast_dev_run=args.fast_dev_run,
        strategy="ddp" if args.num_gpus != 1 else None,
        **cfg_hand["training"].get('pl_cfg', {})
    )

    # run training
    if args.run_val and not args.in_the_wild:
        trainer.test(model, ckpt_path=resume_from_checkpoint, verbose=True)
    elif args.run_val and args.in_the_wild:
        trainer.test(model_in_the_wild, ckpt_path=resume_from_checkpoint, verbose=True)
    else:
        trainer.fit(model)
        model.save_ckpt()
    
    
    # print("ok2")
    # model.model.save_img_sequences(
    #     "video",
    #     "(\d+)\.png",
    #     save_format="mp4",
    #     fps=30,
    #     delete=True,
    # )
    # print("ok3")
