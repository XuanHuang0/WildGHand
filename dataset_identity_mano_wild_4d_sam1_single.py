import json
import cv2
import numpy as np
import os, sys
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from utils.preprocessing_wild import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
import imageio as imageio
import copy
from glob import glob
import os.path as osp
from mis_utils import edge_subdivide, read_mano_uv_obj
from PIL import Image, ImageDraw
import random
import json
# from pycocotools.coco import COCO
import scipy.io as sio
import smplx
from torchvision import transforms
import trimesh
vanerf_path="/home/huangx/vanerf"
# mano layer
smplx_path = vanerf_path+'/smplx/models/'
mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1

import torchvision.transforms as transforms
import pickle

def generate_mask(bbox, img_size):
    """
    根据边界框生成掩码
    
    Args:
    - bbox: 边界框 (x_min, y_min, x_max, y_max)
    - img_size: 图像尺寸 (height, width)
    
    Returns:
    - mask: 与图像尺寸相同的掩码图像
    """
    height, width = img_size
    mask = np.zeros((height, width), dtype=np.uint8)  # 创建一个全零的掩码

    # 填充边界框区域为1
    x_min, y_min, x_l, y_l = bbox
    print("bbox!!!!!!")
    print(bbox)
    x_max, y_max = x_min+x_l, y_min+y_l
    x_min, y_min = int(max([x_min,0.0])), int(max([y_min,0.0]))
    mask[y_min:y_max, x_min:x_max] = 1

    return mask

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

def FillHole(im_in,SavePath,savePathIn):
    # im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE);
    # cv2.imwrite(savePathIn,im_in*255)
    # 复制 im_in 图像
    im_floodfill = im_in.copy()
    
    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    seedPoint=(0,0)
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if(im_floodfill[i][j]==0):
                seedPoint=(i,j)
                isbreak = True
                break
        if(isbreak):
            break
    
    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask,seedPoint, 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv
    
    # 保存结果
    # cv2.imwrite(SavePath, im_out)
    return im_out

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, **kwargs):
        self.split = split # train, test, val
        self.mode = split # train, test
        self.sc_factor = 1
        if split == 'val':
            self.mode = 'test'
        self.mode = 'train'
        # if split == 'train':
        #     self.mode = 'test'
        self.nearmin = 2
        self.farmax = 0
        self.input_per_frame = kwargs.get('input_per_frame_test', 1)
        self.num_input_view = kwargs.get('num_input_view', 1)
        self.if_color_jitter=kwargs.get('color_jitter', False)
        self.if_mask_sa=kwargs.get('mask_sa', False)

        self.pose_sequence = kwargs.get('pose_sequence', None)

        self.stage = kwargs.get('stage', 1)

        self.if_render_mask=kwargs.get('render_mask', False)
        self.if_edge_subdivide=kwargs.get('edge_subdivide', False)
        self.if_edge_subdivide_hd=kwargs.get('edge_subdivide_hd', False)
        self.djd = kwargs.get('djd', False)
        self.add_mask = kwargs.get('add_mask', True)
        if self.djd:
            print('big angle test!!!!!!')
        if self.mode == 'train' and self.if_color_jitter:
            self.jitter = self.color_jitter()
        self.annot_path = vanerf_path+'/InterHand2.6M/annotations'
        joint_regressor = np.load(vanerf_path+'/smplx/models/mano/J_regressor_mano_ih26m.npy')
        self.joint_regressor=torch.tensor(joint_regressor)
        self.image2tensor = transforms.Compose([transforms.ToTensor(), ])
        self.vt, self.ft_l, self.ft_r, self.change_r, self.change_l= self.get_uvf()
        self.sequence_names = []
        self.cam_list=torch.load(os.path.join(vanerf_path+"/processed_dataset/",self.mode,"cam_list2.pth"))
        self.capture_frame_list=torch.load(os.path.join(vanerf_path+"/processed_dataset/",self.mode,"capture_frame_list.pth"))
        
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            self.joints = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot.json')) as f:
            self.manos = json.load(f)

        self.use_intag_preds = kwargs.get('use_intag_preds', False)
        self.repose = kwargs.get('repose', True)
        self.ratio=kwargs.get('ratio', 1)

        self.aux_cam_w2c = []
        self.aux_cam_intrinsic = []
        self.aux_cam = []

        self.wild_path  = kwargs.get('wild_path', '/home/huangx/OmniHands-main/demo_out/Video/output.pkl')
        # with open('/home/huangx/Arbitrary-Hands-3D-Reconstruction-main/demos_outputs/magic_results_0.35/wild.pkl/magic_handwild_0.35.pkl', 'rb') as file:
        if isinstance(self.wild_path, list):
            self.data_wild_video = []
            for path  in self.wild_path:
                with open(path, 'rb') as file:
                    self.data_wild_video.append(pickle.load(file))
        else:
            with open(self.wild_path, 'rb') as file:
                print(self.wild_path)
                self.data_wild_video = pickle.load(file)
                print(len(self.data_wild_video))
        # print(kwargs)
        if self.use_intag_preds:
            if self.mode=='train':
                # self.preds = torch.load(os.path.join('processed_dataset/verts_preds_train.pth'))
                print('using intaghand train preds !!!')
            else:
                # self.preds = torch.load(os.path.join('processed_dataset/verts_preds_test.pth'))
                print('using intaghand test preds !!!')

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
    
    def get_uvf(self):
        vt_r, ft_r, f_r = read_mano_uv_obj(vanerf_path+'/mano_uv/original mano template/hand.obj')  
        vt_l = vt_r
        vt_r=vt_r/2
        vt_l[...,0]=0.5+vt_l[...,0]/2
        vt_l[...,1]=vt_l[...,1]/2
        vt=np.concatenate((vt_r,vt_l))
        change_r = np.load(vanerf_path+"/change/change_r.npy")
        change_l=np.load(vanerf_path+'/change/change_l.npy', allow_pickle=True)
        face_left=np.load(vanerf_path+'/change/face_left.npy', allow_pickle=True)
        ft_l = face_left
        return vt, ft_l, ft_r, change_r, change_l

    def color_jitter(self):
        ops = []
        ops.extend(
            [transforms.ColorJitter(brightness=(0.2, 2),
                                    contrast=(0.3, 2), saturation=(0.2, 2),
                                    hue=(-0.5, 0.5)), ]
        )
        return transforms.Compose(ops)

    @staticmethod
    def get_mask_at_box(bounds, K, R, T, H, W):
        ray_o, ray_d = Dataset.get_rays(H, W, K, R, T)

        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = Dataset.get_near_far(bounds, ray_o, ray_d)
        # print(mask_at_box.shape)
        # print(near)
        # print(far)
        # return mask_at_box.reshape((H, W)),near.min(),far.max()
        return mask_at_box.reshape((H, W))

    
    def load_human_bounds_pred(self, vert_world_pred):
        
        xyz=vert_world_pred

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        # print(bounds)
        return bounds

    def load_human_bounds(self, capture_id,frame_idx, hand_type):
        # mano_valid=np.zeros((2,))
        # if hand_type == 'right' or hand_type == 'left':
            
        #     mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['pose']).view(-1,3)
        #     root_pose = mano_pose[0].view(1,3)
        #     hand_pose = mano_pose[1:,:].view(1,-1)
        #     shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['shape']).view(1,-1)
        #     trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['trans']).view(1,3)
        #     output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
        #     mesh = output.vertices[0].detach().numpy()
        #     if hand_type == 'left':
        #         mano_valid[1]=1
        #         mesh0=np.zeros((778,3))
        #         mesh=np.append(mesh0,mesh,axis=0) #(778*2,3)
        #     else:
        #         mano_valid[0]=1
        #         mesh0=np.zeros((778,3))
        #         mesh=np.append(mesh,mesh0,axis=0) #(778*2,3)
            
        # else:
        #     for hand in ('right', 'left'):
        #         try:
        #             mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['pose']).view(-1,3)
        #             root_pose = mano_pose[0].view(1,3)
        #             hand_pose = mano_pose[1:,:].view(1,-1)
        #             shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['shape']).view(1,-1)
        #             trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['trans']).view(1,3)
        #             output = mano_layer[hand](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
        #             mesh = output.vertices[0].detach().numpy()
        #             if hand == 'left':
        #                 mano_valid[1]=1
        #             else:
        #                 mano_valid[0]=1
        #         except:
        #             mesh=np.zeros((778,3))
        #             mano_pose=np.zeros((16,3))
        #             shape=np.zeros((1,10))
        #             trans=np.zeros((1,3))
        #         if hand == 'left':
        #             mesh_left=mesh
        #         else:
        #             mesh_right=mesh
        #     mesh=np.append(mesh_right,mesh_left,axis=0) #(778*2,3)

        xyz=mesh
        if hand_type == 'right':
            xyz=mesh[:778,:]
        elif hand_type == 'left':
            xyz=mesh[778:,:]

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        return bounds

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    # def big_pose_params(self):

    #     big_pose_params = {}
    #     # big_pose_params = copy.deepcopy(params)
    #     big_pose_params['R'] = np.ones((3,3)).astype(np.float32)
    #     big_pose_params['Th'] = np.zeros((1,3)).astype(np.float32)
    #     big_pose_params['shapes'] = np.zeros((1,10)).astype(np.float32)
    #     big_pose_params['poses'] = np.zeros((1,72)).astype(np.float32)
    #     # big_pose_params['poses'][0, 5] = 45/180*np.array(np.pi)
    #     # big_pose_params['poses'][0, 8] = -45/180*np.array(np.pi)
    #     # big_pose_params['poses'][0, 23] = -30/180*np.array(np.pi)
    #     # big_pose_params['poses'][0, 26] = 30/180*np.array(np.pi)

    #     return big_pose_params

    def load_mano2(self, capture_id,frame_idx, hand_type, t_pose_params=False):
        mano_valid=np.zeros((2,))
        if hand_type == 'right' or hand_type == 'left':
            
            mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['pose']).view(-1,3)

            mano_pose_right = torch.FloatTensor(self.manos[str(0)][str(11388)]['right']['pose']).view(-1,3)
            mano_pose_left = torch.FloatTensor(self.manos[str(0)][str(34127)]['left']['pose']).view(-1,3)
        
            root_pose = mano_pose[0].view(1,3)
            
            Rh = root_pose
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)

            hand_pose = mano_pose[1:,:].view(1,-1)

            if hand_type == 'right':
                hand_pose = mano_pose_right[1:,:].view(1,-1)
            else:
                hand_pose = mano_pose_left[1:,:].view(1,-1)

            shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['shape']).view(1,-1)
            trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['trans']).view(1,3)
            output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
            mesh = output.vertices[0].detach().numpy()
            face = output.faces[0].detach().numpy()

            
            Th = trans.astype(np.float32)

            mano_pose=mano_pose.view(1,-1)
            mano_pose=np.squeeze(mano_pose)
            shape=np.squeeze(shape)
            trans=np.squeeze(trans)
            
            mano_pose0=np.zeros(mano_pose.shape)
            shape0=np.zeros(shape.shape)
            trans0=np.zeros(trans.shape)

            if hand_type == 'left':
                mano_valid[1]=1
                mesh0=np.zeros((778,3))
                mesh=np.append(mesh0,mesh,axis=0) #(778*2,3)

                mano_pose=np.append(mano_pose0,mano_pose,axis=0)
                shape=np.append(shape0,shape,axis=0)
                trans=np.append(trans0,trans,axis=0)
            else:
                mano_valid[0]=1
                mesh0=np.zeros((778,3))
                mesh=np.append(mesh,mesh0,axis=0) #(778*2,3)

                mano_pose=np.append(mano_pose,mano_pose0,axis=0)
                shape=np.append(shape,shape0,axis=0)
                trans=np.append(trans,trans0,axis=0)
            
        else:
            for hand in ('right', 'left'):
                # # try:
                # mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['pose']).view(-1,3)
                # # print(mano_pose.shape) #[16, 3]
                # shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['shape']).view(1,-1)
                # trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['trans']).view(1,3)
                # print("str(frame_idx)")
                # print(str(frame_idx))
                # print(shape.shape) #[1, 10]
                if t_pose_params == True:
                    mano_pose = torch.FloatTensor(np.zeros((16,3)).astype(np.float32))
                    if hand == 'left':
                        trans = torch.FloatTensor(np.ones((1,3)).astype(np.float32))*0.5
                    else:
                        trans = torch.FloatTensor(np.zeros((1,3)).astype(np.float32))
                    shape = torch.FloatTensor(np.zeros((1,10)).astype(np.float32))

                root_pose = mano_pose[0].view(1,3)
                hand_pose = mano_pose[1:,:].view(1,-1)

                # print(self.manos[str(0)].keys())

                # if self.split == 'test':
                #     mano_pose_r = torch.FloatTensor(self.manos[str(0)][str(10279)]['right']['pose']).view(-1,3)
                #     # mano_pose_right = torch.FloatTensor(self.manos[str(0)][str(20625)]['right']['pose']).view(-1,3)
                #     mano_pose_l = torch.FloatTensor(self.manos[str(0)][str(34055)]['left']['pose']).view(-1,3)
                    
                #     if hand == 'right':
                #         hand_pose = mano_pose_r[1:,:].view(1,-1)
                #     else:
                #         hand_pose = mano_pose_l[1:,:].view(1,-1)

                # if hand == 'right':
                #     shape = shape*0.5
                # else:
                #     shape = shape*2

                Th = trans.numpy().astype(np.float32)
                Rh = root_pose.numpy()
                R = cv2.Rodrigues(Rh)[0].astype(np.float32)

                output = mano_layer[hand](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                mesh = output.vertices[0].detach().numpy()
                print("mesh")
                print(mesh.shape)
                face = mano_layer[hand].faces

                joint= torch.matmul(self.joint_regressor, torch.tensor(mesh).float())
                # mesh,face=seal(mesh,face,hand)
                if self.if_edge_subdivide:
                    mesh,face,_=edge_subdivide(mesh, face)
                    mesh,face,_=edge_subdivide(mesh, face)
                    if self.if_edge_subdivide_hd:
                        mesh,face,_=edge_subdivide(mesh, face)

                # if hand == 'right':
                #     xyz = np.dot(mesh - Th, R)
                # else:
                #     xyz = np.dot(mesh - Th_r, R_r)
                nxyz = np.zeros_like(mesh).astype(np.float32)

                if hand == 'left':
                    mano_valid[1]=1
                else:
                    mano_valid[0]=1
           
                if hand == 'left':
                    mesh_left=mesh
                    face_left=face
                    # face_left=np.load('change/face_left.npy', allow_pickle=True)
                    joint_left=joint
                   
                    mano_pose=mano_pose.reshape(1,-1)
                    mano_pose_left=np.squeeze(mano_pose)
                    shape_left=np.squeeze(shape)
                    trans_left=np.squeeze(trans)

                    xyz = np.dot(mesh - Th_r, R_r)
                    xyz_l=xyz
                    Rh_l=Rh
                    Th_l=Th
                    R_l=R
                    cxyz_l = xyz.astype(np.float32)
                    nxyz_l = nxyz.astype(np.float32)

                else:
                    mesh_right=mesh
                    face_right=face
                    joint_right=joint

                    mano_pose=mano_pose.reshape(1,-1)
                    mano_pose_right=np.squeeze(mano_pose)
                    shape_right=np.squeeze(shape)
                    trans_right=np.squeeze(trans)
                    
                    Rh_r=Rh
                    Th_r=Th
                    R_r=R
                    xyz = np.dot(mesh - Th, R)
                    xyz_r=xyz
                    
                    cxyz_r = xyz.astype(np.float32)
                    nxyz_r = nxyz.astype(np.float32)

            xyz = np.append(xyz_r,xyz_l,axis=-2)
            cxyz = np.append(cxyz_r,cxyz_l,axis=-2)
            nxyz = np.append(nxyz_r,nxyz_l,axis=-2)
            Rh = np.append(Rh_r,Rh_l,axis=-2)
            Th = np.append(Th_r,Th_l,axis=-2)
            feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

            # obtain the bounds for coord construction
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            min_xyz -= 0.05
            max_xyz += 0.05

            bounds = np.stack([min_xyz, max_xyz], axis=0)

            # construct the coordinate
            xyz=torch.from_numpy(xyz)
            # circle_v_id = torch.LongTensor([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120])
            # xyz_r=xyz[:778,:]
            # center = (xyz_r[circle_v_id, :]).mean(0).unsqueeze(0)
            # xyz_r = torch.cat([xyz_r, center],dim=0)
            # xyz_l=xyz[778:,:]
            # center = (xyz_l[circle_v_id, :]).mean(0).unsqueeze(0)
            # xyz_l = torch.cat([xyz_l, center],dim=0)
            # xyz=torch.cat([xyz_r, xyz_l],dim=0)
            xyz=xyz.numpy()

            dhw = xyz[:, [2, 1, 0]]
            min_dhw = min_xyz[[2, 1, 0]]
            max_dhw = max_xyz[[2, 1, 0]]
            voxel_size = np.array([0.005, 0.005, 0.005])
            coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

            # construct the output shape
            out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)

            x = 32
            out_sh = (out_sh | (x - 1)) + 1

            # mesh=np.append(mesh_right,mesh_left,axis=0) #(778*2,3)
            # face=np.append(face_right,face_left,axis=0) #(778*2,3)
            # verts=mesh
            # joint_right = torch.matmul(self.joint_regressor, torch.tensor(mesh_right).float())
            # joint_left = torch.matmul(self.joint_regressor, torch.tensor(mesh_left).float())
            joint_world=np.append(joint_right,joint_left,axis=0)

            mesh_right=trimesh.Trimesh(mesh_right,face_right)
            mesh_left=trimesh.Trimesh(mesh_left,face_left)
            mesh=concat_meshes([mesh_right,mesh_left])
            face=mesh.faces
            mesh=mesh.vertices

            vert_right=mesh_right.vertices
            vert_right_uv=vert_right[self.change_r.astype(int),:]
            vert_left=mesh_left.vertices
            vert_left_uv=vert_left[self.change_l.astype(int),:]

            mesh_right_uv=trimesh.Trimesh(vert_right_uv,self.ft_r, process=False)
            mesh_left_uv=trimesh.Trimesh(vert_left_uv,self.ft_l, process=False)
            mesh_uv=concat_meshes([mesh_right_uv,mesh_left_uv])
            face_uv=mesh_uv.faces
            vert_uv=mesh_uv.vertices

            mano_pose=np.append(mano_pose_right.unsqueeze(0),mano_pose_left.unsqueeze(0),axis=0)
            shape=np.append(shape_right.unsqueeze(0),shape_left.unsqueeze(0),axis=0)
            trans=np.append(trans_right.unsqueeze(0),trans_left.unsqueeze(0),axis=0)
            
            # obtain the original bounds for point sampling
            min_mesh = np.min(mesh, axis=0)
            max_mesh = np.max(mesh, axis=0)
            min_mesh -= 0.05
            max_mesh += 0.05
            can_bounds = np.stack([min_mesh, max_mesh], axis=0)

        # verts=mesh
        # joint_right = torch.matmul(self.joint_regressor, torch.tensor(verts[:778,:]).float())
        # joint_left = torch.matmul(self.joint_regressor, torch.tensor(verts[778:,:]).float())
        # joint_world=np.append(joint_right,joint_left,axis=0)

        return joint_world, mesh, face, face_uv, vert_uv, mano_pose, shape, trans, Rh_r, Th_r, R_r, coord, out_sh, can_bounds, bounds, feature

    def load_t_mano(self):
        mano_valid=np.zeros((2,))
        # if hand_type == 'right' or hand_type == 'left':

        #     mano_pose = torch.FloatTensor(np.zeros((1,2,48)).astype(np.float32)).view(-1,3)
        #     t_trans_left = torch.FloatTensor(np.ones((1,3)).astype(np.float32)).to(mano_pose.device)*0.5
        #     t_trans_right = torch.FloatTensor(np.zeros((1,3)).astype(np.float32)).to(mano_pose.device)
        #     t_mano_shape = torch.FloatTensor(np.zeros((1,2,10)).astype(np.float32)).to(mano_pose.device)

        #     root_pose = mano_pose[:,:,:3]
        #     hand_pose = mano_pose[:,:,3:]
            
        #     # mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['pose']).view(-1,3)
        #     # mano_pose_right = torch.FloatTensor(self.manos[str(0)][str(11388)]['right']['pose']).view(-1,3)
        #     # mano_pose_left = torch.FloatTensor(self.manos[str(0)][str(34127)]['left']['pose']).view(-1,3)
        
        #     root_pose = mano_pose[0].view(1,3)
            
        #     Rh = root_pose
        #     R = cv2.Rodrigues(Rh)[0].astype(np.float32)

        #     hand_pose = mano_pose[1:,:].view(1,-1)

        #     if hand_type == 'right':
        #         hand_pose = mano_pose_right[1:,:].view(1,-1)
        #     else:
        #         hand_pose = mano_pose_left[1:,:].view(1,-1)

        #     shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['shape']).view(1,-1)
        #     trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['trans']).view(1,3)
        #     output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
        #     mesh = output.vertices[0].detach().numpy()
        #     face = output.faces[0].detach().numpy()

            
        #     Th = trans.astype(np.float32)

        #     mano_pose=mano_pose.view(1,-1)
        #     mano_pose=np.squeeze(mano_pose)
        #     shape=np.squeeze(shape)
        #     trans=np.squeeze(trans)
            
        #     mano_pose0=np.zeros(mano_pose.shape)
        #     shape0=np.zeros(shape.shape)
        #     trans0=np.zeros(trans.shape)

        #     if hand_type == 'left':
        #         mano_valid[1]=1
        #         mesh0=np.zeros((778,3))
        #         mesh=np.append(mesh0,mesh,axis=0) #(778*2,3)

        #         mano_pose=np.append(mano_pose0,mano_pose,axis=0)
        #         shape=np.append(shape0,shape,axis=0)
        #         trans=np.append(trans0,trans,axis=0)
        #     else:
        #         mano_valid[0]=1
        #         mesh0=np.zeros((778,3))
        #         mesh=np.append(mesh,mesh0,axis=0) #(778*2,3)

        #         mano_pose=np.append(mano_pose,mano_pose0,axis=0)
        #         shape=np.append(shape,shape0,axis=0)
        #         trans=np.append(trans,trans0,axis=0)
            
        # else:
        for hand in ('right', 'left'):
            # # try:
            # mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['pose']).view(-1,3)
            # # print(mano_pose.shape) #[16, 3]
            # shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['shape']).view(1,-1)
            # trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['trans']).view(1,3)
            # print("str(frame_idx)")
            # print(str(frame_idx))
            # print(shape.shape) #[1, 10]
            # if t_pose_params == True:
            mano_pose = torch.FloatTensor(np.zeros((16,3)).astype(np.float32))
            if hand == 'left':
                trans = torch.FloatTensor(np.ones((1,3)).astype(np.float32))*0.5
            else:
                trans = torch.FloatTensor(np.zeros((1,3)).astype(np.float32))
            shape = torch.FloatTensor(np.zeros((1,10)).astype(np.float32))

            root_pose = mano_pose[0].view(1,3)
            hand_pose = mano_pose[1:,:].view(1,-1)

            # print(self.manos[str(0)].keys())

            # if self.split == 'test':
            #     mano_pose_r = torch.FloatTensor(self.manos[str(0)][str(10279)]['right']['pose']).view(-1,3)
            #     # mano_pose_right = torch.FloatTensor(self.manos[str(0)][str(20625)]['right']['pose']).view(-1,3)
            #     mano_pose_l = torch.FloatTensor(self.manos[str(0)][str(34055)]['left']['pose']).view(-1,3)
                
            #     if hand == 'right':
            #         hand_pose = mano_pose_r[1:,:].view(1,-1)
            #     else:
            #         hand_pose = mano_pose_l[1:,:].view(1,-1)

            # if hand == 'right':
            #     shape = shape*0.5
            # else:
            #     shape = shape*2

            Th = trans.numpy().astype(np.float32)
            Rh = root_pose.numpy()
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)

            output = mano_layer[hand](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
            mesh = output.vertices[0].detach().numpy()
            print("mesh")
            print(mesh.shape)
            face = mano_layer[hand].faces

            joint= torch.matmul(self.joint_regressor, torch.tensor(mesh).float())
            # mesh,face=seal(mesh,face,hand)
            if self.if_edge_subdivide:
                mesh,face,_=edge_subdivide(mesh, face)
                mesh,face,_=edge_subdivide(mesh, face)
                if self.if_edge_subdivide_hd:
                    mesh,face,_=edge_subdivide(mesh, face)

            # if hand == 'right':
            #     xyz = np.dot(mesh - Th, R)
            # else:
            #     xyz = np.dot(mesh - Th_r, R_r)
            nxyz = np.zeros_like(mesh).astype(np.float32)

            if hand == 'left':
                mano_valid[1]=1
            else:
                mano_valid[0]=1
        
            if hand == 'left':
                mesh_left=mesh
                face_left=face
                # face_left=np.load('change/face_left.npy', allow_pickle=True)
                joint_left=joint
                
                mano_pose=mano_pose.reshape(1,-1)
                mano_pose_left=np.squeeze(mano_pose)
                shape_left=np.squeeze(shape)
                trans_left=np.squeeze(trans)

                xyz = np.dot(mesh - Th_r, R_r)
                xyz_l=xyz
                Rh_l=Rh
                Th_l=Th
                R_l=R
                cxyz_l = xyz.astype(np.float32)
                nxyz_l = nxyz.astype(np.float32)

            else:
                mesh_right=mesh
                face_right=face
                joint_right=joint

                mano_pose=mano_pose.reshape(1,-1)
                mano_pose_right=np.squeeze(mano_pose)
                shape_right=np.squeeze(shape)
                trans_right=np.squeeze(trans)
                
                Rh_r=Rh
                Th_r=Th
                R_r=R
                xyz = np.dot(mesh - Th, R)
                xyz_r=xyz
                
                cxyz_r = xyz.astype(np.float32)
                nxyz_r = nxyz.astype(np.float32)

        xyz = np.append(xyz_r,xyz_l,axis=-2)
        cxyz = np.append(cxyz_r,cxyz_l,axis=-2)
        nxyz = np.append(nxyz_r,nxyz_l,axis=-2)
        Rh = np.append(Rh_r,Rh_l,axis=-2)
        Th = np.append(Th_r,Th_l,axis=-2)
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05

        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct the coordinate
        xyz=torch.from_numpy(xyz)
        # circle_v_id = torch.LongTensor([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120])
        # xyz_r=xyz[:778,:]
        # center = (xyz_r[circle_v_id, :]).mean(0).unsqueeze(0)
        # xyz_r = torch.cat([xyz_r, center],dim=0)
        # xyz_l=xyz[778:,:]
        # center = (xyz_l[circle_v_id, :]).mean(0).unsqueeze(0)
        # xyz_l = torch.cat([xyz_l, center],dim=0)
        # xyz=torch.cat([xyz_r, xyz_l],dim=0)
        xyz=xyz.numpy()

        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)

        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        # mesh=np.append(mesh_right,mesh_left,axis=0) #(778*2,3)
        # face=np.append(face_right,face_left,axis=0) #(778*2,3)
        # verts=mesh
        # joint_right = torch.matmul(self.joint_regressor, torch.tensor(mesh_right).float())
        # joint_left = torch.matmul(self.joint_regressor, torch.tensor(mesh_left).float())
        joint_world=np.append(joint_right,joint_left,axis=0)

        mesh_right=trimesh.Trimesh(mesh_right,face_right)
        mesh_left=trimesh.Trimesh(mesh_left,face_left)
        mesh=concat_meshes([mesh_right,mesh_left])
        face=mesh.faces
        mesh=mesh.vertices

        vert_right=mesh_right.vertices
        vert_right_uv=vert_right[self.change_r.astype(int),:]
        vert_left=mesh_left.vertices
        vert_left_uv=vert_left[self.change_l.astype(int),:]

        mesh_right_uv=trimesh.Trimesh(vert_right_uv,self.ft_r, process=False)
        mesh_left_uv=trimesh.Trimesh(vert_left_uv,self.ft_l, process=False)
        mesh_uv=concat_meshes([mesh_right_uv,mesh_left_uv])
        face_uv=mesh_uv.faces
        vert_uv=mesh_uv.vertices

        mano_pose=np.append(mano_pose_right.unsqueeze(0),mano_pose_left.unsqueeze(0),axis=0)
        shape=np.append(shape_right.unsqueeze(0),shape_left.unsqueeze(0),axis=0)
        trans=np.append(trans_right.unsqueeze(0),trans_left.unsqueeze(0),axis=0)
        
        # obtain the original bounds for point sampling
        min_mesh = np.min(mesh, axis=0)
        max_mesh = np.max(mesh, axis=0)
        min_mesh -= 0.05
        max_mesh += 0.05
        can_bounds = np.stack([min_mesh, max_mesh], axis=0)

        # verts=mesh
        # joint_right = torch.matmul(self.joint_regressor, torch.tensor(verts[:778,:]).float())
        # joint_left = torch.matmul(self.joint_regressor, torch.tensor(verts[778:,:]).float())
        # joint_world=np.append(joint_right,joint_left,axis=0)

        return joint_world, mesh, face, face_uv, vert_uv, mano_pose, shape, trans, Rh_r, Th_r, R_r, coord, out_sh, can_bounds, bounds, feature



    def __len__(self):
        if self.split=='train':
            # return 14952
            # if self.pose_sequence == 'interacting':
            #     return 14952
            if isinstance(self.data_wild_video[0], list):
                return 1000
            else:
                return int(len(self.data_wild_video)*0.8)
            # return 15
            # return 5
        elif self.split=='val':
            return int(len(self.data_wild_video)*0.9)-int(len(self.data_wild_video)*0.8)
        else:
            return len(self.data_wild_video)-int(len(self.data_wild_video)*0.9)

            # return int(len(self.data_wild_video)*0.95)-int(len(self.data_wild_video)*0.85)

            # return 1
    
    def __getitem__(self, index):

        aux_cam = None

        if isinstance(self.data_wild_video[0], list):
            index_path = random.randint(0,len(self.data_wild_video)-1)
            # index_path = 0
            print(index_path)
            data_wild_video = self.data_wild_video[index_path]
            if self.split=='train':
                index = random.randint(0,int(len(data_wild_video)*0.8)-1)
            # index = 21
        else:
            index_path = -1
            data_wild_video = self.data_wild_video
            print('data_wild_video')
            print(len(data_wild_video)) #417
            print(index)
        # print(self.data_wild_video[0].keys())
        
        if not self.split=='train':
            index = index + int(len(self.data_wild_video)*0.9)
            # index = index + int(len(self.data_wild_video)*0.85)

        # index = 708
        
        data_wild = data_wild_video[index]
        # print(data_wild.keys())
        # results_dict = data_wild['results_dict']
        # print(results_dict.keys())
        # mesh_rendering_orgimgs = results_dict['mesh_rendering_orgimgs']
        # print(mesh_rendering_orgimgs.keys())

        hand_type ='interacting'

        tar_cam={}  
        targets = {}
        input_imgs, input_msks, input_K, input_Rt = [], [], [], []
        input_R, input_t = [],[]
        idx = 1
  
        in_R = torch.Tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]).reshape(3,3).numpy()
        
        in_T = torch.Tensor([[0., 0., 0.]]).reshape(3).numpy()
        in_Rt = np.concatenate((in_R.reshape(3,3), in_T.reshape(3, 1)), axis=1)

        # in_K = anno['camera']['in_K']
        # princpt = np.array( [256, 256], dtype=np.float32) 
        # focal = np.array( [in_K[0, 0], in_K[1, 1]], dtype=np.float32)

        # campos = anno['camera']['campos']
        # camrot = anno['camera']['camrot']
        # img_info=anno['image_info']

        input_msk = cv2.imread(data_wild['mask_sam1_path'])
        # input_msk_mano = data_wild['mask_sam1'] #0为黑色
        input_msk = (input_msk>100).astype(np.uint8)


        # print("input_msk")
        # print(input_msk.max())
        # print(input_msk_mano.shape)
        # bbox_mask = input_msk_mano*0+1
        
        input_img = cv2.imread(data_wild['img_path']).astype(np.float32) / 255.
        input_img = input_img[...,[2,1,0]]

        print('input_path')
        print(data_wild['img_path'])

        focal = data_wild['scaled_focal_length']
        focal = np.array( [focal, focal], dtype=np.float32)
        # print("focal")
        # print(focal)
        princpt = np.array(input_img.shape[:2], dtype=np.float32)
        # focal = np.array( [focal, focal], dtype=np.float32)/princpt.max()*princpt
        princpt = princpt[[1,0]]
        princpt = princpt*0.5
        # print("princpt")
        # print(princpt)
        # input_img = mesh_rendering_orgimgs['org_imgs'][0].astype(np.float32) / 255.
        # input_img[...,1] =  input_img[...,0]
        # input_img[...,2] =  input_img[...,0]
        # cv2.imwrite('input_img.jpg',input_img*255)
        # print(input_img.shape)

        # with open(data_wild['masks_auto_path'], 'rb') as file:
        #     print(data_wild['masks_auto_path'])
        #     masks_auto = pickle.load(file)
        #     print(len(masks_auto))

        if 'bbox_inter' in data_wild.keys():
            bbox = data_wild['bbox_inter']
            # print(input_img.shape)
            # print(input_msk.shape)
            # print("bbox")
            # print(bbox)
            bbox[2] = (bbox[2] - bbox[0])
            bbox[3] = (bbox[3] - bbox[1])
            bbox = process_bbox(bbox, input_img.shape)

            # bbox_mask = generate_mask(bbox, (input_img.shape[0],input_img.shape[1]))
            bbox_mask = data_wild['mask_bbox']

            input_img,trans = augmentation(input_img,bbox)
            input_msk,trans = augmentation(input_msk,bbox)
            bbox_mask,trans = augmentation(bbox_mask,bbox)

            # cv2.imwrite('input_img.jpg',input_img*255)


            # for mask_auto in masks_auto:
            #     sam_mask = mask_auto['segmentation']
            #     sam_mask, trans = augmentation(sam_mask.astype("uint8"),bbox)
            #     mask_auto['segmentation'] = sam_mask
            

            # print(input_img.shape)
            # print(input_msk.shape)

            focal = trans_point2d(focal, trans)
            princpt = trans_point2d(princpt, trans)
            self.ratio = 1
        else:
            self.ratio = 0.5
        in_K=np.array([[focal[0],0,princpt[0]],[0,focal[1],princpt[1]],[0,0,1]])


        # H, W = int(256), int(256)
        # input_img, input_msk = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)
        H, W = int(input_img.shape[0] ), int(input_img.shape[1])

        input_img_all, input_msk_all= input_img, input_msk

        # bbox_mask = input_msk[...,0]*0+1


        # input_msk = (input_msk != 0)  # bool mask : foreground (True) background (False)

        if not 1. in input_msk:
            print('input_msk black!!!!!'+data_wild['img_path'])
        
        # if idx==0:
        #     campos0 = campos
        input_img0=input_img
        #     input_msk0 = input_msk
        input_msk = input_msk.astype(np.uint8) * 255
        input_img = self.image2tensor(input_img)
        input_msk = self.image2tensor(input_msk).bool()
        # input_msk_mano = self.image2tensor(input_msk_mano).bool()
        in_K0 = in_K.copy()
        in_K[:2] = in_K[:2] * self.ratio
        princpt = in_K[0:2, 2].astype(np.float32)
        focal = np.array( [in_K[0, 0], in_K[1, 1]], dtype=np.float32)
        
        # input_img[input_msk[0] == 0] = 0


            # bool mask : foreground (True) background (False)

        # if not 1. in input_msk_all:
        #     print('input_msk black!!!!!'+img_path)
        input_img_all = self.image2tensor(input_img_all)
        input_msk_all = self.image2tensor(input_msk_all).bool()

        input_msk_all = (input_msk_all != 0)
        
        if self.add_mask:
            input_img_all[input_msk_all == 0] = 0

        # params_dict = data_wild['params_dict']
        # verts = data_wild['verts'].cpu()
        # L, R = data_wild['left_hand_num'], data_wild['right_hand_num']
        # print("LR")
        # print(L)
        # print(R)

        # cam_trans = data_wild['cam_trans']
        cam_right = data_wild['cam_aligned_right']
        cam_left = data_wild['cam_aligned_left']
        cam_trans = np.array([cam_right[0], cam_left[0]])
        # print(cam_trans.shape)

        # l_vertices, l_joints, _ = mano_layer['left'](params_dict['poses'][:L], th_betas=params_dict['betas'][:L]) # if empty, return empty
        # r_vertices, r_joints, _ = mano_layer['right'](params_dict['poses'][L:L+R], th_betas=params_dict['betas'][L:L+R])

        mano_pose=np.append(data_wild['mano_pose_right'], data_wild['mano_pose_left'],axis=0)
        shape=np.append(data_wild['mano_shape_right'], data_wild['mano_shape_left'],axis=0)
        # print(mano_pose.shape)
        # print(shape.shape)
        # print(verts.shape)
        vert_right=data_wild['verts3d_world_right'][0] + cam_right[0]
        vert_right_uv=vert_right[self.change_r.astype(int),:]
        vert_left=data_wild['verts3d_world_left'][0] + cam_left[0]
        vert_left_uv=vert_left[self.change_l.astype(int),:]
        mesh_right_uv=trimesh.Trimesh(vert_right_uv,self.ft_r, process=False)
        mesh_left_uv=trimesh.Trimesh(vert_left_uv,self.ft_l, process=False)
        mesh_uv=concat_meshes([mesh_right_uv,mesh_left_uv])

        # mesh_uv.export('/home/huangx/TriplaneGaussian_online/mesh/mesh_uv'+str(index)+'l.obj')


        face_uv=mesh_uv.faces
        vert_uv=mesh_uv.vertices

        face_right = mano_layer['right'].faces
        face_left = mano_layer['left'].faces

        mesh_right=trimesh.Trimesh(vert_right,face_right)
        mesh_left=trimesh.Trimesh(vert_left,face_left)
        mesh=concat_meshes([mesh_right,mesh_left])
        face=mesh.faces
        mesh=mesh.vertices
        # mesh_or, face_or, face_uv_or, vert_uv_or = mesh, face, face_uv, vert_uv
        joint_world_or, mesh_or, face_or, face_uv_or, vert_uv_or, _, _, _, _, _, _, _, _, _, _, _=self.load_t_mano()

        if self.if_edge_subdivide:
            mesh,face,_=edge_subdivide(mesh, face)
            mesh,face,_=edge_subdivide(mesh, face)
            if self.if_edge_subdivide_hd:
                mesh,face,_=edge_subdivide(mesh, face)

        # print(mesh.shape)

        mesh_cam = np.dot(in_R, mesh.transpose(1,0)).transpose(1,0) + in_T.reshape(1,3)
        face_uv_xy_or=self.vt[face_uv_or,:]
        mesh_cam_or = np.dot(in_R, mesh_or.transpose(1,0)).transpose(1,0) + in_T.reshape(1,3)
        face_uv_xy=self.vt[face_uv,:]
    

        mesh_cam = np.dot(in_R, mesh.transpose(1,0)).transpose(1,0) + in_T.reshape(1,3)
        # joint_img = cam2pixel(mesh_cam, focal_s, princpt_s)[:, :2]
        # keypoint=draw_keypoints(input_img0*255, joint_img)
        # cv2.imwrite('/home/huangx/TriplaneGaussian-main/huo/huo_'+str(index)+'.jpg',input_img0[...,[1,2,0]]*255)
        # print('vert_intag.jpg')
        # if idx == 1:
        
        vert_world_pred_tar=mesh
        image = input_img
        torch3d_T_colmap = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        tar_R = (torch3d_T_colmap @ in_R).T
        tar_T = torch3d_T_colmap @ in_T
        tar_cam['input_R']=torch.from_numpy(tar_R).float()
        tar_cam['input_T']=torch.from_numpy(tar_T).float()
        tar_cam['input_focal']=torch.from_numpy(focal).float()
        tar_cam['input_princpt']=torch.from_numpy(princpt).float()
        tar_cam['input_K']=torch.from_numpy(in_K0).float()
        targets['input_mano_pose']=torch.from_numpy(mano_pose).float()
        targets['input_mano_shape']=torch.from_numpy(shape).float()
        # cam_trans[0], cam_trans[1] = cam_trans[1], cam_trans[0]
        cam_trans=torch.cat([torch.from_numpy(cam_trans[1:]), torch.from_numpy(cam_trans[0:1])],dim=0)
        targets['input_mano_trans']=cam_trans.float()

        targets['root_rel']=torch.from_numpy(data_wild['root_rel']).float()

        # if idx==0:
        

        targets.update({
                'vert_world':torch.from_numpy(mesh).float(),
                'vert_cam':torch.from_numpy(mesh_cam).float(),
                'vert_cam_or':torch.from_numpy(mesh_cam_or).float(),
                'face_world':torch.from_numpy(face).float(),
                'face_world_or':torch.from_numpy(face_or).float(),
                'vert_uv':torch.from_numpy(vert_uv).float(),
                'vert_uv_or':torch.from_numpy(vert_uv_or).float(),
                'face_uv_or':torch.from_numpy(face_uv_or).float(),
                'face_uv_xy_or':torch.from_numpy(face_uv_xy_or).float(),
                'face_uv':torch.from_numpy(face_uv).float(),
                # 'mesh_t':torch.from_numpy(mesh_t).float(),
                # 'face_t':torch.from_numpy(face_t).float(),
                'face_uv_xy':torch.from_numpy(face_uv_xy).float(),
        })

            # print(targets['input_uv'].shape)
            # print(targets['input_uv_mask'].shape)


        # if not self.use_intag_preds:
        #     targets['coord'] = coord
        #     targets['out_sh'] = out_sh
        #     targets['bounds'] = bounds

        # else:
        #     if self.use_intag_preds:
        #         vert_world_pred_input=vert_world_pred
            
        # targets["joint_world_input"]=torch.from_numpy(joint_world).float()
        targets["vert_world_input"]=torch.from_numpy(mesh).float()
        # targets['input_uv']=torch.from_numpy(input_uv).float().permute(2,0,1)
        targets['input_img_all']=input_img_all.float()
        targets['input_msk_all']=input_msk_all
        # targets['input_uv_mask']=torch.from_numpy(input_uv_mask).float().unsqueeze(0)
                    
                    
        # append data
        input_imgs.append(input_img)
        input_msks.append(input_msk)
        input_K.append(torch.from_numpy(in_K))
        input_Rt.append(torch.from_numpy(in_Rt))
        input_R.append(torch.from_numpy(in_R))
        input_t.append(torch.from_numpy(in_T))

        
        hand_type_array=self.handtype_str2array(hand_type)
        ret = {
            'images': torch.stack(input_imgs),
            'images_masks': torch.stack(input_msks),
            'K': torch.stack(input_K),
            'Rt': torch.stack(input_Rt),
            'hand_type':torch.from_numpy(hand_type_array),
            'i': index,
            'human_idx': int(index_path),
            'sessision': -1,
            'frame_index': index,
            'human': -1,
            'cam_ind': index,
            "index": {"camera": "cam", "segment": 'VANeRF', "tar_cam_id": index,
                "frame": f"{-1}_{index}", "ds_idx": index},
        }

        ret['targets']=targets
        ret['tar_cam']=tar_cam
        ret["txt"]= "interacting hands with black background"
        
        # if self.use_intag_preds:
        bounds = self.load_human_bounds_pred(vert_world_pred_tar)
        # else:
        #     if self.repose:
        #         bounds = self.load_human_bounds(capture_id,tar_frame, hand_type)
        #     else:
        #         bounds = self.load_human_bounds(capture_id,frame_idx, hand_type)
        ret['mask_at_box'] = self.get_mask_at_box(
            bounds,
            input_K[0].numpy(),
            input_Rt[0][:3, :3].numpy(),
            input_Rt[0][:3, -1].numpy(),
            H, W)
        # ret['znear'], ret['zfar']=near,far
        # print([near,far])
        # if near<self.nearmin:
        #     self.nearmin=near
        #     # print(self.nearmin)
        # if far>self.farmax:
        #     self.farmax=far  
        #     # print(self.farmax)
        ret['bounds'] = bounds
        ret['mask_at_box'] = ret['mask_at_box'].reshape((H, W))
        x, y, w, h = cv2.boundingRect(ret['mask_at_box'].astype(np.uint8))

        # mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)]['right']['pose']).view(-1,3)
        root_pose = (torch.FloatTensor(mano_pose).view(-1,3))[0].reshape(-1)
        
        Rh = root_pose.numpy()
        # print(Rh.shape)
        R,_ = cv2.Rodrigues(Rh)
        R = torch.from_numpy(R)


        # print(index)
        input_img_all = targets['input_img_all'].permute(1,2,0)
        # tar_img_all = targets['tar_img_all'].permute(1,2,0)

        input_msk_all = targets['input_msk_all'].permute(1,2,0)
        # tar_msk_all = targets['tar_msk_all'].permute(1,2,0)

        ray_o, ray_d = Dataset.get_rays(H, W,
            input_K[0].numpy(),
            input_Rt[0][:3, :3].numpy(),
            input_Rt[0][:3, -1].numpy())
        ray_o = torch.from_numpy(ray_o.astype(np.float32))
        ray_d = torch.from_numpy(ray_d.astype(np.float32))
        view_index=torch.as_tensor([0])

        cond_c2w: Float[Tensor, "4 4"] = torch.cat(
            [input_Rt[0], torch.zeros_like(input_Rt[0][:1])], dim=0
        )
        cond_c2w[3, 3] = 1.0

        world_view_transform = torch.tensor(getWorld2View2(input_R[0].numpy(), input_t[0].numpy())).transpose(0, 1)

        cond_w2c = cond_c2w
        cond_c2w=torch.inverse(cond_c2w)
        # campos=campos*0.001
       
        cond_c2w_tar: Float[Tensor, "4 4"] = torch.cat(
            [input_Rt[0], torch.zeros_like(input_Rt[0][:1])], dim=0
        )
        cond_c2w_tar[3, 3] = 1.0

        cond_w2c_tar = cond_c2w_tar
        cond_c2w_tar=torch.inverse(cond_c2w_tar)


        intrinsic_normed_cond = input_K[0].clone()


        intrinsic_normed_cond[..., 0, 2] /= W
        intrinsic_normed_cond[..., 1, 2] /= H
        intrinsic_normed_cond[..., 0, 0] /= W
        intrinsic_normed_cond[..., 1, 1] /= H

        intrinsic_normed_cond_tar = input_K[0].clone()
        intrinsic_normed_cond_tar[..., 0, 2] /= W
        intrinsic_normed_cond_tar[..., 1, 2] /= H
        intrinsic_normed_cond_tar[..., 0, 0] /= W
        intrinsic_normed_cond_tar[..., 1, 1] /= H

        input_img_all = input_img_all[
            :, :, :3
        ]

        out = {
            "rgb_cond": input_img_all.float().unsqueeze(0), #
            # "masks_auto": masks_auto,
            "mask": input_msk_all.float().unsqueeze(0), #
            "tar_img": input_img_all.float().unsqueeze(0),
            "input_R": input_R[0].unsqueeze(0),
            "input_t": input_t[0].unsqueeze(0),
            "tar_R": input_R[0].unsqueeze(0),
            "tar_t": input_t[0].unsqueeze(0),
            "world_view_transform": world_view_transform,
            "c2w_cond": cond_c2w.unsqueeze(0), #
            "w2c_cond": cond_w2c.unsqueeze(0), #
            "mask_cond": input_msk_all.unsqueeze(0), #
            "tar_msk_all":input_msk_all.unsqueeze(0),
            "intrinsic_cond": input_K[0].unsqueeze(0).float(), #
            "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0).float(), #
            "view_index": torch.as_tensor([0]), #
            # "rays_o": ray_o.unsqueeze(0), #
            # "rays_d": ray_d.unsqueeze(0), #
            'bbox_mask' : bbox_mask,
            "intrinsic": input_K[0].unsqueeze(0).float(), #
            "intrinsic_normed": intrinsic_normed_cond.unsqueeze(0).float(), #
            "c2w": cond_c2w.unsqueeze(0).float(), #
            "w2c": cond_w2c.unsqueeze(0).float(), #
            # "camera_positions": torch.as_tensor(campos0).unsqueeze(0).float(), #
            "points":torch.from_numpy(mesh).float(),
            "points_tar":torch.from_numpy(mesh).float(),

            "ret":ret,
            "total_frame":torch.as_tensor([len(self.data_wild_video)]),
            "index":torch.as_tensor([index])
            # "znear":near,
            # "zfar":far
        }

        out["if_aux_cam"] = False

        # out["c2w"][..., :3, 1:3] *= -1
        # out["w2c_cond"][..., :3, 1:3] *= -1

        # instance_id = os.path.split(img_path)[-1].split('.')[0]
        out["index"] = torch.as_tensor(index)
        out["background_color"] = torch.as_tensor([0., 0., 0.]) #
        # out["instance_id"] = instance_id
        aux_index = random.randint(0,13)
        print(random.randint(0,1))
        # if aux_cam is not None and random.randint(0,1) and self.split is not 'test' and self.stage is 2:
        if aux_cam is not None and random.randint(0,10)==1 and self.split is not 'test' and self.stage is 2:

        # if aux_cam is not None and 0 and self.split is not 'test':
        # if aux_cam is not None and 1:
            print("aux!!!!!!!!")
            aux_cam_ = aux_cam[aux_index]
            out["aux_cam"] = aux_cam_


            # aux_img_path = "/home/huangx/processed_dataset/test/aux_cam_img/"+str(int(aux_index))+".jpg"
            aux_img_path = "/home/huangx/processed_dataset/test/aux_cam_img_nmap/"+str(int(aux_index))+".jpg"

            aux_img = imageio.imread(aux_img_path)
            aux_img = aux_img.astype(np.float32) / 255.
            aux_img = self.image2tensor(aux_img)
            out["aux_img"] = aux_img

        if self.pose_sequence == 'oneshot_reg_i' and self.split is 'test':
            self.aux_cam_w2c.append(out["w2c_cond"])
            self.aux_cam_intrinsic.append(out["intrinsic_cond"])
            self.aux_cam.append([out["w2c_cond"], out["intrinsic_cond"], out['c2w_cond'], out["intrinsic_normed_cond"]])

            print(index)
            if index == 13:
                # os.makedirs(osp.join(processed_data_path, split, 'index_identity_os_reg_i'), exist_ok=True)
                print(self.aux_cam)
                with open('/home/huangx/processed_dataset/test/aux_cam_img/index_identity_os_reg_i.pkl', 'wb') as file:
                    pickle.dump(self.aux_cam, file)
        return out

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": 256, "width": 256})
        return batch

    @classmethod
    def from_config(cls, dataset_cfg, data_split, cfg):
        ''' Creates an instance of the dataset.

        Args:
            dataset_cfg (dict): input configuration.
            data_split (str): data split (`train` or `val`).
        '''
        assert data_split in ['train', 'val', 'test', 'test_visualize']

        dataset_cfg = copy.deepcopy(dataset_cfg)
        dataset_cfg['is_train'] = data_split == 'train'
        if f'{data_split}_cfg' in dataset_cfg:
            dataset_cfg.update(dataset_cfg[f'{data_split}_cfg'])
        if dataset_cfg['is_train']:
            dataset = cls(split=data_split, **dataset_cfg)
        elif data_split == 'test_visualize':
            # skip every 6th data sample (there are 6 cameras per person)
            dataset = TestDataset(split='test', sample_frame=1, sample_camera=6, **dataset_cfg)
        else:
            dataset = TestDataset(split=data_split, **dataset_cfg)
        return dataset

    @staticmethod
    def get_rays(H, W, K, R, T):
        rays_o = -np.dot(R.T, T).ravel()

        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32), indexing='xy')

        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        try:
            pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        except:
            pixel_camera = np.dot(xy1, np.linalg.pinv(K).T)

        pixel_world = np.dot(pixel_camera - T.ravel(), R)
        rays_d = pixel_world - rays_o[None, None]
        rays_o = np.broadcast_to(rays_o, rays_d.shape)

        return rays_o, rays_d

    @staticmethod
    def get_near_far(bounds, ray_o, ray_d, boffset=(-0.01, 0.01)):
        """calculate intersections with 3d bounding box"""
        bounds = bounds + np.array([boffset[0], boffset[1]])[:, None]
        nominator = bounds[None] - ray_o[:, None]
        # calculate the step of intersections at six planes of the 3d bounding box
        ray_d[np.abs(ray_d) < 1e-5] = 1e-5
        d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
        # calculate the six interections
        p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
        # calculate the intersections located at the 3d bounding box
        min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
        eps = 1e-6
        p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                        (p_intersect[..., 0] <= (max_x + eps)) * \
                        (p_intersect[..., 1] >= (min_y - eps)) * \
                        (p_intersect[..., 1] <= (max_y + eps)) * \
                        (p_intersect[..., 2] >= (min_z - eps)) * \
                        (p_intersect[..., 2] <= (max_z + eps))
        # obtain the intersections of rays which intersect exactly twice
        mask_at_box = p_mask_at_box.sum(-1) == 2
        p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
            -1, 2, 3)

        # calculate the step of intersections
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        norm_ray = np.linalg.norm(ray_d, axis=1)
        d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
        d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
        near = np.minimum(d0, d1)
        far = np.maximum(d0, d1)

        return near, far, mask_at_box

def draw_keypoints(img, kpts, color=(255, 0, 0), size=3):
    for i in range(kpts.shape[0]):
        kp2 = kpts[i].tolist()
        kp2 = [int(kp2[0]), int(kp2[1])]
        img = cv2.circle(img, kp2, 0, color, size)
    return img

class TestDataset(Dataset):
    def __init__(self, split, sample_frame=30, sample_camera=1, **kwargs):
        super().__init__( split, **kwargs)

def load_cfg(path):
    """ Load configuration file.
    Args:
        path (str): model configuration file.
    """
    if path.endswith('.json'):
        with open(path, 'r') as file:
            cfg = json.load(file)
    elif path.endswith('.yml') or path.endswith('.yaml'):
        with open(path, 'r') as file:
            cfg = yaml.safe_load(file)
    else:
        raise ValueError('Invalid config file.')

    return cfg

if __name__ == "__main__":
    cfg_path = "/home/huangx/TriplaneGaussian_online/online_shade_mano_pose_wild_time_tex_4d_fenli_loss_mano_sam_res_lmask_magic_t_reg_sym_res1_sam1_vtreg_lr_c0_x_50_tw_tsp10_single.json"
    cfg = load_cfg(cfg_path)
    dataset = Dataset.from_config(cfg['dataset'], 'train', cfg)
    # dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)
    for i in tqdm(range(dataset.__len__())):
        dataset.__getitem__(i)
