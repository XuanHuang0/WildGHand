import os
import sys
import pickle
import numpy as np
import json
from glob import glob
import os.path as osp
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import trimesh
import smplx
import torch
from pycocotools.coco import COCO
import cv2
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from utils.transforms import world2cam, cam2pixel, pixel2cam, cam2world
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

# from pytorch3d.utils import ico_sphere
# from pytorch3d.structures import Meshes, join_meshes_as_scene
# from pytorch3d.renderer.mesh.textures import TexturesVertex
# from pytorch3d.renderer import (
#     BlendParams,
#     look_at_view_transform,
#     PerspectiveCameras, 
#     PointLights, 
#     RasterizationSettings, 
#     MeshRenderer, 
#     MeshRasterizer,  
#     SoftPhongShader,
#     TexturesVertex, 
#     SoftSilhouetteShader,
#     HardPhongShader
# )
import torch
from tqdm import tqdm
# from metaseg import SegManualMaskPredictor
from vis_util import draw_2d_skeleton
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

import argparse

# 使用 argparse 解析命令行输入
def parse_args():
    parser = argparse.ArgumentParser(description="Process mask.")
    parser.add_argument("--input_path", help="Input path.")
    return parser.parse_args()

sam_checkpoint = "/home/huangx/TriplaneGaussian_online/EXPERIMENTS/arxive/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device='cuda:7'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

mask_generator = SamAutomaticMaskGenerator(sam)

def draw_keypoints(img, kpts, color=(255, 0, 0), size=3):
    for i in range(kpts.shape[0]):
        kp2 = kpts[i].tolist()
        kp2 = (int(kp2[0]), int(kp2[1]))
        # print(img.shape)
        img = cv2.circle(cv2.UMat(img), kp2, 0, color, size)
    return img

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

@torch.no_grad()
def run(i, image, img_path, joints_uv, or_img_path, rendered_mask=None):
    # predictor = SegManualMaskPredictor()
    # for phase in ['train', 'progress']:
        # test_loader = create_dataloader(phase, subject=subject)

    output_folder = './vis_sam1_wild'
    os.makedirs(output_folder, exist_ok=True)
    
    print(joints_uv.shape) #(1558, 2)

    joints_uv_right = joints_uv[:21]
    min_joints = joints_uv_right.min(0)
    max_joints = joints_uv_right.max(0)
    print(min_joints)
    print(max_joints)
    bbox = [min_joints[0], min_joints[1], max_joints[0], max_joints[1]]
    print(bbox)
    center = (np.array(bbox[0:2]) + np.array(bbox[2:4])) / 2
    scale = 1.5
    x_length = (bbox[2] - bbox[0]) * scale
    y_length = (bbox[3] - bbox[1]) * scale
    bbox = [center[0] - x_length // 2, center[1] - y_length // 2, center[0] + x_length // 2, center[1] + y_length // 2]
    bbox_right = [int(i) for i in bbox]

    joints_uv_left = joints_uv[21:]
    min_joints = joints_uv_left.min(0)
    max_joints = joints_uv_left.max(0)
    print(min_joints)
    print(max_joints)
    bbox = [min_joints[0], min_joints[1], max_joints[0], max_joints[1]]
    print(bbox)
    center = (np.array(bbox[0:2]) + np.array(bbox[2:4])) / 2
    scale = 1.5
    x_length = (bbox[2] - bbox[0]) * scale
    y_length = (bbox[3] - bbox[1]) * scale
    bbox = [center[0] - x_length // 2, center[1] - y_length // 2, center[0] + x_length // 2, center[1] + y_length // 2]
    bbox_left = [int(i) for i in bbox]

    min_joints = joints_uv.min(0)
    max_joints = joints_uv.max(0)
    print(min_joints)
    print(max_joints)
    bbox = [min_joints[0], min_joints[1], max_joints[0], max_joints[1]]
    print(bbox)
    center = (np.array(bbox[0:2]) + np.array(bbox[2:4])) / 2
    scale = 1.5
    x_length = (bbox[2] - bbox[0]) * scale
    y_length = (bbox[3] - bbox[1]) * scale
    bbox = [center[0] - x_length // 2, center[1] - y_length // 2, center[0] + x_length // 2, center[1] + y_length // 2]
    bbox_inter = [int(i) for i in bbox]

    bbox_mask_right = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    x_min, y_min, bbox_width, bbox_height = bbox_right
    # 计算右下角的坐标
    x_max = x_min + bbox_width
    y_max = y_min + bbox_height
    # 在mask上绘制bbox对应的矩形区域，并设置该区域为1 (白色)
    cv2.rectangle(bbox_mask_right, (x_min, y_min), (x_max, y_max), color=1, thickness=-1)

    bbox_mask_left = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    x_min, y_min, bbox_width, bbox_height = bbox_left
    # 计算右下角的坐标
    x_max = x_min + bbox_width
    y_max = y_min + bbox_height
    # 在mask上绘制bbox对应的矩形区域，并设置该区域为1 (白色)
    cv2.rectangle(bbox_mask_left, (x_min, y_min), (x_max, y_max), color=1, thickness=-1)


    input_point = joints_uv.astype(int)
    input_point = input_point[[0,21]]
    input_label = np.ones(input_point.shape[0], dtype=int)
    save_path = os.path.join(output_folder, '{:>04d}.png'.format(i))

    print(input_point.shape)
    print(input_label.shape)

    if i < 10:
        save = True
    else:
        save = False 

    predictor.set_image(image.astype("uint8"))

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        # box=bbox,
        multimask_output=False,
    )
    
    print('masks.shape')
    print(masks.shape)


    masks_auto = mask_generator.generate(image.astype("uint8"))
    print(len(masks_auto)) #122
    print(masks_auto[0].keys()) #['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box']
    print(masks_auto[0]['segmentation'])

    # plt.figure(figsize=(20,20))
    # plt.imshow(image.astype("uint8"))
    # show_anns(masks_auto)
    # plt.axis('off')
    # plt.show() 

    # sam_results = predictor.image_predict(
    #     source=or_img_path,
    #     model_type="vit_l", 
    #     input_point=input_point,
    #     input_label=input_label,
    #     input_box=None,  # XYXY
    #     multimask_output=False,
    #     random_color=False,
    #     show=False,
    #     output_path=save_path,
    #     save=save,
    # )

    print(masks)
    new_mask = ((masks[0]>0.5) * 255)[:, :, None].astype("uint8")
    print(new_mask.shape)
    cv2.imwrite(save_path, new_mask)
    print(f'[INFO] Saving mask: {save_path}')

    # output_folder = './vis_sam2_wild'
    # os.makedirs(output_folder, exist_ok=True)
    image = image[...,[2,1,0]]

    keypoint=draw_keypoints(image.astype("uint8"), joints_uv.astype(int))


    mask_sam_img = ((new_mask>10)*(image)).astype("uint8")*255

    if rendered_mask is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        rendered_mask = cv2.dilate(rendered_mask*255, kernel, 4) # 更改迭代次数为2
        print(rendered_mask.shape)
        # rendered_mask = np.hstack((rendered_mask, dilate))
        rendered_mask_img = ((rendered_mask[...,None]>10)*(image)).astype("uint8")*255
        print(rendered_mask_img.shape)
        mask_p = ((new_mask>10)*(rendered_mask[...,None]>10)).astype("uint8")*255
        mask_p_img = ((mask_p>10)*(image)).astype("uint8")*255

        save_path_p = os.path.join(output_folder, '{:>04d}mask_p.png'.format(i))
        cv2.imwrite(save_path_p, mask_p)
        save_path_p = os.path.join(output_folder, '{:>04d}mask_p_img.png'.format(i))
        cv2.imwrite(save_path_p, mask_p_img)
        save_path_r = os.path.join(output_folder, '{:>04d}mask_render.png'.format(i))
        cv2.imwrite(save_path_r, rendered_mask)
        save_path_r = os.path.join(output_folder, '{:>04d}mask_render_img.png'.format(i))
        cv2.imwrite(save_path_r, rendered_mask_img)

    mask_b = ((new_mask>10)*(((bbox_mask_left[...,None])+(bbox_mask_right[...,None]))>0.9)).astype("uint8")*255
    mask_b_img = ((mask_b>10)*(image)).astype("uint8")*255


    # if i < 5000:
    save_path = os.path.join(output_folder, '{:>04d}mask_sam.png'.format(i))
    cv2.imwrite(save_path, new_mask)

    save_path_k = os.path.join(output_folder, '{:>04d}mask_k.png'.format(i))
    cv2.imwrite(save_path_k, keypoint)

    # save_path_p = os.path.join(output_folder, '{:>04d}mask_p.png'.format(i))
    # cv2.imwrite(save_path_p, mask_p)

    save_path_b = os.path.join(output_folder, '{:>04d}mask_b.png'.format(i))
    cv2.imwrite(save_path_b, mask_b)

    # save_path_r = os.path.join(output_folder, '{:>04d}mask_render.png'.format(i))
    # cv2.imwrite(save_path_r, rendered_mask)

    save_path = os.path.join(output_folder, '{:>04d}mask_sam_img.png'.format(i))
    cv2.imwrite(save_path, mask_sam_img)

    # save_path_p = os.path.join(output_folder, '{:>04d}mask_p_img.png'.format(i))
    # cv2.imwrite(save_path_p, mask_p_img)

    save_path_b = os.path.join(output_folder, '{:>04d}mask_b_img.png'.format(i))
    cv2.imwrite(save_path_b, mask_b_img)

    # save_path_r = os.path.join(output_folder, '{:>04d}mask_render_img.png'.format(i))
    # cv2.imwrite(save_path_r, rendered_mask_img)
        # print(f'[INFO] Saving debug image for SAM: {save_image}')

    # if i < 100:
    #     save_image = cv2.imread(save_path)
    #     save_image = draw_2d_skeleton(image, input_point)
    #     cv2.imwrite(save_path, save_image)
    #     print(f'[INFO] Saving debug image for SAM: {save_image}')

    return mask_b, np.array(bbox_inter), masks_auto

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


dense_path='./processed_dataset/v_color.pkl'
with open(dense_path, 'rb') as file:
    dense_coor = pickle.load(file)
dense_coor = torch.from_numpy(dense_coor)
dense_coor = torch.cat((dense_coor,dense_coor[-1,:].unsqueeze(0),dense_coor,dense_coor[-1,:].unsqueeze(0)), dim=0)

# def render_img(verts, faces, mesh_i_xy, mesh_i_z, R,T, fx, fy, px, py, mask_path='vis.jpg',image_size=(512,334),device='cuda:0'):

#     device = torch.device(device)
#     verts=torch.FloatTensor(verts).to(device)
#     faces = torch.tensor(faces).to(device)

#     mesh_i_xy=torch.FloatTensor(mesh_i_xy).to(device)
#     mesh_i_z=torch.FloatTensor(mesh_i_z).to(device)

#     znear, zfar = 0.71, 1.42
#     h,w=image_size
#     mesh_i_xy[..., 0] =  (mesh_i_xy[..., 0] / (w - 1.0)) 
#     mesh_i_xy[..., 1] =  (mesh_i_xy[..., 1] / (h - 1.0)) 
#     mesh_i_z =  (mesh_i_z - znear) / (zfar - znear)

#     if verts.shape[0] > 800:
#         v_color=(dense_coor.expand(*verts.shape)).to(device).unsqueeze(0)
#     else:
#         v_color=(dense_coor[:779].expand(*verts.shape)).to(device).unsqueeze(0)
#     tex = TexturesVertex(verts_features=v_color)
#     R=torch.FloatTensor(R).view(1, 3, 3).to(device)
#     T=torch.FloatTensor(T).view(1, 3).to(device)
#     mesh = Meshes(verts=[verts], faces=[faces], textures=tex)

#     cameras =  PerspectiveCameras(device=device,R=R,T=T,
#         focal_length=((fx, fy),),
#         principal_point=((px, py),),
#         in_ndc=False,
#         image_size=(image_size,),)

#     blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

#     raster_settings = RasterizationSettings(
#         image_size=image_size,
#         blur_radius=0.0,
#         faces_per_pixel=1
#     )

#     # Make an arbitrary light source
#     lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

#     # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
#     # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
#     # apply the Phong lighting model
#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(
#             cameras=cameras, 
#             raster_settings=raster_settings
#         ),
#         shader=HardPhongShader(
#             device=device, 
#             cameras=cameras,
#             lights=lights
#         )
#     )

#     blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
#     silhouette_renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(
#             cameras=cameras, 
#             raster_settings=raster_settings
#         ),
#         shader=SoftSilhouetteShader(blend_params=blend_params)
#     )
#     im_vis = renderer(mesh.to(device))
#     return im_vis.detach().cpu().numpy()[0, ..., :3]*255, im_vis.detach().cpu().numpy()[0, ..., -1]*255


def load_img(path, order='RGB'):
    
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def seal(mesh_to_seal,face_to_seal,hand_type):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    if hand_type=='left':
        circle_v_id=circle_v_id[::-1]
    center = (mesh_to_seal[circle_v_id, :]).mean(0)

    mesh_to_seal = np.vstack([mesh_to_seal, center])
    center_v_id = mesh_to_seal.shape[0] - 1

    # # pylint: disable=unsubscriptable-object # pylint/issues/3139
    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        face_to_seal = np.vstack([face_to_seal, new_faces])
    return mesh_to_seal, face_to_seal

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

def pad_image_to_square(image):
    height, width = image.shape[:2]
    # 计算需要填充的量
    if width > height:
        # 宽度大于高度，计算上下填充
        padding_top = (width - height) // 2
        padding_bottom = width - height - padding_top
        padded_image = cv2.copyMakeBorder(image, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    else:
        # 高度大于宽度，计算左右填充
        padding_left = (height - width) // 2
        padding_right = height - width - padding_left
        padded_image = cv2.copyMakeBorder(image, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT, value=[0,0,0])
    return new_image

if __name__ == "__main__": 
    # mano layer
    smplx_path = './smplx/models/'
    mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
    # vert_face=torch.load(os.path.join("/data4/huangx/KeypointNeRF/face.pth"))
    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:,0,:] *= -1

    args = parse_args()

    wild_path = args.input_path
    # '/home/huangx/OmniHands-main/demo_out/magic4/output.pkl'
    # wild_path = '/home/huangx/OmniHands-main/demo_out/Video/output.pkl'

    with open(wild_path, 'rb') as file:
        data_wild_video = pickle.load(file)

    i = 0

    idx = 0
    index = 0
    cam_list = {}

    print(len(data_wild_video))
    
    for acount, aid in tqdm(enumerate(data_wild_video)):
        data_wild = data_wild_video[acount]
        # print(data_wild.keys())
        # results_dict = data_wild['results_dict']
        # print(results_dict.keys())
        # mesh_rendering_orgimgs = results_dict['mesh_rendering_orgimgs']
        # print(mesh_rendering_orgimgs.keys())

        # pj2d = torch.cat([data_wild['joints2d_right'],data_wild['joints2d_left']],dim=0)
        pj2d_42 = np.concatenate([data_wild['joints2d_world_right'][0],data_wild['joints2d_world_left'][0]], axis=0)
        print(pj2d_42.shape)
        print(pj2d_42)
        # pj2d_42 = torch.cat([pj2d[0],pj2d[1]],dim=0)

        # input_msk_mano = mesh_rendering_orgimgs['valid_mask'][0].astype(np.uint8) #0为黑色
        # input_msk = input_msk_mano
        input_img = cv2.imread(data_wild['img_path'])

        # input_img_or = mesh_rendering_orgimgs['figs'][0]
        # print(input_img_or.shape)

        # pj2d_42 = ((pj2d_42+1)/2 * input_img.shape[1])
        # print(pj2d_42.shape)

        # H, W = int(256), int(256)
        # input_img, input_msk = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)
        # H, W = int(input_img.shape[0] ), int(input_img.shape[1])

        mask_sam_b, bbox_inter, masks_auto = run(acount, input_img, 'mask.jpg', pj2d_42,'mask.jpg')
        print('output')
        print(mask_sam_b.shape)
        print(bbox_inter.shape)

        mask_sam1_path = data_wild['img_path'].split('.')[-2]+'_sam1_mask.'+data_wild['img_path'].split('.')[-1]
        print(mask_sam1_path)
        cv2.imwrite(mask_sam1_path, mask_sam_b)

        data_wild_video[acount]['mask_sam1_path'] = mask_sam1_path
        data_wild_video[acount]['bbox_inter'] = bbox_inter
    
        # processed_data_path = '/home/huangx/processed_dataset'
        # os.makedirs((osp.join(processed_data_path ,'wild','index')), exist_ok=True)
        masks_auto_path = data_wild['img_path'].split('.')[-2]+'_sam1_mask_auto.pkl'
        with open(masks_auto_path, 'wb') as file:
            pickle.dump(masks_auto, file)
        data_wild_video[acount]['masks_auto_path'] = masks_auto_path

    data_wild_video = sorted(data_wild_video, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x['img_path'])).group(0)))

    with open(wild_path, 'wb') as f:
        pickle.dump(data_wild_video, f)
    
    #     image_id = ann['image_id']
    #     img = db.loadImgs(image_id)[0]
    #     capture_id = str(img['capture'])
    #     seq_name = str(img['seq_name'])
    #     cam_idx = str(img['camera'])
    #     cam=img['camera']
    #     hand_type = ann['hand_type']

    #     if int(cam) in occlusion_cam :
    #         continue  
    #     if str(cam)[:2]=='41':
    #         continue

    #     if hand_type == 'left':
    #         continue
    #     if hand_type == 'right':
    #         continue

    #     img_width, img_height = img['width'], img['height']

    #     bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
    #     bbox = process_bbox(bbox, (img_height, img_width))

    #     frame_idx = str(img['frame_idx'])
    #     img_path = osp.join(img_root_path, split, img['file_name'])
    #     save_path = osp.join(img_root_path, split+'mask', img['file_name']+'.npy')
    #     meshdir=os.path.split(save_path)[0]

    #     # if split == 'test':
    #     #     if 'ROM01_No_Interaction_2_Hand' in img['file_name']:
    #     #         continue

    #     save_path = meshdir
    #     frame_idx = img_path.split('/')[-1][5:-4]
    #     image = load_img(img_path)

    #     img_height, img_width, _ = image.shape

    #     prev_depth = None
    #     mesh={}
    #     face={}
    #     joint_world={}
    #     data_info = {}
    #     data_info['mano'] = {'pose':{},'shape':{},'trans':{}}
    #     if_continue = 0
    #     if hand_type == 'interacting':
    #         for hand in ('right', 'left'):
    #             try:
    #                 mano_param = mano_params[capture_id][frame_idx][hand]
    #                 if mano_param is None:
    #                     if_continue = 1
    #                     continue
    #             except KeyError:
    #                 if_continue = 1
    #                 continue

    #             # get MANO 3D mesh coordinates (world coordinate)
    #             mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
    #             root_pose = mano_pose[0].view(1,3)
    #             hand_pose = mano_pose[1:,:].view(1,-1)
    #             shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
    #             trans = torch.FloatTensor(mano_param['trans']).view(1,3)

    #             data_info['mano']['pose'][hand]=mano_pose
    #             data_info['mano']['shape'][hand]=shape
    #             data_info['mano']['trans'][hand]=trans

    #             output = mano_layer[hand](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    #             mesh[hand] = output.vertices[0].numpy() # meter to milimeter
    #             face[hand] = mano_layer[hand].faces
    #             # face = np.array(face, dtype=np.int32)
    #             joint_world[hand]= torch.matmul(joint_regressor, torch.tensor(mesh[hand]).float()).numpy()
    #             mesh[hand], face[hand]=seal(mesh[hand],face[hand],hand)
    #         try:
    #             mesh_i=np.concatenate([mesh['right'],mesh['left']],0)
    #             joint_i=np.concatenate([joint_world['right'], joint_world['left']],0)
    #         except KeyError:
    #             if_continue = 1
    #             continue

    #         mesh_right=trimesh.Trimesh(mesh['right'],face['right'])
    #         mesh_left=trimesh.Trimesh(mesh['left'],face['left'])
    #         mesh=concat_meshes([mesh_right,mesh_left])
    #         face=mesh.faces
    #         mesh_i=mesh.vertices
    #     else:  
    #         try:
    #             mano_param = mano_params[capture_id][frame_idx][hand_type]
    #             if mano_param is None:
    #                 if_continue = 1
    #                 continue
    #         except KeyError:
    #             if_continue = 1
    #             continue
    #         # get MANO 3D mesh coordinates (world coordinate)
    #         mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
    #         root_pose = mano_pose[0].view(1,3)
    #         hand_pose = mano_pose[1:,:].view(1,-1)
    #         shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
    #         trans = torch.FloatTensor(mano_param['trans']).view(1,3)

    #         data_info['mano']['pose'][hand_type]=mano_pose
    #         data_info['mano']['shape'][hand_type]=shape
    #         data_info['mano']['trans'][hand_type]=trans

    #         output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    #         mesh[hand_type] = output.vertices[0].numpy() # meter to milimeter
    #         face = mano_layer[hand_type].faces
    #         face = np.array(face, dtype=np.int32)
    #         joint_world[hand_type]= torch.matmul(joint_regressor, torch.tensor(mesh[hand]).float()).numpy()
    #         mesh[hand_type], face=seal(mesh[hand_type],face,hand_type)
    #         mesh_i=mesh[hand_type]
    #         joint_i=joint_world[hand_type]

        
    #     if if_continue == 1:
    #         continue

    #     data = {'idx':idx,'capture': capture_id, 'cam': cam, 'frame':frame_idx}
    #     data_info['aid'] = aid
    #     data_info['idx'] = idx
    #     data_info['image_info'] = img

    #     cam_param = cam_params[capture_id]
    #     focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
    #     princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
    #     t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3)/1000, np.array(cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3,3)
    #     t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
        
    #     mesh_i_cam = np.dot(R, mesh_i.transpose(1,0)).transpose(1,0) + t.reshape(1,3)        
    #     mesh_i_img = cam2pixel(mesh_i_cam, focal, princpt)
    #     mesh_i_xy = mesh_i_img[:, :2]
    #     mesh_i_z = mesh_i_img[:, 2:]

    #     joint_i_cam = np.dot(R, joint_i.transpose(1,0)).transpose(1,0) + t.reshape(1,3)        
    #     joint_i_img = cam2pixel(joint_i_cam, focal, princpt)
    #     joint_i_xy = joint_i_img[:, :2]

    #     print(joint_i_xy.shape)

    #     torch3d_T_colmap = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    #     R = (torch3d_T_colmap @ R).T
    #     t = torch3d_T_colmap @ t
        
    #     # densepose, rendered_mask=render_img(mesh_i, face, mesh_i_xy, mesh_i_z, R=R,T=t, fx=focal[0], fy=focal[1], px=princpt[0], py=princpt[1],image_size=(img_height, img_width))

    #     os.makedirs(osp.join(processed_data_path, split, 'densepose', 'capture'+str(capture_id),'cam'+str(cam)), exist_ok=True)
    #     os.makedirs(osp.join(processed_data_path, split, 'mask', 'capture'+str(capture_id),'cam'+str(cam)), exist_ok=True)
    #     os.makedirs(osp.join(processed_data_path, split, 'image', 'capture'+str(capture_id),'cam'+str(cam)), exist_ok=True)

    #     print('image.shape')
    #     print(image.shape)
    #     i = i+1
    #     mask_sam = run(i, image, processed_data_path+split+'/image/capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg', joint_i_xy, img_path)

    #     image,trans = augmentation(image,bbox)
    #     # densepose,trans = augmentation(densepose,bbox)
    #     # print(densepose.shape)
    #     # print(rendered_mask.shape)
    #     # rendered_mask,trans = augmentation(rendered_mask,bbox)
    #     print(mask_sam.shape) #(512, 334, 3)
    #     mask_sam,trans = augmentation(mask_sam.astype("uint8"),bbox)
    #     print(mask_sam.max())
    #     # mask_sam=FillHole((mask_sam[...,0]>50).astype(np.uint8), None, None)
    #     # mask_sam=(mask_sam.astype(np.uint8))*255
    #     mask_sam_img = ((mask_sam[...,None]>10)*(image)).astype("uint8")*255
    #     try:
    #         rendered_mask = cv2.imread(processed_data_path+split+'/mask/capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')
    #         print(rendered_mask.shape)
    #     except:
    #         continue
    #     kernel = np.ones((3, 3), dtype=np.uint8)
    #     rendered_mask = cv2.dilate(rendered_mask, kernel, 2) # 更改迭代次数为2
    #     # rendered_mask = np.hstack((rendered_mask, dilate))
    #     rendered_mask_img = ((rendered_mask>10)*(image)).astype("uint8")*255
    #     print(rendered_mask.shape)
    #     mask_p = ((mask_sam[...,None]>10)*(rendered_mask>10)).astype("uint8")*255
    #     mask_p_img = ((mask_p>10)*(image)).astype("uint8")*255

    #     output_folder = './vis_sam2'
    #     os.makedirs(output_folder, exist_ok=True)
    #     print('i')
    #     print(i)
    #     if i < 500:
    #         save_path = os.path.join(output_folder, '{:>04d}mask_sam.png'.format(i))
    #         cv2.imwrite(save_path, mask_sam)

    #         save_path_p = os.path.join(output_folder, '{:>04d}mask_p.png'.format(i))
    #         cv2.imwrite(save_path_p, mask_p)

    #         save_path_r = os.path.join(output_folder, '{:>04d}mask_render.png'.format(i))
    #         cv2.imwrite(save_path_r, rendered_mask)

    #         save_path = os.path.join(output_folder, '{:>04d}mask_sam_img.png'.format(i))
    #         cv2.imwrite(save_path, mask_sam_img)

    #         save_path_p = os.path.join(output_folder, '{:>04d}mask_p_img.png'.format(i))
    #         cv2.imwrite(save_path_p, mask_p_img)

    #         save_path_r = os.path.join(output_folder, '{:>04d}mask_render_img.png'.format(i))
    #         cv2.imwrite(save_path_r, rendered_mask_img)
    #         # print(f'[INFO] Saving debug image for SAM: {save_image}')

    #     img_save_path = processed_data_path+split+'/image/capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg'
    #     save_mask_path = img_save_path.replace('image', 'masks_SAM2')
    #     assert 'masks_SAM2' in save_mask_path
    #     os.makedirs(os.path.dirname(save_mask_path), exist_ok=True)
    #     try:
    #         cv2.imwrite(save_mask_path, mask_p)
    #         print(f'[INFO] Saving mask: {save_mask_path}')
    #     except:
    #         ptint(f'[INFO] Saving mask fail: {save_mask_path}')


    #     campos, camrot = np.array(cam_params[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cam_params[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
    #     focal, princpt = np.array(cam_params[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cam_params[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
    #     in_T, in_R = np.array(campos, dtype=np.float32).reshape(3)/1000., np.array(camrot, dtype=np.float32).reshape(3,3)
    #     in_T = -np.dot(in_R,in_T.reshape(3,1)).reshape(3) # -Rt -> t
    #     in_Rt = np.concatenate((in_R.reshape(3,3), in_T.reshape(3, 1)), axis=1)

    #     focal = trans_point2d(focal, trans)
    #     princpt = trans_point2d(princpt, trans)
    #     in_K=np.array([[focal[0],0,princpt[0]],[0,focal[1],princpt[1]],[0,0,1]])

    #     data_info['camera'] = {'R': in_R.reshape(3,3), 't': in_T.reshape(3, 1), 'in_K': in_K, 'campos':campos, 'camrot':camrot}

    #     # os.makedirs(osp.join(processed_data_path, split, 'annotation', 'capture'+str(capture_id)+'/cam'+str(cam)), exist_ok=True)
    #     # with open(osp.join(processed_data_path, split, 'annotation', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.pkl'), 'wb') as file:
    #     #     pickle.dump(data_info, file)

    #     if not frame_idx in cam_list.keys():
    #         cam_list[frame_idx]={}
    #     if capture_id not in cam_list[frame_idx].keys():
    #         cam_list[frame_idx][capture_id]=[]
    #     cam_list[frame_idx][capture_id].append((cam,aid))

    #     if frame_idx in cam_list.keys() and capture_id in cam_list[frame_idx].keys():
    #         if len(cam_list[frame_idx][capture_id])==4:
    #             with open(osp.join(processed_data_path ,split,'index', '{}.pkl'.format(index)), 'wb') as file:
    #                 pickle.dump(data, file)
    #             index=index+1
                
    #     image_save_path=processed_data_path+split+'/image/capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg'
    #     densepose_save_path=processed_data_path+split+'/densepose/capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg'
    #     mask_save_path=processed_data_path+split+'/mask/capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg'

    #     # cv2.imwrite(densepose_save_path, densepose)
    #     # cv2.imwrite(mask_save_path, rendered_mask)
    #     # cv2.imwrite(image_save_path, image[...,[2,1,0]])

    #     idx=idx+1
    #     # print(mask_save_path)
    # # torch.save(cam_list, os.path.join(processed_data_path, split, "cam_list.pth"))
