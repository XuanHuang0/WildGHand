import os
import sys
import pickle
import numpy as np
import json
from glob import glob
import os.path as osp
import torch
from pycocotools.coco import COCO
import cv2
import re


if __name__ == "__main__": 

    wild_path = '/home/huangx/OmniHands-main/demo_out/magic3/output.pkl'
    # wild_path = '/home/huangx/OmniHands-main/demo_out/Video/output.pkl'

    with open(wild_path, 'rb') as file:
        data_wild_video = pickle.load(file)

    data_wild_video = sorted(data_wild_video, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x['img_path'])).group(0)))

    #int(os.path.basename(x).split('.')[0])
    
    with open(wild_path, 'wb') as f:
        pickle.dump(data_wild_video, f)
    
