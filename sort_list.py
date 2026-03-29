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

PROJECT_ROOT = osp.dirname(osp.abspath(__file__))

if __name__ == "__main__": 

    wild_path = osp.join(PROJECT_ROOT, 'dataset', 'WildGHand', 'capture0_subsample3_single', 'output.pkl')

    with open(wild_path, 'rb') as file:
        data_wild_video = pickle.load(file)

    data_wild_video = sorted(data_wild_video, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x['img_path'])).group(0)))

    #int(os.path.basename(x).split('.')[0])
    
    with open(wild_path, 'wb') as f:
        pickle.dump(data_wild_video, f)
    
