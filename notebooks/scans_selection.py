import sys
sys.path.append( '/Volumes/scratch/alegretelena/LLaVA-3D/open3dsg' )


import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from open3dsg.preprocess_3rscan import Preprocessor
from open3dsg.open_dataset import Open2D3DSGDataset
from open3dsg.get_object_frame import run, read_json


CONF_PATH_R3SCAN_RAW = '/Volumes/projects/open3dsg/data/3RScan/data'
CONF_PATH_R3SCAN_PROCESSED = '/Volumes/projects/open3dsg/output/datasets/OpenSG_3RScan'

def load_scan(base_path, file_path):
    return json.load(open(os.path.join(base_path, file_path)))["scans"]

def select_relationships(data_dict, threshold=10):
    selected_scenes = []
    scene_id = []
    for scene in data_dict:
        if len(scene['rel2frame_path']) >= 56:
            # Count the number of empty lists in rel2frame_path
            empty_count = sum(1 for value in scene['rel2frame_path'] if not value)
            if empty_count < threshold:
                selected_scenes.append(scene)
                scene_id.append(scene['scene_id'])

    return selected_scenes, scene_id

if __name__ == '__main__':
    # /mnt/scratch/LLaVA-3D/data/3rscan/relationships_train.json
    D3SSG = load_scan('/mnt/scratch/LLaVA-3D/data/3rscan/', 'relationships_train.json')
    
    dataset = Open2D3DSGDataset(
        relationships_R3SCAN=D3SSG,
        relationships_scannet=None,
        openseg=False,
        img_dim=224,
        rel_img_dim=224,
        top_k_frames=5,
        scales=3,
        mini=False,
        load_features=None,
        blip=True,
        llava=False,
        half=False,
        max_objects=9,
        max_rels=72
    )


    import torch

    if torch.cuda.is_available():
        print("GPU is available:", torch.cuda.get_device_name(0))
    else:
        print("GPU not available, running on CPU")

    data_dict, scene_id = select_relationships(dataset, 4)

    with open('scans_selections', "w") as fitxer:
        for element in scene_id:
            fitxer.write(f"{element}\n")  