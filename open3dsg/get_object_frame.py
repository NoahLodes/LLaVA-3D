# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import argparse
import os
import json
import pickle
import multiprocessing
from functools import partial
import trimesh
import cv2
import torch
import numpy as np
from tqdm import tqdm
from const import CONF_PATH_R3SCAN_PROCESSED, CONF_PATH_R3SCAN_RAW
import matplotlib.pyplot as plt

lock = multiprocessing.Lock()

def read_intrinsic(intrinsic_path, mode='rgb'):
    with open(intrinsic_path, "r") as f:
        data = f.readlines()

    m_versionNumber = data[0].strip().split(' ')[-1]
    m_sensorName = data[1].strip().split(' ')[-2]

    if mode == 'rgb':
        m_Width = int(data[2].strip().split(' ')[-1])
        m_Height = int(data[3].strip().split(' ')[-1])
        m_Shift = None
        m_intrinsic = np.array([float(x) for x in data[7].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    else:
        m_Width = int(float(data[4].strip().split(' ')[-1]))
        m_Height = int(float(data[5].strip().split(' ')[-1]))
        m_Shift = int(float(data[6].strip().split(' ')[-1]))
        m_intrinsic = np.array([float(x) for x in data[9].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))

    m_frames_size = int(float(data[11].strip().split(' ')[-1]))

    return dict(
        m_versionNumber=m_versionNumber,
        m_sensorName=m_sensorName,
        m_Width=m_Width,
        m_Height=m_Height,
        m_Shift=m_Shift,
        m_intrinsic=np.matrix(m_intrinsic),
        m_frames_size=m_frames_size
    )


def read_txt_to_list(file):
    output = []
    with open(file, 'r') as f:
        for line in f:
            entry = line.rstrip().lower()
            output.append(entry)
    return output

def read_extrinsic(extrinsic_path):
    pose = []
    with open(extrinsic_path) as f:
        lines = f.readlines()
    for line in lines:
        pose.append([float(i) for i in line.strip().split(' ')])
    return pose


def get_label(label_path):
    label_list = []
    with open(label_path, "r") as f:
        for line in f:
            label_list.append(line.strip())
    return label_list


def read_pointcloud_R3SCAN(scan_id):
    """
    Reads a pointcloud from a file and returns points with instance label.
    """
    plydata = trimesh.load(os.path.join(CONF_PATH_R3SCAN_RAW, scan_id, 'labels.instances.annotated.v2.ply'), process=False)
    points = np.array(plydata.vertices)
    labels = np.array(plydata.metadata['_ply_raw']['vertex']['data']['objectId'])

    return points, labels


def read_json(root, split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    selected_scans = selected_scans.union(read_txt_to_list(os.path.join(root, f'{split}_scans.txt')))
    with open(os.path.join(root, f"relationships_{split}.json"), "r") as read_file:
        data = json.load(read_file)

    # convert data to dict('scene_id': {'obj': [], 'rel': []})
    scene_data = dict()
    for i in data['scans']:
        if i['scan'] in selected_scans:
            if i['scan'] not in scene_data.keys():
                scene_data[i['scan']] = {'obj': dict(), 'rel': list()}
            scene_data[i['scan']]['obj'].update(i['objects'])
            scene_data[i['scan']]['rel'].extend(i['relationships'])

    return scene_data, selected_scans

####################################################


def read_scan_info_R3SCAN(scan_id, mode='depth'):
    scan_path = os.path.join(CONF_PATH_R3SCAN_RAW, scan_id)
    sequence_path = os.path.join(scan_path, "sequence")
    intrinsic_path = os.path.join(sequence_path, "_info.txt")
    intrinsic_info = read_intrinsic(intrinsic_path, mode=mode)
    # mode_template = 'color.jpg' if mode == 'rgb' else 'depth.pgm'

    image_list, color_list, extrinsic_list, frame_paths = [], [], [], []

    for i in range(0, intrinsic_info['m_frames_size'], 10):
        frame_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6) + 'depth.pgm')
        color_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6) + 'color.jpg')
        frame_paths.append("frame-%s." % str(i).zfill(6) + 'color.jpg')
        extrinsic_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6) + "pose.txt")
        assert os.path.exists(frame_path) and os.path.exists(extrinsic_path)

        color_list.append(np.array(plt.imread(color_path)))

        image_list.append(cv2.imread(frame_path, -1).reshape(-1))
        # inverce the extrinsic matrix, from camera_2_world to world_2_camera
        extrinsic = np.matrix(read_extrinsic(extrinsic_path))
        extrinsic_list.append(extrinsic)

    return np.array(image_list), np.array(color_list), np.array(extrinsic_list), intrinsic_info, frame_paths

####################################################


def compute_mapping(world_to_camera, coords, depth, intrinsic, cut_bound, vis_thres, image_dim):
    mapping = np.zeros((3, coords.shape[0]), dtype=int)
    coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
    assert coords_new.shape[0] == 4, "[!] Shape error"

    p = np.matmul(world_to_camera, coords_new)
    p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
    p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
    z = p[2].copy()
    pi = np.round(p).astype(int)  # simply round the projected coordinates
    inside_mask = (pi[0] >= cut_bound) * (pi[1] >= cut_bound) \
        * (pi[0] < image_dim[0]-cut_bound) \
        * (pi[1] < image_dim[1]-cut_bound)
    depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
    occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                            - p[2][inside_mask]) <= \
        vis_thres * depth_cur

    inside_mask[inside_mask == True] = occlusion_mask
    mapping[0][inside_mask] = pi[1][inside_mask]
    mapping[1][inside_mask] = pi[0][inside_mask]
    mapping[2][inside_mask] = 1

    return mapping


def image_3d_mapping(scan, image_list, color_list, img_names, point_cloud, instances, extrinsics, intrinsics, instance_names, image_width, image_height, boarder_pixels=0, vis_tresh=0.05, scene_data=None):
    object2frame = dict()

    squeezed_instances = instances.squeeze()
    image_dim = np.array([image_width, image_height])
    for i, (extrinsic, depth, color) in enumerate(zip(extrinsics, image_list, color_list)):
        world_to_camera = np.linalg.inv(extrinsic)
        depth = depth.reshape(image_dim[::-1])/1000
        for inst in instance_names.keys():
            locs_in = point_cloud[squeezed_instances == int(inst)]
            mapping = compute_mapping(world_to_camera, locs_in, depth, np.array(intrinsics), boarder_pixels, vis_tresh, image_dim).T
            homog_points = mapping[:, 2] == 1
            ratio = (homog_points).sum()/mapping.shape[0]
            pixels = mapping[:, -1].sum()
            if pixels > 12 and ((ratio > 0.3 or pixels > 80) or (instance_names[inst] in ['wall', 'floor'] and pixels > 80)):
                if inst not in object2frame:
                    object2frame[inst] = []
                obj_points = mapping[homog_points]
                unique_mapping = np.unique(mapping[homog_points][:, :2], axis=0).astype(np.uint16)
                
                object2frame[inst].append((img_names[i], pixels, ratio, (obj_points[:, 1].min(), obj_points[:, 0].min(
                ), obj_points[:, 1].max(), obj_points[:, 0].max()), unique_mapping))

    return object2frame

def run(scan, scene_data, export_path):
    export_dir = export_path
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=False)

    output_filepath = os.path.join(export_dir, f"{scan}_object2image.pkl")
    if os.path.exists(output_filepath):
        # print('path already exists: ', output_filepath)
        return
    instance_names = scene_data[scan]['obj']
    pc_i, instances_i = read_pointcloud_R3SCAN(scan)
    image_list, color_list, extrinsic_list, intrinsic_info, img_names = read_scan_info_R3SCAN(scan)
    #intrinsic_info['m_Width'], intrinsic_info['m_Height'] = 320, 240

    object2frame = image_3d_mapping(scan, image_list, color_list, img_names, pc_i, instances_i, extrinsic_list,
                                    intrinsic_info['m_intrinsic'], instance_names, intrinsic_info['m_Width'], intrinsic_info['m_Height'], 0, 0.20, scene_data)
    if object2frame:
        lock.acquire()
        with open(output_filepath, "wb") as f:
            pickle.dump(object2frame, f, protocol=pickle.HIGHEST_PROTOCOL)
        lock.release()


# if __name__ == '__main__':
#     mode = "train"
#     export_path = CONF_PATH_R3SCAN_PROCESSED
#     root = None

#     root = CONF_PATH_R3SCAN_RAW

#     scene_data, selected_scans = read_json(root, mode)
#     export_path = os.path.join(export_path, "views")

#     scans = sorted(list(scene_data.keys()))
#     print("Storing views in: ", export_path)

#     for scan in tqdm(scans):
#         run(scan, scene_data, export_path=export_path)
