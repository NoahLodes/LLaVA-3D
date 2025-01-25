from transformers import CLIPImageProcessor, ProcessorMixin
import numpy as np
from pathlib import Path
from transformers.image_utils import to_numpy_array
from PIL import Image
import os
import torch
from torch import Tensor
import json
import cv2
from scipy.spatial.transform import Rotation as R


class RGBDVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("RGBDVideoTokenizer")
    r"""
    Constructs a RGBD video processor. Each RGBD video should include a series of  RGB images, depth images, 
    poses and camera intrinsic.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
    def __init__(self, vision_tower_name, cam_params_path, num_frames=24, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.vision_tower_name = vision_tower_name
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.tokenizer = tokenizer
        self.num_frames = num_frames

        with open(os.path.join(cam_params_path, 'embodiedscan_infos_full.json'), 'r') as file:
            self.scene = json.load(file)

    def valid_pose(self, video_poses):
        valid_video_poses = []
        for video_pose in video_poses:
            pose = np.loadtxt(video_pose)
            if np.any(np.isinf(pose)):
                continue
            valid_video_poses.append(video_pose)
        return valid_video_poses


    def inpaint_depth(self, depth):
        """
        inpaints depth using opencv
        Input: torch tensor with depthvalues: H, W
        Output: torch tensor with depthvalues: H, W
        """
        depth_inpaint = cv2.inpaint(depth, (depth == 0).astype(np.uint8), 5, cv2.INPAINT_NS)
        depth[depth == 0] = depth_inpaint[depth == 0]
        return depth

    def extract_frames(self, frames):
        
        if not isinstance(frames, list):
            frames = [frames]

        if 'scannet' in frames[0]:
            images = []
            depths = []
            poses = []
            path = Path(frames[0])
            video = str(Path(*path.parts[:-2]))
            for frame in frames:
                image = frame
                depth = frame.replace('color', 'depth')
                # we need to ensure that the frame has valid pose
                pose = frame.replace('color', 'pose').replace('png', 'txt')
                images.append(image)
                depths.append(depth)
                poses.append(pose)
            depth_intrinsic_file = os.path.join(video, 'intrinsic/intrinsic_depth.txt')
            axis_align_matrix_file = os.path.join(video, 'axis_align_matrix.txt')
            video_info = dict()
            video_info['sample_image_files'] = images
            video_info['sample_depth_image_files'] = depths
            video_info['sample_pose_files'] = poses
            video_info['depth_intrinsic_file'] = depth_intrinsic_file
            video_info['axis_align_matrix_file'] = axis_align_matrix_file
        else:
            raise NotImplementedError

        return video_info

    def subsample_embodiedscan_frames(self, video):
        image_path = Path(video)
        video_frames = sorted(image_path.glob("*.jpg"))  # find all the color frames of the rgbd video
        video_frames = [str(video_frame) for video_frame in video_frames]
        sample_factor = len(video_frames) // self.num_frames
        start_point = 0
        sample_ids = [(start_point + i*sample_factor) % len(video_frames) for i in range(self.num_frames)]
        sample_frames = [video_frames[i] for i in sample_ids]
        video_info = self.extract_embodiedscan_frames(sample_frames) 
        return video_info

    def extract_openscan_video(self, video):
        
        with open(os.path.join(video, 'poses.txt'), 'r') as f:
            ori_poses = f.readlines()

        video_frames = ori_poses[1:]
        sample_factor = len(video_frames) // self.num_frames
        start_point = 0
        sample_ids = [(start_point + i*sample_factor) % len(video_frames) for i in range(self.num_frames)]
        sample_frames = [video_frames[i] for i in sample_ids]

        images = []
        depths = []
        poses = []
        for frame in sample_frames:
            timestamp, x, y, z, qx, qy, qz, qw = frame.split()
            x, y, z, qx, qy, qz, qw = float(x), float(y), float(z), float(
                qx), float(qy), float(qz), float(qw)
            rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
            transform_matrix = np.identity(4)
            transform_matrix[:3, :3] = rot_matrix @ [[0, 0, 1], [-1, 0, 0],
                                                        [0, -1, 0]]
            transform_matrix[:3, 3] = [x, y, z]  # CAM to NOT ALIGNED GLOBAL

            image = os.path.join(video, 'rgb', timestamp + '.jpg')
            depth = os.path.join(video, 'depth', timestamp + '.png')
            pose = transform_matrix
            images.append(image)
            depths.append(depth)
            poses.append(pose)

        intrinsic_file = np.loadtxt(os.path.join(video, 'intrinsic.txt'))
        axis_align_matrix_file = np.loadtxt(os.path.join(video, 'axis_align_matrix.txt'))

        video_info = dict()
        video_info['sample_image_files'] = images
        video_info['sample_depth_image_files'] = depths
        video_info['sample_pose_files'] = poses
        video_info['intrinsic_file'] = intrinsic_file
        video_info['axis_align_matrix_file'] = axis_align_matrix_file
        video_info['sample_frame_num'] = len(sample_frames)
        video_info['dataset'] = 'openscan'

        return video_info

    def extract_embodiedscan_video(self, video, data_dict, use_relationship):
        # video is the full path for the video
        video_path = Path(video)                                    # video_path:    LLaVA-3D-Demo-Data/scannet/scene0356_00         // data/3RScan/754e884c-ea24-2175-8b34-cead19d4198d
        video_name = str(Path(*video_path.parts[-2:])).lower()      # video_name:    scannet/scene0356_00                            // 3RScan/754e884c-ea24-2175-8b34-cead19d4198d
        dataset = (video.split('/')[-2]).lower()                    # dataset:       scannet                                         // 3RScan
        video_folder = str(Path(*video_path.parts[:-2]))            # video_folder:  LLaVA-3D-Demo-Data                              // data
                                                                    # self.scene dict keys: scannet/scene0191_00                     // 3rscan/3rscan0002
        
        if dataset == '3rscan':
            print('         [extract_embodiedscan_video]: Processing 3rscan: ')
            video_info = dict(filter(lambda item: item[0].startswith(dataset), self.scene.items()))   # video_info: {3rscan/3rscan0002: {}}
            video_frames = []                                   # video_frames: {'3rscan/754e884c-ea24-2175-8b34-cead19d4198d/sequence/frame-000000.color.jpg': {pose: list, }}
            for _, scene_attr in video_info.items():
                for attr in scene_attr.keys(): 
                    if attr.startswith(video_name):
                        video_frames.append(scene_attr)
        else:         
            video_info = self.scene[video_name]                                                 # video dict keys: 'scannet/posed_images/scene0356_00/00000.jpg'
            video_frames = [str(key) for key in video_info.keys() if key.startswith(dataset)]   # Remove other paramters, keep scannet/posed_images/scene0356_00/00000.jpg
        import pdb 

        if not use_relationship and len(video_frames) > self.num_frames:
            sample_factor = len(video_frames) // self.num_frames
            start_point = 0
            sample_ids = [(start_point + i*sample_factor) % len(video_frames) for i in range(self.num_frames)]
            sample_frames = [video_frames[i] for i in sample_ids] # sample_id: scannet/posed_images/scene0356_00/00000.jpg
        elif not use_relationship and len(video_frames) < self.num_frames:
            repeat_times = (self.num_frames // len(video_frames)) + 1
            # Extend the list by repeating it and then slice to get exactly self.num_frames elements
            sample_frames = (video_frames * repeat_times)[:self.num_frames]
        elif use_relationship: 
            # Select the common frames of the relationship
            sample_frames = []
            for video_frame in video_frames[0].keys(): 
                if video_frame in data_dict['rel2frame_path'][int(use_relationship)]: 
                    sample_frames.append(video_frame)
            
            #sample_frames = [video_frame for video_frame in video_frames if video_frame.key() in data_dict['common_frames'][int(common_frames)]]
        else:
            sample_frames = video_frames

        images = []
        depths = []
        poses = []
        if dataset == 'matterport3d':
            intrinsics = []

        for frame in sample_frames:
            
            pose = np.array(video_frames[0][frame]['pose']) # 4x4 array
            image = os.path.join(video_folder, frame)
            
            if 'scannet' in frame:
                depth = os.path.join(video_folder, video_info[frame]['depth'])
            elif '3rscan' in frame:
                depth = os.path.join(video_folder, frame.replace('color.jpg', 'depth.pgm')) #.replace('3rscan', '3rscan_depth'))
            elif 'matterport' in frame:
                depth = os.path.join(video_folder, video_info[frame]['depth'])
                intrinsic = np.array(video_info[frame]['intrinsic'])
                intrinsics.append(intrinsic)  # (4, 4)
            else:
                raise NotImplementedError
            images.append(image)
            depths.append(depth)
            poses.append(pose)
            
        sampled_video_info = dict()
        if dataset == 'matterport3d':
            intrinsic_file = np.stack(intrinsics, axis=0) # Vx4x4 array
            axis_align_matrix_file = np.array(video_info['axis_align_matrix'])  # 4x4 array
        elif dataset == '3rscan': 
            intrinsic_file = np.array(video_frames[0]['intrinsic'])              # 4x4 array
            depth_intrinsic_file = np.array(video_frames[0]['depth_intrinsic'])  # 4x4 array
            sampled_video_info['depth_intrinsic_file'] = depth_intrinsic_file
            axis_align_matrix_file = np.array(video_frames[0]['axis_align_matrix'])  # 4x4 array
        else:
            intrinsic_file = np.array(video_info['intrinsic'])              # 4x4 array
            depth_intrinsic_file = np.array(video_info['depth_intrinsic'])  # 4x4 array
            sampled_video_info['depth_intrinsic_file'] = depth_intrinsic_file
            axis_align_matrix_file = np.array(video_info['axis_align_matrix'])  # 4x4 array
        
        sampled_video_info['sample_image_files'] = images
        sampled_video_info['sample_depth_image_files'] = depths
        sampled_video_info['sample_pose_files'] = poses
        sampled_video_info['intrinsic_file'] = intrinsic_file
        sampled_video_info['axis_align_matrix_file'] = axis_align_matrix_file
        sampled_video_info['dataset'] = dataset
        sampled_video_info['sample_frame_num'] = len(images)

        return sampled_video_info

    def extract_embodiedscan_frames(self, frames):
        if not isinstance(frames, list):
            frames = [frames]
        if 'scannet' in frames[0] or '3rscan' in frames[0]:
            images = []
            depths = []
            poses = []
            if 'scannet' in frames[0]:
                video = frames[0].split('/')[-4] + '/' + frames[0].split('/')[-2]
            elif '3rscan' in frames[0]:
                video = frames[0].split('/')[-4] + '/' + frames[0].split('/')[-3]
            video_info = self.scene[video]
            for frame in frames:
                path = Path(frame)
                frame_name = str(Path(*path.parts[-4:]))
                pose = np.array(video_info[frame_name]['pose']) # 4x4 array
                image = frame
                if 'scannet' in frame:
                    depth = frame.replace('jpg', 'png')
                elif '3rscan' in frame:
                    depth = frame.replace('color.jpg', 'depth.png')
                else:
                    raise NotImplementedError
                # we need to ensure that the frame has valid pose
                images.append(image)
                depths.append(depth)
                poses.append(pose)
            depth_intrinsic_file = np.array(video_info['depth_intrinsic'])  # 4x4 array
            intrinsic_file = np.array(video_info['intrinsic']) # 4x4 array
            axis_align_matrix_file = np.array(video_info['axis_align_matrix'])  # 4x4 array
            video_info = dict()
            video_info['sample_image_files'] = images
            video_info['sample_depth_image_files'] = depths
            video_info['sample_pose_files'] = poses
            video_info['depth_intrinsic_file'] = depth_intrinsic_file
            video_info['intrinsic_file'] = intrinsic_file
            video_info['axis_align_matrix_file'] = axis_align_matrix_file
        else:
            raise NotImplementedError

        return video_info

    def subsample_frames(self, video):
        """
        Actually we may need to adapt this function for different datasets
        """ 
        if 'scannet' in video:
            image_path = Path(video) / 'color'
            video_frames = sorted(image_path.glob("*.png"))  # find all the color frames of the rgbd video
            video_poses = [str(video_frame).replace('color', 'pose').replace('png', 'txt') for video_frame in video_frames]
            assert len(video_frames) == len(video_poses)
            valid_video_poses = self.valid_pose(video_poses)
            sample_factor = len(valid_video_poses) // self.num_frames
            start_point = 0
            sample_ids = [(start_point + i*sample_factor) % len(valid_video_poses) for i in range(self.num_frames)]
            sample_poses = [valid_video_poses[i] for i in sample_ids]
            sample_images = [sample_pose.replace('pose', 'color').replace('txt', 'png')  for sample_pose in sample_poses]
            sample_depths = [sample_image.replace('color', 'depth') for sample_image in sample_images]

            depth_intrinsic_file = os.path.join(video, 'intrinsic/intrinsic_depth.txt')
            axis_align_matrix_file = os.path.join(video, 'axis_align_matrix.txt')

            video_info = dict()
            video_info['sample_image_files'] = sample_images
            video_info['sample_depth_image_files'] = sample_depths
            video_info['sample_pose_files'] = sample_poses
            video_info['depth_intrinsic_file'] = depth_intrinsic_file
            video_info['axis_align_matrix_file'] = axis_align_matrix_file
        else:
            raise NotImplementedError

        return video_info
    
    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):  # (V, 4, 4) (resize_shape) (h, w)
        
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)
        
        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic

    def preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):

        width, height = depth_image.size
        requested_new_short = self.image_processor.crop_size['height']
        if width < height:
            scale = requested_new_short / width
            new_width = requested_new_short
            new_height = int(height * scale)
        else:
            scale = requested_new_short / height
            new_height = requested_new_short
            new_width = int(width * scale)
        resized_depth_image = depth_image.resize((new_width, new_height), Image.NEAREST)
        target_height = self.image_processor.crop_size['height']  # 336
        target_width  = self.image_processor.crop_size['width']  # 336
        # center crop
        left = (new_width - target_width) / 2
        top = (new_height - target_height) / 2
        right = (new_width + target_width) / 2
        bottom = (new_height + target_height) / 2

        img = resized_depth_image.crop((left, top, right, bottom))
        # rescale the depth image
        img = to_numpy_array(img)
        if do_depth_scale:
            img = img / depth_scale

        return img, (new_width, new_height)


    def preprocess(self, 
                   video: str, 
                   return_tensors='pt', 
                   mode='random', 
                   data_dict=None,
                   use_relationship=False,
                   device=None, 
                   text=None,
                   do_rescale=True,
                   do_normalize=True,
                   do_depth_scale=True):
        """
            video:  1. str video id / single video frame
                    2. list  list of video frames
        """
        print('     Starting the preprocessing fo the process[video]. ')
        if isinstance(video, list):   # list of video frames only could be embodiedscan data
            print('         Inside extract_embodiedscan_frames')
            video_info = self.extract_embodiedscan_frames(video)
        elif video.endswith('png') or video.endswith('jpg'):
            print('         Inside  extract_frames')
            video_info = self.extract_frames(video)
        elif 'frames' in video:  # scene-based odin data
            print('         Inside frames ')
            if mode == 'random':
                video_info = self.subsample_frames(video) 
            else:
                raise NotImplementedError
        elif 'openscan' in video:
            print('         Inside openscan')
            video_info = self.extract_openscan_video(video)
        else:
            print('         Inside  extract_embodiedscan_video')
            video_info = self.extract_embodiedscan_video(video, data_dict, use_relationship)

        dataset = video_info['dataset']
        sample_frame_num = video_info['sample_frame_num']
        if dataset == 'matterport3d':
            depth_scale = 4000
        else:
            depth_scale = 1000

        images = []
        depth_images = []
        poses = []

        if 'depth_intrinsic_file' in video_info:
            depth_intrinsic = video_info['depth_intrinsic_file']
            if not isinstance(depth_intrinsic, np.ndarray):
                depth_intrinsic = np.loadtxt(depth_intrinsic)

        intrinsic = video_info['intrinsic_file']  # (V, 4, 4) or (4, 4)
        if not isinstance(intrinsic, np.ndarray):
            intrinsic = np.loadtxt(intrinsic)

        for id, image_file in enumerate(video_info['sample_image_files']):
            image = Image.open(image_file).convert('RGB')
            if dataset == "3rscan":
                image = image.rotate(-90, expand=True)
            image_size = image.size
            image = self.image_processor.preprocess(images=image, do_rescale=do_rescale, do_normalize=do_normalize, return_tensors=return_tensors)['pixel_values'][0] # [3, H, W]
            depth_image = Image.open(video_info['sample_depth_image_files'][id])
            depth_image_size = depth_image.size
            depth_image, resize_shape = self.preprocess_depth_image(depth_image, do_depth_scale=do_depth_scale, depth_scale=depth_scale)
            depth_image = torch.as_tensor(np.ascontiguousarray(depth_image)).float() # [H, W]
            pose = video_info['sample_pose_files'][id]
            if not isinstance(pose, np.ndarray):
                pose = np.loadtxt(pose)
            pose = torch.from_numpy(pose).float()  # [4, 4]
            images.append(image)
            depth_images.append(depth_image)
            poses.append(pose)

        if dataset == 'scannet':
            intrinsic = self.preprocess_instrinsic(depth_intrinsic, depth_image_size, resize_shape)
        else:
            intrinsic = self.preprocess_instrinsic(intrinsic, image_size, resize_shape)  # 3rscan / matterport

        intrinsic = torch.from_numpy(intrinsic).float()

        if intrinsic.dim() == 2:  # scannet/3rscan
            intrinsic = intrinsic.unsqueeze(0).repeat(sample_frame_num, 1, 1)  # (V, 4, 4)

        axis_align_matrix = video_info['axis_align_matrix_file']
        if not isinstance(axis_align_matrix, np.ndarray):
            axis_align_matrix = np.loadtxt(axis_align_matrix)

        axis_align_matrix = torch.from_numpy(axis_align_matrix).float()
        
        # transform pose to axis_align pose
        poses = [axis_align_matrix @ pose for pose in poses]

        video_dict = dict()
        video_dict['images'] = torch.stack(images)  # (V, 3, 336, 336)
        video_dict['depth_images'] = torch.stack(depth_images)  # (V, 336,336)
        video_dict['poses'] = torch.stack(poses)  # (V, 4, 4)
        video_dict['intrinsic'] = intrinsic  # (V, 4, 4)

        return video_dict
