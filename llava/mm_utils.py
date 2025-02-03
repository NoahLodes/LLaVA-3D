from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast
import re

from typing import List, Dict, Any, Tuple, Union
from .constants import DEFAULT_BOX_TOKEN, DEFAULT_POINTS_TOKEN

Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, LOC_TOKEN_INDEX


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches

def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def process_videos(videos, video_processor, mode='random', data_dict=None, use_relationship=False, balance_img_with=None, device=None, text=None):

    if isinstance(videos, str):
        videos = [videos] # [..., './data/3rscan/754e884c-ea24-2175-8b34-cead19d4198d', ...]
    
    new_videos = []
    for video in videos:
        # video = ./data/3rscan/754e884c-ea24-2175-8b34-cead19d4198d // mode = random 
        video = video_processor.preprocess(video, return_tensors='pt', mode=mode, data_dict=data_dict, use_relationship=use_relationship, balance_img_with=balance_img_with, device=device, text=text)
        new_videos.append(video)

    new_images = [video['images'] for video in new_videos]
    new_depths = [video['depth_images'] for video in new_videos]
    new_poses = [video['poses'] for video in new_videos]
    new_intrinsics = [video['intrinsic'] for video in new_videos]
    
    videos_dict = dict()
    videos_dict['images'] = torch.stack(new_images, dim=0)
    videos_dict['depths'] = torch.stack(new_depths, dim=0)
    videos_dict['poses'] = torch.stack(new_poses, dim=0)
    videos_dict['intrinsics'] = torch.stack(new_intrinsics, dim=0)
    return videos_dict


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')] # 被 <image> 分开的部分分别 tokenize

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    
    input_ids = []
    offset = 0
    # 这里是因为有一个 begin token 所以 offset +1 
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):  # [-200, -200]
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def tokenizer_special_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, box_token_index=LOC_TOKEN_INDEX, return_tensors=None):
    import re
    separators = re.compile(r'(<image>|<boxes>)')
    input_ids = []
    offset = 0
    prompt_chunks = separators.split(prompt)  # ['"This is a test ', '<image>', ' with an image ', '<boxes>', ' and some boxes ', '<image>', ' another image']
    first_chunk = tokenizer(prompt_chunks[0]).input_ids
    if first_chunk[0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(first_chunk[0])
    for prompt_chunk in prompt_chunks:
        if prompt_chunk not in ['<image>', '<boxes>']:
            input_ids.extend(tokenizer(prompt_chunk).input_ids[offset:])
        elif prompt_chunk == '<image>':
            input_ids.extend([image_token_index])
        elif prompt_chunk == '<boxes>':
            input_ids.extend([box_token_index])
        else:
            raise NotImplementedError
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def map_obj(boxes_value: List[List[float]], boxes_seq: List[List[int]]) -> List[List[List[float]]]:
    """
    >>> boxes = [[2.3, 1.1, 4.2, 2.1, 0.5, 0,3 ,-1.2], [4.2, 1.2, -2.2, 3.2, 2.6, 0.3, 0.0], [3.3, 2.7, 1.3, 0.3, 0.2, 0.1, -1.2]]
    >>> boxes_seq_ = [[3, 1], [2]]
    >>> var = map_obj(boxes, boxes_seq_)
    """
    try:
        ret = []
        for boxes in boxes_seq:
            boxes_ret = []
            for box_index in boxes:
                if isinstance(box_index, (list, tuple)):
                    boxes_ret.append(boxes_value[box_index[0]][box_index[1]])
                else:
                    boxes_ret.append(boxes_value[box_index])
            ret.append(boxes_ret)
        return ret
    except:
        raise SystemExit(f"error: map obj {boxes_value} {boxes_seq}")
    
def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


class BoxFormatter:
    def __init__(self, bboxes_token=DEFAULT_BOX_TOKEN, points_token=DEFAULT_POINTS_TOKEN):
        self.bboxes_token = bboxes_token
        self.points_token = points_token
        # normally the bboxes_token_pat is the same as bboxes_token if u not use some weird token
        self.bboxes_token_pat = re.compile(bboxes_token)
        self.points_token_pat = re.compile(points_token)

    def __call__(self, sentence: str, bboxes_seq: BoxesSeq) -> str:
        all_box = self.bboxes_token_pat.findall(sentence)
        assert len(all_box) == len(bboxes_seq), f"not match. sentence: {sentence}. boxes:{bboxes_seq}"
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_box(bboxes) for bboxes in bboxes_seq]
        converted = sentence.replace(self.bboxes_token, '{}').format(*bboxes_strs)
        return converted

    def call_on_point(self, sentence: str, points_seq: BoxesSeq) -> str:
        all_box = self.points_token_pat.findall(sentence)
        assert len(all_box) == len(points_seq), f"not match. sentence: {sentence}. boxes:{points_seq}"
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_point(bboxes) for bboxes in points_seq]
        converted = sentence.replace(self.points_token, '{}').format(*bboxes_strs)
        return converted

    def format_point(self, points) -> str:
        raise NotImplementedError

    def format_box(self, bboxes: Boxes) -> str:
        raise NotImplementedError

    def extract(self, string: str) -> List[Boxes]:
        raise NotImplementedError

    def extract_point(self, string: str) -> List[Boxes]:
        raise NotImplementedError


class PlainBoxFormatter(BoxFormatter):

    def __init__(self, *args, precision=3, use_small_brackets=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.use_small_brackets = use_small_brackets

        small_brackets_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\)')
        small_brackets_point_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?)(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?))*\)')

        middle_brackets_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')
        middle_brackets_point_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?)(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?))*\]')

        self.pat = small_brackets_pat if use_small_brackets else middle_brackets_pat
        self.point_pat = small_brackets_point_pat if use_small_brackets else middle_brackets_point_pat

    def format_box(self, boxes: Boxes) -> str:
        box_strs = []
        for box in boxes:
            box_strs.append(','.join([f"{elem:.{self.precision}f}" for elem in box]))
        box_str = ';'.join(box_strs)
        if self.use_small_brackets:
            return "(" + box_str + ")"
        return "[" + box_str + "]"

    def format_point(self, points) -> str:
        return self.format_box(points)

    def extract(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

    def extract_point(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.point_pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

# ============================= Integration of Open3dsg ====================================

import os
import json 
import numpy as np

def load_scan(base_path, file_path):
    return json.load(open(os.path.join(base_path, file_path)))["scans"]

def obtain_frames(obj1_frames, obj2_frames, data_dict):    
    """
    Filters and finds common frames between two sets of object frames.
        :param obj1_frames (list): List of tuples representing frames for object 1.
        :param obj2_frames (list): List of tuples representing frames for object 2.
        
    Returns (list): A list of common frame names (strings) between the two objects.
    """
    if '3rscan' in data_dict['dataset']:
        path =  data_dict['dataset'] +'/'+ data_dict['scene_id'] +'/sequence/'
    elif 'scannet' in data_dict['dataset']:
        path = data_dict['dataset'] +'/'+ data_dict['scene_id'] +'/color/'

    # Filter frames starting with 'frame' for both objects
    frames_obj1 = [path+frame[0] for frame in obj1_frames if isinstance(frame, tuple) and isinstance(frame[0], str) and frame[0].startswith('frame')]
    frames_obj2 = [path+frame[0] for frame in obj2_frames if isinstance(frame, tuple) and isinstance(frame[0], str) and frame[0].startswith('frame')]
     
    # Find common frames
    common_frames = np.intersect1d(frames_obj1, frames_obj2)
    return common_frames.tolist()

def obtain_the_common_images(data_dict): 
    """
    Processes relationships between objects and finds common frames for each pair of related objects.
        :param data_dict (dict): A dictionary containing:
            - 'triples' (list): A list of object relationships in the format [obj1, relation, obj2].
            - 'obj2frame' (dict): A dictionary mapping object IDs to their respective frame lists.
    
    Returns common_frames (list): A list of common frames for each relationship, stored in `data_dict['common_frames']`.
    """
    # Validate that required keys exist
    if 'triples' not in data_dict or 'obj2frame' not in data_dict:
        raise KeyError("The dictionary must contain the keys 'triples' and 'obj2frame'.")
    
    relationships = data_dict['triples']
    data_dict['common_frames'] = []  # Initialize the list of common frames
    
    for relationship in relationships:
        obj1 = relationship[0]
        obj2 = relationship[1]
        
        # Validate that objects exist in obj2frame
        if obj1 not in data_dict['obj2frame'] or obj2 not in data_dict['obj2frame']:
            print(f"Warning: {obj1} or {obj2} not found in 'obj2frame'.")
            data_dict['common_frames'].append([])
            continue
        
        obj1_frames = data_dict['obj2frame'][obj1]
        obj2_frames = data_dict['obj2frame'][obj2]
        
        # Obtain common frames and add them
        common_frames = obtain_frames(obj1_frames, obj2_frames, data_dict)
        data_dict['common_frames'].append(common_frames)
    
    print("Processing completed. Common frames are stored in 'common_frames'.")
    return data_dict
