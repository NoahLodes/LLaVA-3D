import argparse
import torch
import pdb


from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    process_videos,
    tokenizer_special_token,
    get_model_name_from_path,
    obtain_the_common_images, 
    load_scan
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import json
import numpy as np
import os

from open3dsg.const import CONF_PATH_R3SCAN_RAW
from open3dsg.open_dataset import Open2D3DSGDataset

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args, data_dict, common_frames):
    # Model
    disable_torch_init()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    mode = None

    if args.video_path:
        print(f"Video path provided: {args.video_path}")
        mode = 'video'
    if args.image_file:
        print(f"Image file provided: {args.image_file}")
        mode = 'image'

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, torch_dtype=torch_dtype
    )

    qs = args.query

    matches = re.search(r"\[([^\]]+)\]", qs)
    if matches:
        coord_list = [float(x) for x in matches.group(1).split(',')]
        coord_list = [round(coord, 3) for coord in coord_list[:3]]
        qs = re.sub(r"\[([^\]]+)\]", "<boxes>", qs)
        clicks = torch.tensor([coord_list])
    else:
        clicks = torch.zeros((0,3))

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "3D" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if mode == 'image': 
        # Load images from common frames
        image_files = image_parser(args)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            processor['image'],
            model.config
        ).to(model.device, dtype=torch_dtype)
        depths_tensor = None
        poses_tensor = None
        intrinsics_tensor = None
        clicks_tensor = None

    if mode == 'video': 
        # process_video_common_frames
        videos_dict = process_videos(
            args.video_path,
            processor['video'],
            mode='random',
            device=model.device,
            text=args.query, 
            data_dict=data_dict,
            common_frames=common_frames,
        )
        images_tensor = videos_dict['images'].to(model.device, dtype=torch_dtype) # Shape: [B, num_frames, channels, H, W]=[1, 20, 3, 336, 336]
        print(f'Video process has finished:: {images_tensor.shape[1]} number of frames for relationship: {common_frames}')
        #pdb.set_trace()
        
        #TODO: Store the img 90 depths and poses instrinsics
        """        
        first_frame = images_tensor[0, 0]  # First frame
        tenth_frame = images_tensor[0, 9]  # 10th frame

        # Convert to (H, W, C) format and move to CPU
        first_frame_np = first_frame.permute(1, 2, 0).cpu().to(torch.float32).numpy()
        tenth_frame_np = tenth_frame.permute(1, 2, 0).cpu().to(torch.float32).numpy()

        # Convert pixel values to uint8 (assuming input is in range [0,1])
        first_frame_np = (first_frame_np * 255).astype('uint8')
        tenth_frame_np = (tenth_frame_np * 255).astype('uint8')

        # Convert to PIL images
        first_image = Image.fromarray(first_frame_np)
        tenth_image = Image.fromarray(tenth_frame_np)

        # Save images to disk
        from pathlib import Path
        current_directory = Path.cwd()
        print(f"Current directory: {current_directory}")
        first_image.save('scene0356_00_00000.jpg')
        tenth_image.save('scene0356_00_00009.jpg')

        print("Images saved successfully.")
        """

        depths_tensor = videos_dict['depths'].to(model.device, dtype=torch_dtype)
        poses_tensor = videos_dict['poses'].to(model.device, dtype=torch_dtype)
        intrinsics_tensor = videos_dict['intrinsics'].to(model.device, dtype=torch_dtype)
        clicks_tensor = clicks.to(model.device, dtype=torch.bfloat16)

    input_ids = (
        tokenizer_special_token(prompt, tokenizer, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            depths=depths_tensor,
            poses=poses_tensor,
            intrinsics=intrinsics_tensor,
            clicks=clicks_tensor,
            image_sizes=None,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-path", type=str, help="Path to the video file")
    group.add_argument("--image-file", type=str, help="Path to the image file")

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--common_frames", type=bool, default=False)
    args = parser.parse_args()

    # Load dataset and relationships
    scan_id = '754e884c-ea24-2175-8b34-cead19d4198d'
    D3SSG = load_scan(CONF_PATH_R3SCAN_RAW, "relationships_train.json")

    for r in D3SSG:
        if r['scan'] == scan_id:
            D3SSG = [r]

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
    import pdb
    
    # Process relationships and pass them to eval_model
    for instance in dataset: 
        data_dict = obtain_the_common_images(instance)
    #data_dict = [obtain_the_common_images(instance) for instance in dataset]

    eval_model(args, data_dict=data_dict, common_frames=4) 
