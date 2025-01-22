from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llava.model import *
import torch


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

model_path = "ChaimZhu/LLaVA-3D-7B"
model_name = get_model_name_from_path(model_path)

cache_dir = "./models/llava"

kwargs = {"device_map": {'': 0}}
#kwargs['torch_dtype'] = torch.bfloat16
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Download the tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, cache_dir=cache_dir)
model = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    **kwargs,
    cache_dir=cache_dir
)
