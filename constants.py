def get_paths(name):
    if name=="Noah":
        blip_cache_dir = '/content/drive/MyDrive/master_practical/models/BLIP'
        llava_cache_dir = '/content/drive/MyDrive/master_practical/models/LLaVA3D'
        path_3rscan_raw = '/content/drive/MyDrive/master_practical/data/3rscan'
        path_3rscan_metadata = './data/3rscan'
        path_3rscan_processed = './data/3rscan/processed'
    
    elif name=="Elena":
        blip_cache_dir = '/content/drive/MyDrive/4tCarrera/master_practical/models/BLIP'
        llava_cache_dir = '/content/drive/MyDrive/4tCarrera/master_practical/models/LLaVA3D'
        path_3rscan_raw = '/content/drive/MyDrive/4tCarrera/master_practical/data/3rscan'
        path_3rscan_metadata = './data/3rscan'
        path_3rscan_processed = './data/3rscan/processed'


    return blip_cache_dir, llava_cache_dir, path_3rscan_raw, path_3rscan_metadata, path_3rscan_processed
