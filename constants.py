def get_paths(name):
    if name=="Noah":
        blip_cache_dir = '/content/drive/MyDrive/master_practical/models/BLIP'
        llava_cache_dir = '/content/drive/MyDrive/master_practical/models/LLaVA3D'
        path_3rscan_raw = '/content/drive/MyDrive/master_practical/data/3RScan'
        path_3rscan_metadata = './data/3RScan'
        path_3rscan_processed = './data/3RScan/processed'

    return blip_cache_dir, llava_cache_dir, path_3rscan_raw, path_3rscan_metadata, path_3rscan_processed
