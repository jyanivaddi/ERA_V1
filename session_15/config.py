from pathlib import Path

def get_config():
    return {
        "batch_size": 2048,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": 'en',
        "lang_tgt": 'it',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": True,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "ds_mode": "disk",
        "ds_path": "/home/e183534/OpusBooks",
        "save_ds_to_disk": False,
    }

def get_weights_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)