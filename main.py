import os
import gc
import torch
import pickle
import json
import torchinfo
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
#from numba import cuda
from typing import Union, List
import torch.multiprocessing as mp 
from torch.cuda.amp import autocast
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from transformers import AutoProcessor, CLIPVisionModel
from pytorch_lightning.callbacks import Callback
import torchmetrics
import wandb

from dataset import fetch_captions_and_images, PreTrainDataset
from model import LitMultiModalGPT, SimpleLinearBlock, model_summary


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, checkpoint_save_dir, save_freq, verbose: bool = False, every_n_train_steps=0, every_n_epochs=0):
        super().__init__()
        self.verbose = verbose
        self.save_dir = checkpoint_save_dir
        self.save_freq = save_freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.save_freq == 0:
            # save the model at the end of every epoch
            model_filename = os.path.join(self.save_dir, f"ckpt_{trainer.global_step}.pt")
            # first remove 
            # Save only the state_dict of projection layer
            torch.save(trainer.model.projection_layer.state_dict(), model_filename)
    
    def on_validation_end(self, trainer, pl_module):
        pass


class PrintAccuracyAndLoss(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        train_loss = trainer.callback_metrics['train_loss']
        trainer.model.log("train_step_loss", train_loss)
        if batch_idx % 500 == 0:
            print(f"Step: {trainer.global_step}: train_loss={train_loss:.4f}")


def train_multimodal_gpt_model(model, train_dataloader, val_dataloader, logger=None, ckpt_path=None, max_training_steps=2):
    trainer = Trainer(
        enable_checkpointing=True,
        #max_steps=max_training_steps,
        max_epochs = 100,
        accelerator="auto", #"auto" if torch.cuda.is_available() else "cpu",
        devices = 1, 
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="step"),
                   TQDMProgressBar(refresh_rate=10),
                   PeriodicCheckpoint(check_point_save_dir, save_freq, verbose=True),
                   PrintAccuracyAndLoss()],
        num_sanity_val_steps=0,
        val_check_interval = val_check_interval,
        precision="16"
    )
    
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    return trainer



if __name__ == '__main__':



    # Define configuration
    #raw_images_path = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
    #train_dataset_path = '/kaggle/input/coco2017-clip-image-embeddings/coco_embeddings_clip_vision_1x768'
    #captions_path = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json'
    #captions_key = 'annotations'
    json_path = './data/captions_train2017.json'
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    projection_layer_in_channels = 768
    projection_layer_out_channels = 2560
    max_training_steps = 100000
    seq_len = 64
    log_dir = 'phi_pretrain'
    exp_name = 'phi2_proj_layer'
    check_point_save_dir = 'phi2_projection_checkpoints/'
    os.makedirs(check_point_save_dir,exist_ok = True)
    save_freq = 1000
    val_check_interval = 1000
    log_dir = './logs'
    lr = 1e-4
    resume_training=False
    proj_weights_path = 'phi2_projection_checkpoints/ckpt_latest.pt'

    # Instantiate WandB logger
    wandb.login()
    wandb_logger = WandbLogger(save_dir=log_dir, name='mm_phi_stage1_ec2')
    text_table = wandb.Table(columns=["training_step", "loss", "text"])
    val_data_log = []



    # Define the models and tokenizers
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_preprocessor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")


    # Define dataset and dataloaders
    captions, raw_images_list, image_ids_list = fetch_captions_and_images(json_path)
    image_ids_list = np.array(image_ids_list)
    rand_indices = np.arange(len(image_ids_list))
    np.random.shuffle(rand_indices)
    val_indices = rand_indices[:100]
    train_indices = rand_indices[100:]
    train_image_ids = image_ids_list[train_indices]
    val_image_ids = image_ids_list[val_indices]

    # lets keep val size to 100
    print(f"Train dataset size: {len(train_indices)}")
    print(f"Valid dataset size: {len(val_indices)}")

    train_ds = PreTrainDataset(train_image_ids, 
                                raw_images_list, 
                                captions, 
                                phi_tokenizer,
                                clip_model,
                                clip_preprocessor,
                                device,
                                seq_len=seq_len)

    val_ds = PreTrainDataset(val_image_ids, 
                                raw_images_list, 
                                captions, 
                                phi_tokenizer,
                                clip_model,
                                clip_preprocessor,
                                device,
                                seq_len=seq_len)

    train_dataloader = DataLoader(dataset = train_ds,
                                batch_size = batch_size,
                                num_workers = 3,
                                collate_fn = None,
                                shuffle = True)
    val_dataloader = DataLoader(dataset = val_ds,
                                batch_size = 1,
                                num_workers = 3,
                                collate_fn = None,
                                shuffle = True)

    #test_model = SimpleLinearBlock(projection_layer_in_channels, projection_layer_out_channels)
    #model_summary(test_model, (1,50, 768))

    # Define multimodal model
    multimodal_gpt_model = LitMultiModalGPT(projection_layer_in_channels,
                                            projection_layer_out_channels,
                                            )
    optimizer = torch.optim.Adam(multimodal_gpt_model.parameters(), lr=1.0e-4, eps=1e-9)
    multimodal_gpt_model.set_optimizer(optimizer)

    # if resume training, load the model 
    if resume_training:
        multimodal_gpt_model.projection_layer.load_state_dict(torch.load(proj_weights_path))
        print("loaded projection weights")
        

    # define and train the model
    trainer = train_multimodal_gpt_model(multimodal_gpt_model, train_dataloader, val_dataloader, logger = wandb_logger, max_training_steps=max_training_steps)
