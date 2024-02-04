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
from transformers import BitsAndBytesConfig
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

from llava_instruct_dataset import LlavaFinetuneDataset, LlavaCollator, split_data_to_train_and_val
from model import LitMultiModalGPT, SimpleLinearBlock, model_summary


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, 
                 projection_save_dir, 
                 phi_save_dir, 
                 save_freq, 
                 verbose: bool = False, 
                 every_n_train_steps=0, 
                 every_n_epochs=0,
                 save_on_train_epoch_end=False):
        super().__init__()
        self.verbose = verbose
        self.projection_save_dir = projection_save_dir
        self.phi_save_dir = phi_save_dir
        self.save_freq = save_freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.save_freq == 0:
            print("Saving checkpoint for projection layer")
            projection_file_name = os.path.join(self.save_dir, f"projection_layer_ckpt_finetuning_{trainer.global_step}.pt")
            torch.save(trainer.model.projection_layer.state_dict(), projection_file_name)

            print("Saving checkpoint for adapter layers")
            adapter_dir = os.path.join(self.save_dir, f"adapter_layer_ckpt_finetuning_{trainer.global_step}")
            os.makedirs(adapter_dir,exist_ok=True)
            trainer.model.llm_model.save_pretrained(adapter_dir, save_adapter=True, save_config=True)

    
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


def finetune_multimodal_gpt_model(model, train_dataloader, val_dataloader, logger=None, ckpt_path=None, max_training_steps=2):
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
    batch_size = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    projection_layer_in_channels = 768
    projection_layer_out_channels = 2560
    max_training_steps = 100000
    seq_len = 64
    max_ques_length = 32
    log_dir = 'phi_pretrain'
    exp_name = 'phi2_proj_layer'
    check_point_save_dir = 'phi2_finetune_checkpoints/'
    os.makedirs(check_point_save_dir,exist_ok = True)
    save_freq = 1000
    val_check_interval = 1000
    log_dir = './logs'
    lr = 1e-4
    stage1_projection_checkpoints = 'phi2_projection_checkpoints/ckpt_latest.pt'
    resume_training=False

    bb_config = BitsAndBytesConfig( 
    load_in_4bit=True, # Load the model in 4 bits
    bnb_4bit_quant_type="nf4", # 4 bit quant type
    bnb_4bit_use_double_quant=True, # double quant saves more bits
    bnb_4bit_compute_dtype=torch.float16, # use bfloat16 
    )


    # Instantiate WandB logger
    wandb.login()
    wandb_logger = WandbLogger(save_dir=log_dir, name='mm_phi_finetune_ec2')
    text_table = wandb.Table(columns=["training_step", "loss", "text"])
    val_data_log = []



    # Define the models and tokenizers
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_preprocessor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")


    # Define dataset and dataloaders
    train_data, val_data = split_data_to_train_and_val(json_path)

    # lets keep val size to 100
    print(f"Train dataset size: {len(train_data)}")
    print(f"Valid dataset size: {len(val_data)}")

    collator = LlavaCollator(tokenizer = phi_tokenizer, max_length = max_ques_length)

    train_ds = LlavaFinetuneDataset(train_data,
                                    phi_tokenizer,
                                    clip_model,
                                    clip_preprocessor,
                                    device,
                                    max_seq_len=max_ques_length)

    val_ds = LlavaFinetuneDataset(val_data,
                                  phi_tokenizer,
                                  clip_model,
                                  clip_preprocessor,
                                  device,
                                  max_seq_len=max_ques_length)


    train_dataloader = DataLoader(dataset = train_ds,
                                batch_size = batch_size,
                                num_workers = 3,
                                collate_fn = collator,
                                shuffle = True)

    val_dataloader = DataLoader(dataset = val_ds,
                                batch_size = 1,
                                num_workers = 3,
                                collate_fn = collator,
                                shuffle = True)


    # Define multimodal model
    multimodal_gpt_model = LitMultiModalGPT(projection_layer_in_channels,
                                            projection_layer_out_channels,
                                            )
    optimizer = torch.optim.Adam(multimodal_gpt_model.parameters(), lr=1.0e-4, eps=1e-9)
    multimodal_gpt_model.set_optimizer(optimizer)

    # load the projection weights 
    multimodal_gpt_model.projection_layer.load_state_dict(torch.load(stage1_projection_checkpoints))
    print("loaded projection weights")

    # if resuming, load the appropriate checkpoint
    if resume_training:
        pass
        

    # define and train the model
    trainer = finetune_multimodal_gpt_model(multimodal_gpt_model, train_dataloader, val_dataloader, logger = wandb_logger, max_training_steps=max_training_steps)
