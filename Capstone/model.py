import os
import gc
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from typing import Union, List
from torch.cuda.amp import autocast
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import Callback
import torchmetrics
import wandb


def model_summary(model, input_size):
    torchinfo.summary(model, 
                      input_size = input_size, 
                      batch_dim=0, 
                      col_names=("kernel_size",
                                 "input_size",
                                 "output_size",
                                 "num_params",
                                 "mult_adds"),
                       verbose=1,) 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SimpleLinearBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.pre_norm = nn.LayerNorm(in_size)
        self.proj = nn.Sequential
        (
            nn.Linear(in_size, out_size),
            nn.GELU()
            )
        
    def forward(self,x):
        return self.proj(self.pre_norm(x))


class LitMultiModalGPT(LightningModule):
    """
    Pytorch Lightning module for Transformer

    """
    def __init__(self,
                 projection_layer_in_channels,
                 projection_layer_out_channels,
                 num_validation_examples=2,
                 num_training_steps=100000):
        super().__init__()
        self.num_validation_examples = num_validation_examples
        self.num_training_steps = num_training_steps
        self.scheduler = None
        self.scheduler_dict = {}
        self.optimizer = None
        self.this_step_train_loss = None
        self.predicted_list = []
        self.expected_list = []
        self.save_hyperparameters(ignore=['loss_criterion', 'epoch'])
        self.projection_layer = SimpleLinearBlock(projection_layer_in_channels,projection_layer_out_channels)
        self.llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.train_loss_values = []
        self.val_loss_values = []
        self.COMMENT_TOKEN_ID = 23893
        self.EOS_TOKEN_ID = 50256

     
        # freeze the llm
        for param in self.llm_model.parameters():
            #print(param.dtype)
            param.requires_grad = False

    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    
    def set_scheduler_dict(self, scheduler, freq='step'):
        self.scheduler = scheduler
        self.scheduler_dict = {
            "scheduler": self.scheduler,
            "interval": freq,
        }

    def configure_optimizers(self):
        if self.scheduler_dict:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler_dict}
        return {"optimizer": self.optimizer}
         
   
    def forward(self, batch):
        x = batch['image_embeddings']
        targets = batch['tokenized_captions']
        x = self.projection_layer(x)
        outputs_dict = self.llm_model(inputs_embeds = x,
                                     labels = targets,
                                     return_dict = True) 
        return outputs_dict
    

    def proj_output(self, image_embeds):
        return self.projection_layer(image_embeds)


    def generate(self, x):
        proj_outs = self.projection_layer(x)
        with torch.no_grad():
            pred_logits, outputs = self.llm_model(inputs_embeds = proj_outs, return_dict=False)
            output_tokens = torch.argmax(pred_logits, axis=-1)  
            generated_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        return generated_text
    
    
    def evaluate(self,batch, stage):
        if stage:
            predicted = self.generate(batch['image_embeddings'])
            #self.predicted_list.append(predicted)
            #self.expected_list.append(batch['caption'])
            # print the source, target, and the model output
            #print("*****************************************")
            input_text = f"{f'caption: ' :>12}{batch['captions'][0]}"
            pred_text = f"{f'predicted: ' :>12}{predicted}"
            print(f"*****************************************\n{input_text}\n{pred_text}")
            # log a W&B Table that has a text caption, an image and audio
            columns = ["step", "caption", "prediction"]

            # data should be a list of lists
            val_data_log.append([self.global_step,input_text , pred_text])
            # log the Table
            wandb_logger.log_table(key="val_samples", columns=columns, data=val_data_log)
        return predicted


    def training_step_all_together(self, batch):
        targets = batch['tokenized_captions']  # (B, seq_len)
        llm_inputs_embeds = self.proj_output(batch['image_embeddings'])# (B, 32, 2560)

        next_embeds = self.llm_model.model.embed_tokens(torch.tensor(self.COMMENT_TOKEN_ID, dtype=torch.int64).to(llm_inputs_embeds.device)).unsqueeze(0).unsqueeze(0) # 
        conc_inputs = torch.cat([llm_inputs_embeds, next_embeds], dim=1) # (B, 33, 2560)
        outputs_dict = self.llm_model(inputs_embeds = conc_inputs,
                                      labels = targets,
                                      return_dict = True) 
        loss = outputs_dict.loss
        del conc_inputs
        del next_embeds
        gc.collect()
        torch.cuda.empty_cache()
        return loss


    def training_step(self, batch):
        targets = batch['tokenized_captions']  # (B, seq_len)
        #print(targets.shape)
        max_size = len(targets)
        pos = 0
        loss = 0
        with autocast(True):
            llm_inputs_embeds = self.proj_output(batch['image_embeddings'])# (B, 32, 2560)
            #print(llm_inputs_embeds.dtype)
            #print(llm_inputs_embeds)
            next_embeds = self.llm_model.model.embed_tokens(torch.tensor(self.COMMENT_TOKEN_ID, dtype=torch.int64).to(llm_inputs_embeds.device)).unsqueeze(0).unsqueeze(0) # 
            #print(next_embeds.dtype)
            while pos < max_size:
                inputs = torch.cat([llm_inputs_embeds, next_embeds], dim=1) # (B, 33, 2560)
                #print(inputs.dtype)
                pred_logits, _ = self.llm_model.forward(inputs_embeds = inputs, return_dict=False) # (B, 33, 512000)
                pred_next_token_logits = pred_logits[:, -1, :]  # (B, 512000)
                #print(pred_next_token_logits)
                predicted_token = torch.argmax(pred_logits, axis=-1) # (B, 1)
                gt_token = targets[0,pos].contiguous().view(-1) # (B,1)
                #print(f"label:{gt_token} predicted:{predicted_token}")
                this_loss = torch.nn.functional.cross_entropy(pred_next_token_logits, gt_token, ignore_index = self.EOS_TOKEN_ID, label_smoothing=0.1)
                loss+=this_loss
                #print(loss)
                #if pos < 5:
                    #next_token = self.llm_model.model.embed_tokens(targets[pos]) # (1, 512000)
                    #print(next_token.shape)
                    #print(next_embeds.shape)
                next_embeds = torch.cat([next_embeds, self.llm_model.model.embed_tokens(targets[pos]).unsqueeze(0)],axis=1) # ()
                #else:
                #    next_embeds = torch.cat([next_embeds, self.llm_model.model.embed_tokens(predicted_token).unsqueeze(0)],axis=1)
                pos+=1
            loss = loss/max_size
        del inputs
        del next_embeds
        del pred_next_token_logits
        del pred_logits
        gc.collect()
        torch.cuda.empty_cache()
        #self.llm_model.to("cpu")
        self.log("train_loss", loss.item(), prog_bar=True)
        self.this_step_train_loss = loss.item()
        self.train_loss_values.append(self.this_step_train_loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        if batch_idx < self.num_validation_examples:
            predicted = self.evaluate(batch, "val")
            #if batch_idx % 10000 == 0:
            #    raw_img_path = batch['raw_image_path'][0]
            #    print(raw_img_path)
            #    image = Image.open(raw_img_path)
            #    plt.imshow(image)
            #    plt.show()
            #    #print("*****************************************")
            #    #print(f"{f'Input: ' :>12}{batch['caption']}")
            #    #print(f"{f'predicted: ' :>12}{predicted}")
            #    #print("*****************************************\n")


    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
    