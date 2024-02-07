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
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
import torchinfo
import torchmetrics
import wandb


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['projection_layer']
    for name, module in model.named_modules():
        #if any(mm_keyword in name for mm_keyword in multimodal_keywords):
        #    continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


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
        super(SimpleLinearBlock, self).__init__()
        self.pre_norm = nn.LayerNorm(in_size)
        self.proj = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.GELU())
        
    def forward(self,x):
        return self.proj(self.pre_norm(x))



class LitMultiModalPhiFineTune(LightningModule):
    """
    Pytorch Lightning module for Transformer

    """
    def __init__(self,
                 projection_layer_in_channels,
                 projection_layer_out_channels,
                 quantization_config,
                 num_validation_examples=2,
                 num_training_steps=100000):
        super(LitMultiModalPhiFineTune, self).__init__()
        self.quantization_config = quantization_config
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
        self.llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", quantization_config = quantization_config, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.vocab_size = 51200

        # Define LORA config
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=find_all_linear_names(self.llm_model),
            #target_modules=["q_proj",
            #                "k_proj",
            #                "v_proj",
            #                "fc1",
            #                "fc2"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM")

        self.train_loss_values = []
        self.val_loss_values = []
        self.COMMENT_TOKEN_ID = 23893
        self.IMAGE_TOKEN_ID = 5159
        self.EOS_TOKEN_ID = 50256
        self.image_embedding_dim = 49
        #self.prepare_model_for_finetuning()
        self.llm_model = get_peft_model(self.llm_model, self.lora_config)
        self.print_all_trainable_params()


    def prepare_model_for_finetuning(self):
        # prepare for 4 bit training
        self.llm_model.config.torch_dtype=torch.float32 
        #self.llm_model = prepare_model_for_kbit_training(self.llm_model, use_gradient_checkpointing=False)
        self.llm_model = get_peft_model(self.llm_model, self.lora_config)

        # convert the projection layer to float 16
        #self.projection_layer.to(dtype=torch.float16)

        # convert all modules in LLM to float16
        for name, module in self.llm_model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if module.weight.dtype == torch.float32:
                        module = module.to(torch.float16)


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
         
   

    def proj_output(self, image_embeds):
        return self.projection_layer(image_embeds)


    def print_all_trainable_params(self):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.llm_model.print_trainable_parameters()
        # print trainable paramaeters
        print("Number of Training Parameters")
        print("********************************")
        print(f"Projection Layer:{count_parameters(self.projection_layer)}")
        print(f"Phi Model:{count_parameters(self.llm_model)}")
        print("********************************")


    def generate(self, batch):
        batch_size = batch['ques_tokenized'].shape[0]

        image_embeddings = batch['image_embeddings']
        proj_outs = self.projection_layer(image_embeddings)
        device = proj_outs.device

        # define comment and im start tokens
        comment_token = torch.tensor(self.COMMENT_TOKEN_ID).repeat(batch_size, 1).to(device)
        comment = self.llm_model.model.model.embed_tokens(comment_token).to(device) # 

        im_start_token = torch.tensor(self.IMAGE_TOKEN_ID).repeat(batch_size, 1).to(device)
        im_start = self.llm_model.model.model.embed_tokens(im_start_token).to(device) # 

        question_tokens = batch['ques_tokenized']
        question_embeddings = self.llm_model.model.model.embed_tokens(question_tokens).to(device)

        # prepare input embeddings
        inputs_embeds = torch.cat([im_start, # <IM> [B x 1 x 2560]
                                  proj_outs, # [B x 49 x 2560]
                                  comment, # [B x 1 x 2560]
                                  question_embeddings, # [B x 1 x 2560]
                                  ], dim=1) # total dim: (B, 64, 2560)  
        with torch.no_grad():
            with autocast(True):
                pred_logits = self.llm_model.generate(inputs_embeds = inputs_embeds, max_new_tokens=64)
                generated_text = self.tokenizer.batch_decode(pred_logits, skip_special_tokens=True, clean_up_tokenization_spaces=True, verbose=False)[0]
        return generated_text
    
    
    def evaluate(self,batch, stage):
        if stage:
            predicted = self.generate(batch)
            #self.predicted_list.append(predicted)
            #self.expected_list.append(batch['caption'])
            # print the source, target, and the model output
            image_url = f"{f'Image URL: ' :>12}{batch['image_urls'][0]}"
            question = f"{f'Question: ' :>12}{batch['questions'][0]}"
            answer = f"{f'Answer:' :>12}{batch['answers'][0]}"
            pred_text = f"{f'Predicted:' :>12}{predicted}"
            print(f"*****************************************\n{image_url}\n{question}\n{answer}\n{pred_text}")
            # log a W&B Table that has a text caption, an image and audio
            #columns = ["step", "caption", "prediction"]

            # data should be a list of lists
            #val_data_log.append([self.global_step,input_text , pred_text])
            ## log the Table
            #wandb_logger.log_table(key="val_samples", columns=columns, data=val_data_log)
        return predicted


    def preprocess_inputs(self, batch):
        """
        input format: <IMAGE_TOKEN> [1 x 49 x 2560] <COMMENT_TOKEN> [1 x 13 x 2560]
        targets: [1 x 49 x 2560] <COMMENT_TOKEN> [1 x 14 x 2560]
        """
        # project image embeddings to llm dim
        batch_size = batch['ques_tokenized'].shape[0]
        image_embeddings_llm_input = self.proj_output(batch['image_embeddings'])# (1, 1, 49, 2560)
        device = image_embeddings_llm_input.device

        question_tokens = batch['ques_tokenized']  # (1, 13)
        answer_tokens = batch['ans_tokenized']  # (1, 13)
        im_start_token = torch.tensor(self.IMAGE_TOKEN_ID).repeat(batch_size, 1).to(device)
        comment_token = torch.tensor(self.COMMENT_TOKEN_ID).repeat(batch_size, 1).to(device)

        im_start = self.llm_model.model.model.embed_tokens(im_start_token).to(device) # 
        comment = self.llm_model.model.model.embed_tokens(comment_token).to(device) # 
        question_embeddings = self.llm_model.model.model.embed_tokens(question_tokens).to(device)
        # prepare input embeddings
        inputs_embeds = torch.cat([im_start, # <IM> [Bx 1 x 2560]
                                  image_embeddings_llm_input, # [B x 49 x 2560]
                                  comment, # [B x 1 x 2560]
                                  question_embeddings, # [B x n x 2560]
                                  ], dim=1) # total dim: (B, N, 2560)  
        # prepare labels
        labels = torch.cat([answer_tokens, # [B x M]
                            torch.tensor([self.EOS_TOKEN_ID]).repeat(batch_size,1).to(device), # [B x 1]
                            ], dim=1) # total dim: (B, 64)  
        return inputs_embeds, labels.to(device)

    
    def training_step(self, batch):
        inputs_embeds, labels = self.preprocess_inputs(batch)
        outputs_dict = self.llm_model(inputs_embeds = inputs_embeds,
                                      return_dict = True) 
        # extract only the answer portion from the outputs
        ans_start_idx = 1+batch['image_embeddings'].shape[1] # start, image tokens, comment followed by ans
        pred_logits = outputs_dict.logits[:,ans_start_idx:,:]
        pred_logits = pred_logits.reshape(-1,self.vocab_size)
        pred_loss = torch.nn.functional.cross_entropy(pred_logits, labels.contiguous().view(-1), label_smoothing=0.1, ignore_index=self.EOS_TOKEN_ID)
        del inputs_embeds
        gc.collect()
        torch.cuda.empty_cache()
        self.log("train_loss", pred_loss.item(), prog_bar=True)
        return pred_loss

    
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
    