import torch
from torchvision.transforms import ToTensor
from utils import set_timesteps, get_output_embeds, get_style_embeddings, get_EOS_pos_in_prompt, invert_loss, pil_to_latent, latents_to_pil
from base64 import b64encode
import numpy as np
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import os
import cv2
import torchvision.transforms as T


class StableDiffusion:
    def __init__(self, num_inference_steps=30, height=512, width=512, guidance_scale=7.5, custom_loss_fn=None, custom_loss_scale=100.0):
        # Load the autoencoder
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder='vae')
    
        # Load tokenizer and text encoder to tokenize and encode the text
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    
        # Unet model for generating latents
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder='unet')
    
        # Noise scheduler
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
        # Move everything to GPU
        self.vae = vae.to(torch_device)
        self.text_encoder = text_encoder.to(torch_device)
        self.unet = unet.to(torch_device)

        # additional properties
        self.num_inference_steps = num_inference_steps
        self.height = height                        # default height of Stable Diffusion
        self.width = width                         # default width of Stable Diffusion
        self.guidance_scale = guidance_scale                # Scale for classifier-free guidance
        self.custom_loss_fn = custom_loss_fn
        self.custom_loss_scale = custom_loss_scale



    def additional_guidance(self, latents, noise_pred, t, sigma):
        #### ADDITIONAL GUIDANCE ###
        # Requires grad on the latents
        latents = latents.detach().requires_grad_()

        # Get the predicted x0:
        latents_x0 = latents - sigma * noise_pred
        #print(f"latents: {latents.shape}, noise_pred:{noise_pred.shape}")
        #latents_x0 = scheduler.step(noise_pred, t, latents).pred_original_sample

        # Decode to image space
        denoised_images = self.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)

        # Calculate loss
        loss = self.custom_loss_fn(denoised_images) * self.custom_loss_scale

        # Get gradient
        cond_grad = torch.autograd.grad(loss, latents, allow_unused=False)[0]

        # Modify the latents based on this gradient
        latents = latents.detach() - cond_grad * sigma**2
        return latents, loss


    def generate_with_embs(self, text_embeddings, max_length, random_seed, loss_fn = None):

        generator = torch.manual_seed(random_seed)   # Seed generator to create the inital latent noise
        batch_size = 1

        uncond_input = self.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prep Scheduler
        set_timesteps(self.scheduler, self.num_inference_steps)

        # Prep latents
        latents = torch.randn( (batch_size, unet.in_channels, self.height // 8, self.width // 8), generator=generator,)
        latents = latents.to(torch_device)
        latents = latents * self.scheduler.init_noise_sigma

        # Loop
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.scheduler.sigmas[i]
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            if loss_fn is not None:
                if i%2 == 0:
                    latents, custom_loss = self.additional_guidance(latents, noise_pred, t, sigma, loss_fn)
                    print(i, 'loss:', custom_loss.item())

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents_to_pil(latents)[0]


    def generate_image_with_custom_style(self, prompt, style_token_embedding=None, random_seed=41):
        eos_pos = get_EOS_pos_in_prompt(prompt)

        # tokenize
        text_input = self.tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        max_length = text_input.input_ids.shape[-1]
        input_ids = text_input.input_ids.to(torch_device)

        # get token embeddings
        token_emb_layer = self.text_encoder.text_model.embeddings.token_embedding
        token_embeddings = token_emb_layer(input_ids)

        # Append style token towards the end of the sentence embeddings
        if style_token_embedding is not None:
            token_embeddings[-1, eos_pos, :] = style_token_embedding

        # combine with pos embs
        pos_emb_layer = self.text_encoder.text_model.embeddings.position_embedding
        position_ids = self.text_encoder.text_model.embeddings.position_ids[:, :77]
        position_embeddings = pos_emb_layer(position_ids)
        input_embeddings = token_embeddings + position_embeddings

        #  Feed through to get final output embs
        modified_output_embeddings = get_output_embeds(input_embeddings)

        # And generate an image with this:
        generated_image = self.generate_with_embs(modified_output_embeddings, max_length, random_seed)
        return generated_image
        