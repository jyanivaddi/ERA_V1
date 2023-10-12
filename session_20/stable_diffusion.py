import torch
from torchvision.transforms import ToTensor
from utils import get_style_embeddings, get_EOS_pos_in_prompt, invert_loss 
from base64 import b64encode
import numpy as np
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
from PIL import Image
import os
import cv2
import torchvision.transforms as T


class StableDiffusion:
    def __init__(self, torch_device, num_inference_steps=30, height=512, width=512, guidance_scale=7.5):
        # Load the autoencoder
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder='vae')
    
        # Load tokenizer and text encoder to tokenize and encode the text
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    
        # Unet model for generating latents
        unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder='unet')
    
        # Noise scheduler
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
        # Move everything to GPU
        self.torch_device = torch_device
        self.vae = vae.to(self.torch_device)
        self.text_encoder = text_encoder.to(self.torch_device)
        self.unet = unet.to(self.torch_device)

        # additional properties
        self.num_inference_steps = num_inference_steps
        self.height = height                        # default height of Stable Diffusion
        self.width = width                         # default width of Stable Diffusion
        self.guidance_scale = guidance_scale                # Scale for classifier-free guidance


    # Prep Scheduler
    def set_timesteps(self):
        self.scheduler.set_timesteps(self.num_inference_steps)
        self.scheduler.timesteps = self.scheduler.timesteps.to(torch.float32) # minor fix to ensure MPS compatibility, fixed in diffusers PR 3925
        

    def additional_guidance(self, latents, noise_pred, t, sigma, custom_loss_fn, custom_loss_scale):
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
        loss = custom_loss_fn(denoised_images) * custom_loss_scale

        # Get gradient
        cond_grad = torch.autograd.grad(loss, latents, allow_unused=False)[0]

        # Modify the latents based on this gradient
        latents = latents.detach() - cond_grad * sigma**2
        return latents, loss


    def generate_with_embs(self, text_embeddings, max_length, random_seed, custom_loss_fn, custom_loss_scale):

        generator = torch.manual_seed(random_seed)   # Seed generator to create the inital latent noise
        batch_size = 1

        uncond_input = self.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prep Scheduler
        self.set_timesteps()

        # Prep latents
        latents = torch.randn( (batch_size, self.unet.in_channels, self.height // 8, self.width // 8), generator=generator,)
        latents = latents.to(self.torch_device)
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
            if custom_loss_fn is not None:
                if i%2 == 0:
                    latents, custom_loss = self.additional_guidance(latents, noise_pred, t, sigma, custom_loss_fn, custom_loss_scale)
                    print(i, 'loss:', custom_loss.item())

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return self.latents_to_pil(latents)[0]


    def get_output_embeds(self, input_embeddings):
        # CLIP's text model uses causal mask, so we prepare it here:
        bsz, seq_len = input_embeddings.shape[:2]
        causal_attention_mask = self.text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)

        # Getting the output embeddings involves calling the model with passing output_hidden_states=True
        # so that it doesn't just return the pooled final predictions:
        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=None, # We aren't using an attention mask so that can be None
            causal_attention_mask=causal_attention_mask.to(self.torch_device),
            output_attentions=None,
            output_hidden_states=True, # We want the output embs not the final output
            return_dict=None,
        )

        # We're interested in the output hidden state only
        output = encoder_outputs[0]

        # There is a final layer norm we need to pass these through
        output = self.text_encoder.text_model.final_layer_norm(output)

        # And now they're ready!
        return output


    def pil_to_latent(self, input_im):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        with torch.no_grad():
            latent = self.vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(self.torch_device)*2-1) # Note scaling
        return 0.18215 * latent.latent_dist.sample()


    def latents_to_pil(self, latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images


    def generate_image_with_custom_style(self, prompt, style_token_embedding=None, random_seed=41, custom_loss_fn = None, custom_loss_scale=None):
        eos_pos = get_EOS_pos_in_prompt(prompt)

        # tokenize
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        max_length = text_input.input_ids.shape[-1]
        input_ids = text_input.input_ids.to(self.torch_device)

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
        modified_output_embeddings = self.get_output_embeds(input_embeddings)

        # And generate an image with this:
        generated_image = self.generate_with_embs(modified_output_embeddings, max_length, random_seed, custom_loss_fn, custom_loss_scale)
        return generated_image
        