import torch
from matplotlib import pyplot as plt




def get_style_embeddings(style_file):
    style_embed = torch.load(style_file)
    style_name = list(style_embed.keys())[0]
    return style_embed[style_name]


def get_EOS_pos_in_prompt(prompt):
    return len(prompt.split())+1


def invert_loss(gen_image):
    loss = torch.nn.functional.mse_loss(gen_image[:,0], gen_image[:,2]) + torch.nn.functional.mse_loss(gen_image[:,2], gen_image[:,1]) + torch.nn.functional.mse_loss(gen_image[:,0], gen_image[:,1])
    return loss


def blue_loss(images):
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,2] - 0.9).mean() # [:,2] -> all images in batch, only the blue channel
    return error


def show_images(images_list):
    # Let's visualize the four channels of this latent representation:
    fig, axs = plt.subplots(1, len(images_list), figsize=(16, 4))
    for c in range(len(images_list)):
        axs[c].imshow(images_list[c])
    plt.show()