import sys
sys.path.append('/home')
path_to_stylegan = '/home/stylegan2-pytorch'

from Inversion.pytorch_stylegan_encoder.models.image_to_latent import ImageToLatent
import os
import torch
from stylegan2.model import Generator, Discriminator

def get_stylegan_models(device='cuda', 
                        size=1024, 
                        strict=True, 
                        checkpoint='/home/datadrive/asya/checkpoints/stylegan2/stylegan2-ffhq-config-f.pt'):
    
    # get the stylegan2 pretrained model, these parameters almost never change
    args_dict = {}
    args_dict['size'] = size
    args_dict['latent'] = 512
    args_dict['n_mlp'] = 8
    args_dict['channel_multiplier'] = 2
    args_dict['ckpt'] = checkpoint
    
    stylegan_gen = Generator(
        args_dict['size'], args_dict['latent'], args_dict['n_mlp'], channel_multiplier=args_dict['channel_multiplier']
    ).to(device)
    stylegan_gen.eval()
    
    stylegan_disc = Discriminator(size=args_dict['size'], channel_multiplier=args_dict['channel_multiplier']).to(device)
    stylegan_disc.eval()
    
    checkpoint = torch.load(args_dict['ckpt'])
    
    stylegan_gen.load_state_dict(checkpoint['g_ema'], strict)
    stylegan_disc.load_state_dict(checkpoint['d'], strict)
    
    try:
        latent_avg = checkpoint['latent_avg']
    except KeyError:
        print("Error, latent average is not in checkpoint. Returning None for this value")
        latent_avg = None
    
    return stylegan_gen, stylegan_disc, latent_avg



def get_encoder(checkpoint_basename, 
                device='cuda', 
                checkpoint_dir='/home/datadrive/asya/checkpoints/stylegan2/latent_encoder', 
                input_size=224):
    
    fullpath = os.path.join(checkpoint_dir, checkpoint_basename)
    
    if not os.path.exists(fullpath):
        print("error, encoder does not exist")
        raise ValueError
    
    name, _ = os.path.splitext(checkpoint_basename)
    latent_n = int(name.split('_')[-1])
    
    image_to_latent = ImageToLatent(latent_n, input_size=input_size).to(device)
    print('input_size:', input_size)
    
    _, _, state_dict = torch.load(fullpath)
    image_to_latent.load_state_dict(state_dict)
    return image_to_latent