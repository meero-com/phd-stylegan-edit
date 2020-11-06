import sys

sys.path.append('/home')

from asya_utils.model_tools.customdatasets import ImageTestSet
from asya_utils.model_tools.model_loaders import get_stylegan_models, get_encoder
from asya_utils.model_tools.losses import LogCoshLoss


import numpy as np
import torch
from torchvision import transforms

from pathlib import Path
from fastai.vision.all import get_image_files
from fastai.vision.data import imagenet_stats

import lpips
import os
from glob import glob
import argparse

import tqdm

from Inversion.pytorch_stylegan_encoder.models.image_to_latent import EvalImageLatentDataset


def eval_encoder(args):
    encoder_checkpoint = args.encoder_checkpoint

    # 1. Get our pre-trained models
    
    if int(encoder_checkpoint.split('_')[-1].split('.')[0]) == 1:
        input_size = 224
    else:
        input_size = 256

    
    # get image_to_latent encoder
    image_to_latent = get_encoder(encoder_checkpoint, input_size=input_size)
    image_to_latent.eval()


    # get the stylegan2 pretrained model 
    styleganmodel, dis, avglat = get_stylegan_models()
    styleganmodel.eval()
    
    
    # 2. get the necessary dataset, datatransforms, etc
    
    # this is the arguments to display the input image (simply to be able to view it again side-by-side with output)
    augments_big = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[.5, .5, .5])
    ])

    # this is for the input into the resnet model
    augments_small = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_stats[0],
                             std=imagenet_stats[1])
    ])
    
    augments_encoder = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])
    ])
    
    
    if args.dataset == 'real':
        parent_directory = '/home/data/images'
        aligned_image_path = os.path.join(parent_directory, 'aligned_images/')

        test_filenames = sorted(glob(aligned_image_path + "/*.png"))
        test_dataset = ImageTestSet(test_filenames, transforms1=augments_big, transforms2=augments_small)
        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize)
    
    
    elif args.dataset == 'latent':

        parent = '/home/datadrive/asya/datasets/StyleGAN2_ffhq_latent_image_pairs'
        basename = 'latent_image_pairs_500000_1024_1'

        data_dir = Path(parent) / basename
        if not os.path.exists(data_dir):
            raise ValueError("Basename directory does not exist. Please verify.")

        filenames = sorted(get_image_files(data_dir))
        dlatents = np.load(str(data_dir/'w.npy')) # shape: (nb_samples, 512) or (nb_samples, latent_n, 512) if latent_n != 1


        test_ratio = 0.03
        cutoff = int(len(filenames) * (1 - test_ratio))
        test_filenames = filenames[cutoff:]
        test_latents = dlatents[cutoff:]
        
        test_dataset = EvalImageLatentDataset(test_filenames, test_latents, transforms1=augments_encoder, transforms2=augments_big)
        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize)
        
    
    elif args.dataset == 'celebahq':
        celeba_path = Path('/home/datadrive/asya/datasets/celeba_hq/val')
        celeba_files = get_image_files(celeba_path)
        celeba = ImageTestSet(celeba_files,
                    transforms1=augments_big,
                    transforms2=augments_small)
        dataloader = torch.utils.data.DataLoader(celeba, batch_size=args.batchsize)

    else:
        raise ValueError("error")
    
    
    
    
    # define criterions
    latent_criterion = LogCoshLoss()
    lpips_criterion = lpips.LPIPS(net='alex')
    mse_criterion = torch.nn.MSELoss()
    
    
    
    
    
    
    # let's evaluate
    all_mse = []
    all_lpips = []
    all_lat = []

    progress_bar = tqdm.tqdm(dataloader)
    for i, data in enumerate(progress_bar):
        #print(i, '/', len(dataloader))
        # if the dataset is real or celebahq, then data is: (imbig, imsmall)
        # otherwise data is: (imsmall, imbig, latents)
        
        if args.dataset in ['celebahq', 'real']:
            imbig, imsmall = data
            latents = torch.zeros(512)
 
        if args.dataset == 'latent':
            imsmall, imbig, latents = data
        
        if input_size == 256:
            # make the latents into (batchsize, 18, 512)
            latents = latents.repeat(1, 18).reshape((-1, 18, 512))
                
        pred_latents = image_to_latent(imsmall.to('cuda'))
        pred_images, _ = styleganmodel([pred_latents.to('cuda')], input_is_latent=True)
        
        lat_loss =  latent_criterion(pred_latents.to('cpu'), latents)
        lpips_loss = lpips_criterion(pred_images.to('cpu'), imbig).reshape((-1)).mean()
        mse_loss = mse_criterion(pred_images.to('cpu'), imbig)
        
        all_mse.append(mse_loss.item())
        all_lpips.append(lpips_loss.item())
        all_lat.append(lat_loss.item())
        
    
        progress_bar.set_description("mse: {:.3}, lpips: {:.3f}, latent : {:.3f}".
                                     format(mse_loss.item(),
                                            lpips_loss.item(),
                                            lat_loss.item()))
        
            
    np.save('/home/data/results/expes/encoder/pipscores_{}_{}.npy'.format(args.dataset, input_size), np.array(all_lpips))
    np.save('/home/data/results/expes/encoder/msescores_{}_{}.npy'.format(args.dataset, input_size), np.array(all_mse))
    np.save('/home/data/results/expes/encoder/latscores_{}_{}.npy'.format(args.dataset, input_size), np.array(all_lat))
       
            
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_checkpoint', type=str, required=True, default='latent_image_pairs_500000_1024_1.pt', help='The basename of the encoder checkpoint. By default encodes the latent codes of [1,512] dimension')
    
    parser.add_argument('--dataset', type=str, default='celebahq', help='The dataset we wish to evaluate on. This can either be [celebahq, imagelatent, real]')
    
    parser.add_argument('--batchsize', type=int, default=16, help='Batch size for image generation')
    
    args = parser.parse_args()
    
    eval_encoder(args)

    
    
    
    
    
# python metrics.py --encoder_checkpoint latent_image_pairs_100000_1024_18.pt --dataset celebahq --batchsize 8

# python metrics.py --encoder_checkpoint latent_image_pairs_500000_1024_1.pt --dataset celebahq --batchsize 8