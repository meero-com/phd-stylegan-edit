import argparse
from fastai.vision.all import get_image_files, imagenet_stats
from pathlib import Path
import os
import re

from torchvision import transforms
import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from PIL import Image

import sys
sys.path.append('/home')

# our models for image to latent learning
from Inversion.pytorch_stylegan_encoder.models.image_to_latent import ImageToLatent, ImageLatentDataset
from asya_utils.model_tools.losses import LogCoshLoss

import wandb

# for stylegan2 model
from model import Generator

import setproctitle
setproctitle.setproctitle("[asya] - image2latent")

def train(model, train_generator, validation_generator, optimizer, criterion, config, styleganmodel=None):
    if config.resume_checkpoint:
        print("resuming checkpoint")
        step, min_validation_loss, model_state_dict = torch.load(config.checkpoint)
        model.load_state_dict(model_state_dict)
    else:
        step = 0
        min_validation_loss = float('inf')


    nb_images_to_show = 5
    validation_loss = 0.0
    one_time = 1

    progress_bar = tqdm(range(config.epochs))
    
    for epoch in progress_bar:    
        running_loss = 0.0

        model.train()
        for i, (images_cpu, latents_cpu) in enumerate(train_generator, 1):
            optimizer.zero_grad()

            images, latents = images_cpu.cuda(), latents_cpu.cuda()
            
            #print("images :", images)
            #print("imagse shape:", images.shape)
                
            pred_latents = model(images)

            loss = criterion(pred_latents, latents)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            train_metrics = {'(Train) Average Running Loss': running_loss / i, 'Train Loss': loss.item()}

            wandb.log(train_metrics, step=step)

            progress_bar.set_description("Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))

            # visualize images if we gave stylegan model as input
            if (styleganmodel is not None) and (i - 1) % 100 == 0:
                
                # let's visualize random images 
                randinds = np.random.randint(0, pred_latents.shape[0], nb_images_to_show)

                output, _ = styleganmodel([pred_latents[randinds]], input_is_latent=True)
                pred_image = output.detach().clamp_(min=-1, max=1).add(1).div_(2)#.mul(255)#.to('cpu').numpy()
                pred_image = F.interpolate(pred_image, size=256).to('cpu')

                wandb.log({"train images": [wandb.Image(images_cpu[randinds], caption="real image"), wandb.Image(pred_image, caption="predicted projection")]}, step=step)

            if i % 5000 == 0:
                # save model
                print("saving model")
                torch.save([step, min_validation_loss, model.state_dict()], '/mnt/datadrive/asya/checkpoints/stylegan2/latent_encoder/temp_checkpoint.pt')

            step += 1

        validation_loss = 0.0

        model.eval()
        for i, (images_cpu, latents_cpu) in enumerate(validation_generator, 1):
            with torch.no_grad():
                images, latents = images_cpu.cuda(), latents_cpu.cuda()

                pred_latents = model(images)
                loss =  criterion(pred_latents, latents)

                validation_loss += loss.item()

        validation_loss /= i

        if styleganmodel is not None:
            # get nb_images_to_show random indexes
            randinds = np.random.randint(0, pred_latents.shape[0], nb_images_to_show)

            output, _ = styleganmodel([pred_latents[randinds]], input_is_latent=True)
            pred_image = output.detach().clamp_(min=-1, max=1).add(1).div_(2)#.mul(255)#.to('cpu').numpy()
            pred_image = F.interpolate(pred_image, size=256).to('cpu')
            wandb.log({"validation images": [wandb.Image(images_cpu[randinds], caption="real image"), wandb.Image(pred_image, caption="predicted projection")]}, step=step)

        valid_metrics = {'(Validation) Average Running Loss': validation_loss}
        wandb.log(valid_metrics, step=step)
        progress_bar.set_description("Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))

        # we save our model at the end of each epoch if the validation loss has been improved
        if validation_loss < min_validation_loss:
            print("New best model, saving.")
            # let's save this model 
            torch.save([step, min_validation_loss, model.state_dict()], config.checkpoint)
            

def verify_basename(basename):
    r = re.compile('latent_image_pairs_[0-9]{1,6}_[0-9]{1,4}_1[8]*')
    if r.match(basename) is not None:
        return True
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basename', type=str, required=True, help='The basename image/latent pairs folder contained in parent folder')
    parser.add_argument('--parent', type=str, default='/home/datadrive/asya/datasets/StyleGAN2_ffhq_latent_image_pairs', help='This is the parent directory of all of our image/latent pairs. By default it is "/home/datadrive/asya/datasets/StyleGAN2_latent_image_pairs". The format of the subfolders containing the data here are all named latent_image_pairs_{nb_samples}_{image_size}_{latent_dimension}. This format must be respected.')
    
    parser.add_argument('--checkpoint_dir', type=str, default='/home/datadrive/asya/checkpoints/stylegan2/latent_encoder')
    parser.add_argument('--stylegan_ckpt', type=str, default='/home/datadrive/asya/checkpoints/stylegan2/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--resume_training', default=False, action='store_true', help="if we want to to resume training (from checkpoint: checkpoint_dir/image_to_latent.pt)")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    
    args = parser.parse_args()
    
    if not verify_basename(args.basename):
        raise ValueError("Please verify the basename argument (it must respect format 'latent_image_pairs_<nb_samples>_<image_size>_<latent_dim>')")
    
    _, _, _, nb_samples, im_size, latent_n = args.basename.split('_')
    nb_samples = int(nb_samples)
    im_size = int(im_size)
    latent_n = int(latent_n)
    
    #####################
    ### data loading: ###
    #####################
    
    data_dir = Path(args.parent) / args.basename
    if not os.path.exists(data_dir):
        raise ValueError("Basename directory does not exist. Please verify.")
        
    filenames = sorted(get_image_files(data_dir))
    dlatents = np.load(str(data_dir/'w.npy')) # shape: (nb_samples, 512) or (nb_samples, latent_n, 512) if latent_n != 1
    
    # this will be used to save checkpoints as well as to save the wandb files
    checkpoint_dir = Path(args.checkpoint_dir)
    
    
    #######################################
    #### hyperparameters + dataloaders ####
    #######################################
    
    val_ratio = args.val_ratio
    cutoff = int(nb_samples * (1 - val_ratio))
    batch_size = args.batch_size
    learning_rate = args.lr

    image_resize = 224 # this is so that resnet works
    
    augments = transforms.Compose([
        transforms.Resize(image_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])
    ])
    
    train_filenames = filenames[:cutoff]
    val_filenames = filenames[cutoff:]
    
    train_latents = dlatents[:cutoff]
    val_latents = dlatents[cutoff:]
    
    train_dataset = ImageLatentDataset(train_filenames, train_latents, transforms=augments)
    val_dataset = ImageLatentDataset(val_filenames, val_latents, transforms=augments)
    
    train_gen = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_gen = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    #####################
    ### define model  ###
    #####################


    # Instantiate model

    image_to_latent = ImageToLatent(latent_n).cuda()
    optimizer = torch.optim.Adam(image_to_latent.parameters(), lr=learning_rate)
    criterion = LogCoshLoss() # this loss basically just compares the predicted latent to the true latent by doing an operation on their difference.

    # train model
    config = dict(
        epochs = 20,
        learning_rate = learning_rate,
        checkpoint = checkpoint_dir/'{}.pt'.format(args.basename),
        resume_checkpoint = args.resume_training
    )

    wandb.init(config=config, project="image_to_latent", resume=args.resume_training, dir=str(checkpoint_dir))
    config = wandb.config
    wandb.watch(image_to_latent)

    # get the stylegan2 pretrained model 
    args_dict = {}
    args_dict['device'] = 'cuda'
    args_dict['size'] = im_size
    args_dict['latent'] = 512
    args_dict['n_mlp'] = 8
    args_dict['n_latent'] = latent_n
    args_dict['channel_multiplier'] = 2
    args_dict['ckpt'] = args.stylegan_ckpt
    styleganmodel = Generator(
        args_dict['size'], args_dict['latent'], args_dict['n_mlp'], channel_multiplier=args_dict['channel_multiplier']
    ).to(args_dict['device'])
    checkpoint = torch.load(args_dict['ckpt'])
    styleganmodel.load_state_dict(checkpoint['g_ema'])

    train(image_to_latent, train_gen, val_gen, optimizer, criterion, config, styleganmodel)



# 