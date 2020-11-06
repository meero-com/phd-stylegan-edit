'''
I wrote this new version on august 26th in order to add perceptual loss into my training. I decided to start from 
scratch writing this which is why I did not modify the original train_w (which should still be working). 
'''

import setproctitle
setproctitle.setproctitle("[asya] latent projection")

import sys
sys.path.append('/home')


from asya_utils.processing.image_tools import load_images, images_to_video 
from asya_utils.processing.network_postproc import PostSynthesisProcessing, GeneratedImageHook
from asya_utils.model_tools.model_loaders import get_stylegan_models, get_encoder
from asya_utils.model_tools.losses import PerceptualLoss

import argparse
from fastai.vision.data import imagenet_stats
from torchvision import transforms
from PIL import Image
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
import math 
import lpips

import wandb
from tqdm import tqdm 

import os
import glob
import numpy as np

import pprint

import shutil

# based on this code: https://github.com/jacobhallberg/pytorch_stylegan_encoder/blob/master/encode_image.py

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

def get_number(run_name):
    num = run_name.split('_run_')
    try:
        return int(num[1])
    except Exception:
        return 0

def get_next_number(basepath):
    max_number = 0
    current_runs = glob.glob(basepath + "_run_*")
    numbers = sorted([get_number(run) for run in current_runs])
    if len(numbers) > 0:
        max_number = numbers[-1]
        # make sure there is a 'ckpt.pt' file inside
        if not os.path.exists(os.path.join(basepath + '_run_{}'.format(max_number), 'ckpt.pt')):
            max_number -= 1
    return max_number + 1


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise


# here we will train noise and latent
def train(latent, noises, ref_image, gen, disc, config):
    
    init_lat = latent.clone()
    
    train_joint = (config.opt_w and config.opt_noise) or (config.opt_z and config.opt_noise)
    train_noise = not train_joint and config.opt_noise
    train_latent = not train_joint and not train_noise and (config.opt_z or config.opt_w)
    
    # this is for the generator to know if it is taking w or z as input
    if train_latent or train_joint:
        input_is_w = config.opt_w
    else:
        input_is_w = "opt_w" in config.opt_noise_from
    
    print("input is w?", input_is_w)
    #train_joint = True
    
    if train_joint:
        print("optimizing noise and latent vector simultaneously")
        parameters_to_optimize = noises + [latent] 
        latent.requires_grad = True
        for noise in noises:
            noise.requires_grad = True
    elif train_noise:
        print("optimizing noise")
        parameters_to_optimize = noises
        latent.requires_grad = False
        for noise in noises:
            noise.requires_grad = True
    elif train_latent:
        print("optimizing latent")
        parameters_to_optimize = [latent]
        latent.requires_grad = True
        for noise in noises:
            noise.requires_grad = False
        
    

    
    
    # create zero noises for visualization
    zero_noise = gen.make_noise()
    for noise in zero_noise:
        noise *= 0
    
    # here, w can also be z, just a little default thing
    
    
    
    # parameters that are here par default mais peut etre a changer
    # config.use_noise = True # to change maybe # now this is an argument parameter
    config.pro_model = True
    config.noise = 0.05
    config.noise_ramp = 0.75
    #config.noise_regularize = 1e5
    
    step = 1
    
    # show the initial w
    img_gen, _ = gen([latent], input_is_latent=input_is_w, noise=noises)
    img_gen_no_noise, _ = gen([latent], input_is_latent=input_is_w, noise=zero_noise)
    wandb.log({"Image Prediction": [wandb.Image(ref_image, caption="input image"), wandb.Image(img_gen, caption="predicted image"),  
                                    wandb.Image(img_gen_no_noise, caption="predicted image - noise")]}
              , step=step)


    # for video
    if config.video:
        postprocess = PostSynthesisProcessing()
        generated_image_hook = GeneratedImageHook(postprocess, every_n=10, change_factor=10)
    
        
    # MODEL DEFINITION !!!
    
    optimizer = optim.Adam(parameters_to_optimize, lr=config.lr)
    
    
#     if config.pro_model:
#         optimizer = optim.Adam(parameters_to_optimize, lr=config.lr) # from projector.py

#     else:
#         optimizer = torch.optim.SGD(parameters_to_optimize, lr=config.lr) 
        
    scheduler = StepLR(optimizer, step_size=500, gamma=0.05)
        
    
    # OK HERE IS THE FUN PART ! let's try to use our perceptual discriminator loss and see what happens
    #percept = PerceptualDiscLoss(disc, pattern=config.pattern, fact=config.percept_fact, layers=config.layers)
    percept = PerceptualLoss('disc')
    percept_vgg = PerceptualLoss('vgg')

    latent_std = .5
    progress_bar = tqdm(range(step, config.nb_it))
    
    
    
    for step in progress_bar:
        
        if config.pro_model:
            t = step / config.nb_it
            lr = get_lr(t, config.lr)
            optimizer.param_groups[0]['lr'] = lr
        
#         if train_joint:
#             noise_strength = latent_std * config.noise * max(0, 1 - t / config.noise_ramp) ** 2
#             latent_n = latent_noise(latent, noise_strength) # latent noise
#             n_loss = 0 #noise_regularize(noises)
        
#         else:
        latent_n = latent
        n_loss = noise_regularize(noises)
        
        
        if not input_is_w:
            latent_n = latent_n.reshape((18, 512))
            latent_n = gen.style(latent_n) #(1, 18, 512)
            latent_n = latent_n.reshape((1, 18, 512))
        
            
        # in any case input_is_latent is True now
        img_gen, _ = gen([latent_n], input_is_latent=True, noise=noises)

        if config.video:
            im_proced = postprocess(img_gen) # this is for video
            
            
        ######
        ###### losses
        ######
        
        if config.pdisc_coef != 0:
            p_loss = percept.forward_for_keys(img_gen, ref_image, keys=['conv_7'])
        else:
            p_loss = torch.tensor(0)
        
        p_loss_vgg = percept_vgg.forward_for_keys(F.interpolate(img_gen, size=224), 
                                                  F.interpolate(ref_image, size=224), 
                                                  keys=['conv11', 'conv12', 'conv32', 'conv42']) # to be able to input it into the vgg network
        
        
        rescale_imgen = ((img_gen - img_gen.min()) / (img_gen.max() - img_gen.min())) * 255.
        rescale_ref_image = ((ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())) * 255.
        mse_loss = F.mse_loss(rescale_imgen, rescale_ref_image)
        
        init_loss =  torch.sum((latent_n - init_lat)**2)
        
        
        loss =  config.pdisc_coef * p_loss + config.pvgg_coef * p_loss_vgg + config.mse_coef * mse_loss  + n_loss * config.noise_reg + config.init_reg * init_loss
        

        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        optimizer.step()
        scheduler.step()
        progress_bar.set_description("Step: {}, Loss: {:.5f}, Pdisc_loss (m): {:.2f} ({:.4f}), Pvgg_loss (m): {:.2f} ({:.5f}), mse_loss (m): {:.5f} ({:.5f})".
                                     format(step, loss, p_loss.item(), 
                                            config.pdisc_coef * p_loss.item(), 
                                            p_loss_vgg.item(), 
                                            config.pvgg_coef * p_loss_vgg.item(), 
                                            mse_loss.item(), 
                                            config.mse_coef * mse_loss.item()))
        wandb.log({"loss": loss, 
                   "pdisc_loss": p_loss.item(), 
                   "mse_loss": mse_loss.item(), 
                   "pvgg_loss": p_loss_vgg.item(), 
                   "pdisc_mult": p_loss.item() * config.pdisc_coef, 
                   "vgg_mult": config.pvgg_coef * p_loss_vgg.item(), 
                   'mse_mult': config.mse_coef * mse_loss.item(), 
                   'learning_rate': optimizer.param_groups[0]['lr'], 
                   'nloss (m)': n_loss * config.noise_reg, 
                   'nloss': n_loss,
                   'initloss': init_loss,
                   'initloss (m)': config.init_reg * init_loss}, step=step)
        
        
        # show the image every 100 steps
        if step % 100 == 0:
            img_gen_no_noise, _ = gen([latent_n], input_is_latent=True, noise=zero_noise)
            wandb.log({"Image Prediction": 
                       [wandb.Image(ref_image, caption="input image"), 
                        wandb.Image(img_gen, caption="predicted image"), 
                        wandb.Image(img_gen_no_noise, caption="predicted image - noise")]}, 
                      step=step)

            # save the model
            torch.save([step, latent, noises], config.checkpoint)

    print("saving model in ", config.checkpoint)
    torch.save([step, latent, noises], config.checkpoint)


    if config.video:
        print("saving video")
        images_to_video(generated_image_hook.get_images(), config.video_path, image_size=1024)



# # here, we first train the w, then we train the noises     
# def train_alternate(latent, ref_image, gen, disc, config):
#     zero_noises = gen.make_noise()
#     for noise in zero_noises:
#         noise *= 0
            
            
#     print("alternating training strategy")
#     # here, w can also be z, just a little default thing
#     latent.requires_grad = True
    
#     # parameters that are here par default mais peut etre a changer
#     # config.use_noise = True # to change maybe # now this is an argument parameter
#     config.pro_model = True
#     config.noise = 0.05
#     config.noise_ramp = 0.75
#     config.noise_regularize = 1e5
#     config.mse = 0.9
    
#     step = 0
    
#     # show the initial w
#     generated_image, _ = gen([latent], input_is_latent=is_w)
#     wandb.log({"Image Prediction": [wandb.Image(ref_image, caption="input image"), wandb.Image(generated_image, caption="predicted image")]}, step=step)

#     # for video
#     if config.video:
#         postprocess = PostSynthesisProcessing()
#         generated_image_hook = GeneratedImageHook(postprocess, every_n=10, change_factor=10)
    

#     noises = []
        

#     optimizer = optim.Adam([latent], lr=config.lr)
    
        
#     scheduler = StepLR(optimizer, step_size=500, gamma=0.05)
        
    
#     # OK HERE IS THE FUN PART ! let's try to use our perceptual discriminator loss and see what happens
#     percept = PerceptualDiscLoss(disc, pattern=config.pattern, fact=config.percept_fact, layers=config.layers)
    
#     percept_vgg = lpips.PerceptualLoss(
#         model='net-lin', net='vgg', use_gpu=True
#         )

#     latent_std = .5
#     progress_bar = tqdm(range(step, config.nb_it))
    
    
#     for step in progress_bar:

#         if config.pro_model:
#             t = step / config.nb_it
#             lr = get_lr(t, config.lr)
#             optimizer.param_groups[0]['lr'] = lr

#         noises = None
#         latent_n = latent
#         n_loss = 0

#         img_gen, _ = gen([latent_n], input_is_latent=is_w, noise=noises)

#         if config.video:
#             im_proced = postprocess(img_gen) # this is for video
        
#         p_loss = percept(img_gen, ref_image)
#         #p_loss = percept(img_gen, ref_image).sum()
        
#         interpolate_size = 256
#         p_loss_vgg = percept_vgg(F.interpolate(img_gen, size=interpolate_size), F.interpolate(ref_image, size=interpolate_size))


#         mse_loss = F.mse_loss(img_gen, ref_image)
        
#         loss = config.pdisc_coef * p_loss + config.pvgg_coef * p_loss_vgg + config.mse_coef * mse_loss

#         optimizer.zero_grad()
#         loss.backward()
#         loss = loss.item()
#         optimizer.step()
#         scheduler.step()
#         progress_bar.set_description("Step: {}, Loss: {:.5f}, Pdisc_loss (m): {:.2f} ({:.4f}), Pvgg_loss (m): {:.2f} ({:.5f}), mse_loss (m): {:.5f} ({:.5f})".format(step, loss, p_loss.item(), config.pdisc_coef * p_loss.item(), p_loss_vgg.item(), config.pvgg_coef * p_loss_vgg.item(), mse_loss.item(), config.mse_coef * mse_loss.item()))
#         wandb.log({"loss": round(loss, 2), "pdisc_loss": round(p_loss.item(), 2), "mse_loss": round(mse_loss.item(), 2), "pvgg_loss": round(p_loss_vgg.item(), 2), "pdisc_mult": round(p_loss.item() * config.pdisc_coef, 2), "vgg_mult": round(config.pvgg_coef * p_loss_vgg.item(), 2), 'mse_mult': round(config.mse_coef * mse_loss.item(), 2), 'learning_rate': round(optimizer.param_groups[0]['lr'], 4)}, step=step)
        
#         # show the image every 100 steps
#         if step % 100 == 0:
#             generated_image, _ = gen([w], input_is_latent=is_w, noise=zero_noises)
#             wandb.log({"Image Prediction": [wandb.Image(ref_image, caption="input image"), wandb.Image(img_gen, caption="predicted image"), wandb.Image(generated_image, caption="predicted image - noise")]}, step=step)

#             # save the model
#             torch.save([step, w, noises], config.checkpoint)
    
#     print("now, optimizing noises")

#     noises = gen.make_noise()
#     for noise in noises:
#         noise.requires_grad = True
            
#     optimizer = optim.Adam(noises, lr=0.1)
    
#     print("only optimizing mse")
#     pdisc_coef = 0
#     pvgg_coef = 0
#     mse_coef = 1
    
#     latent_std = .5
    
#     # do 500 steps for the noise
#     progress_bar = tqdm(range(step, step + 500))
#     for step in progress_bar:

#         if config.pro_model:
#             t = step / config.nb_it
#             lr = get_lr(t, config.lr)
#             optimizer.param_groups[0]['lr'] = lr

#             noise_strength = latent_std * config.noise * max(0, 1 - t / config.noise_ramp) ** 2
#             latent_n = latent_noise(w, noise_strength) # latent noise
#             n_loss = noise_regularize(noises)

#         img_gen, _ = gen([latent_n], input_is_latent=is_w, noise=noises)

#         if config.video:
#             im_proced = postprocess(img_gen) # this is for video
        
#         p_loss = percept(img_gen, ref_image) # make the perceptual discriminator loss 0 for this
        
#         interpolate_size = 256
#         p_loss_vgg = percept_vgg(F.interpolate(img_gen, size=interpolate_size), F.interpolate(ref_image, size=interpolate_size))
#         mse_loss = F.mse_loss(img_gen, ref_image)
        
        
        
#         loss = pdisc_coef * p_loss + pvgg_coef * p_loss_vgg + mse_coef * mse_loss

#         optimizer.zero_grad()
#         loss.backward()
#         loss = loss.item()
#         optimizer.step()
#         scheduler.step()
#         progress_bar.set_description("Step: {}, Loss: {:.5f}, Pdisc_loss (m): {:.2f} ({:.4f}), Pvgg_loss (m): {:.2f} ({:.5f}), mse_loss (m): {:.5f} ({:.5f})"
#                                      .format(step, loss, p_loss.item(), pdisc_coef * p_loss.item(), p_loss_vgg.item(), pvgg_coef * p_loss_vgg.item(), mse_loss.item(), mse_coef * mse_loss.item()))
        
#         wandb.log({"loss": round(loss, 2), 
#                    "pdisc_loss": round(p_loss.item(), 2), 
#                    "mse_loss": round(mse_loss.item(), 2), 
#                    "pvgg_loss": round(p_loss_vgg.item(), 2), 
#                    "pdisc_mult": round(pdisc_coef * p_loss.item(), 2), 
#                    "vgg_mult": round(pvgg_coef * p_loss_vgg.item(), 2), 
#                    'mse_mult': round(mse_coef * mse_loss.item(), 2), 
#                    'learning_rate': round(optimizer.param_groups[0]['lr'], 4)}, 
#                   step=step)
        
        
#         # show the image every 100 steps
#         if step % 100 == 0:
#             generated_image, _ = gen([w], input_is_latent=is_w)
#             wandb.log({"Image Prediction": [wandb.Image(ref_image, caption="input image"), wandb.Image(img_gen, caption="predicted image"), wandb.Image(generated_image, caption="predicted image - noise")]}, step=step)

#             # save the model
#             torch.save([step, w, noises], config.checkpoint)
            

#     print("saving model in ", config.checkpoint)
#     torch.save([step, w, noises], config.checkpoint)


#     if config.video:
#         print("saving video")
#         images_to_video(generated_image_hook.get_images(), config.video_path, image_size=1024)
#         # actually for video we have to call the post process function !!
        

        
        
        
        
        
        
def main_func(args):
    '''
    I will save in the following way:
        checkpoint_dir / personname_nblatents
            opt_z_init_e_run_{}
                video
                checkpoint
                wandb
                    config
    '''
    
    # get the checkpoint path
    person_name, _ = os.path.splitext(os.path.basename(args.im_path))
    parent_dir = os.path.join(args.checkpoint_dir, person_name + '_{}'.format(args.nlatent)) # checkpointdir / asya_01_1

    joint = False
    
    if args.opt_noise:
        # let's see if we're optimizing jointly
        if args.opt_z:
            sub_dir = "opt_zn_init_" + args.init
            joint = True
        elif args.opt_w:
            sub_dir = "opt_wn_init_" + args.init
            joint = True
        
        else:
            # if we're only optimizing the noise, the user must have specified which run to use before as a string
            if args.opt_noise_from is None:
                raise ValueError("Error. No previous checkpoint given where to optimize noise. option --opt_noise_from is mandatory")
            
            noise_from_folder = os.path.join(parent_dir, args.opt_noise_from)
            prev_ckpt_path = os.path.join(noise_from_folder, "ckpt.pt")
            
            if not os.path.exists(noise_from_folder) or not os.path.exists(prev_ckpt_path):
                print()
                print(noise_from_folder, os.path.exists(noise_from_folder))
                print(prev_ckpt_path, os.path.exists(prev_ckpt_path))
                raise ValueError("Error. Checkpoint or folder {} does not exist".format(noise_from_folder))
                
            sub_dir = "opt_noise_{}".format(args.opt_noise_from)
            
    elif args.opt_z:
        sub_dir = "opt_z_init_" + args.init
        
    elif args.opt_w:
        sub_dir = "opt_w_init_" + args.init
        
    else:
        raise ValueError("No optimization method given")
                                                                                                  

    # current format of sub_dir: opt_z_init_r
    
    print("PERSON:", person_name)
    
    run_dir = os.path.join(parent_dir, sub_dir)
    run_dir = run_dir + "_run_{}".format(get_next_number(run_dir))
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    checkpoint_path = os.path.join(run_dir, 'ckpt.pt')
    video_path = os.path.join(run_dir, 'video.avi')
    wandb_directory = run_dir
    print("wandb dir:", wandb_directory)
    if os.path.exists(os.path.join(wandb_directory, 'wandb')):
        shutil.rmtree(os.path.join(wandb_directory, 'wandb'))
    

    # get our models 
    
        
    # load generator and discriminator
    gen, disc, avg_latent = get_stylegan_models()
    
    # first, get original image
    transform = transforms.Compose(
    [
        transforms.Resize(1024), # which is what stylegan should generate
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    input_image = np.asarray(Image.open(args.im_path))
    
    if input_image.shape[-1] > 3:
        input_image = input_image[:, :, :3]
        
    input_image = Image.fromarray(np.uint8(input_image))
    
    
    
    ref_image = transform(input_image).to(args.device)



    ####################################
    ####################################
    #### our initialization method #####
    ####################################
    ####################################


    
    if args.opt_noise and not joint:
        # we will not optimize the latent; we will just get it from our checkpoint
        _, latent_ini, _ = torch.load(prev_ckpt_path)
    
    else:
        if args.opt_z:
            if args.init == 'e':
                print("error, tried to initialize with encoder but optimizing z! using average vector instead.")
                args.init = 'a'
            if args.init == 'a': # average, initialize with 0
                print("optimizing z; initializing with average vector (zeros)")
                z = torch.zeros((1, args.nlatent, 512), device='cuda')
                temp = z.detach().to(args.device)

            elif args.init == 'r': # random, initialize with random
                print("optimizing z; initializing with random normal vector")
                z = torch.randn((1, args.nlatent, 512), device='cuda')
                temp = z.detach().to(args.device)
            



        elif args.opt_w: # optimizing w

            if args.init == 'a': # average, initialize with average w
                print("optimizing w; initializing with average vector") # avg_latent is (512) dimensional
                w = avg_latent.unsqueeze(0).unsqueeze(0).repeat(1, args.nlatent, 1) # now it is size (1, args.nlatent, 512)
                temp = w.detach().to(args.device)

            elif args.init == 'r': # random, initialize with random z transformed
                print("optimizing w; initializing with random vector")
                sample_z = torch.randn((1, args.nlatent, 512), device='cuda')
                w = gen.style(sample_z)
                temp = w.detach().to(args.device)

            elif args.init == 'e': # encoder, use the encoder that I already have trained
                print("optimizing w; intializing with encoded image")

                resize_encoder = 224
                transform_encoder = transforms.Compose(
                [
                    transforms.Resize(resize_encoder),
                    transforms.ToTensor(),
                    transforms.Normalize(imagenet_stats[0], imagenet_stats[1]),
                ]
                )
                enc_image = transform_encoder(input_image)
                enc_image = enc_image.unsqueeze(0)
                enc_image = enc_image.to(args.device)

                # i should just always use my encoder which produces the 1x18 images, it is much better
                #encoder = get_encoder('latent_image_pairs_{}_1024_{}.pt'.format(nb_ims, args.nlatent), checkpoint_dir=args.encoder_dir, input_size=resize_encoder)
                encoder = get_encoder('latent_image_pairs_500000_1024_1.pt', input_size=resize_encoder)
                encoder.eval()
                w = encoder(enc_image) # this is (1, 512 dimensional)
                w = w.repeat(args.nlatent, 1).unsqueeze(0) # this is to repeat it across 18 dimensions
                temp = w.detach().to(args.device)   
    
        # temps is size (1, args.nlatent, 512); we want it to be just (1, 512) if args.nlatent is 1
        latent_ini = temp.squeeze(1) # if nlatent = 1 then we just remove this dimension, otherwise if its 18 then this line won't do anything
    
    print(" ")
    print(" SHAPE OF LATENT INI : ", latent_ini.shape)
    print()
        
    n_ini = gen.make_noise()
    if args.zero_noise:
        for noise in n_ini:
            noise *= 0
        
    config = None 
    

    wandb.init(config=config, project="w_" + person_name, dir=wandb_directory, reinit=True, name=os.path.basename(wandb_directory))
    
    wandb.config.update(args, allow_val_change=True) # adds all of the arguments as config variables
    config = wandb.config
    config.video_path = video_path
    config.checkpoint = checkpoint_path
    
    #avg_lat = avg_latent.unsqueeze(0).unsqueeze(0).repeat(1, args.nlatent, 1) 
    #avg_lat = avg_lat.to(args.device)
    
    print('save direcrtory for wandb', wandb_directory)
    # delete any wandb directory in this directory; it doesn't have a checkpoint path so we don't care

    pp = pprint.PrettyPrinter(width=41, compact=False)
    print()
    print()
    pp.pprint(vars(args))
    print()
    print()
    
    train(latent_ini, n_ini, ref_image.unsqueeze(0), gen, disc, config)
    
    wandb.join() # end this run

    
    
    
    
    
    
# we will always generate 1024 images in this code because stylegan2 does not work very well to generate smaller iamges
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--im_path', type=str, required=True, help='The path to the image we want to find a latent vector to.')
    parser.add_argument('--stylegan_ckpt', 
                        type=str, 
                        default='/home/datadrive/asya/checkpoints/stylegan2/stylegan2-ffhq-config-f.pt', 
                        help='StyleGAN2 checkpoint (should contain generator and discriminator)')
    parser.add_argument('--checkpoint_dir', 
                        type=str, 
                        default='/home/datadrive/asya/checkpoints/stylegan2/w_projections', 
                        help='The checkpoint directory where wandb and the checkpoints will be saved')
    parser.add_argument('--nlatent', type=int, default=18, help='Can either be 1 or 18 (number of latents of size 512 which encode the latent vector)')
    
    # which parameter to optimize
    parser.add_argument('--opt_z', default=False, action='store_true', help='Add option to optimize z')
    parser.add_argument('--opt_noise', 
                        default=False, 
                        action='store_true', 
                        help='Whether we want to now optimize the noise from a previous path (must be mentioned in opt_noise_from)')
    parser.add_argument('--opt_noise_from', 
                        default=None, 
                        help='The path where we optimize the noise from. This is of the form [opt_z_init_e_run_num] (parent director is inferred)')
    parser.add_argument('--opt_w', default=False, action='store_true', help='Add option to optimize w')
    
    # initialization
    parser.add_argument('--init', default='e', help='Options: e (use encoder), r (use random), a (use average)')
    parser.add_argument('--zero_noise', default=False, action='store_true', help="Initialize all noise with 0; otherwise random initialization")
    
    # for discriminator perceptual loss
    # parser.add_argument('--pattern', default='norm', type=str, help='possible options: [norm, inc, dec, zigup, zigdown]. for perceptual disc loss')
    # parser.add_argument('--percept_fact', default=1., type=float, help='the multiplier factor for the perceptual loss')
    # parser.add_argument('--layers', default=None, help='the layers to use for the perceptual discriminator loss')
    # losses - for now I will only use perceptual loss but later add option to use MSE AND VGG perceptual loss!!!
    
    # training options
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--nb_it', type=int, default=500)


    parser.add_argument('--encoder_dir', default='/home/datadrive/asya/checkpoints/stylegan2/latent_encoder')
    
    parser.add_argument('--mse_coef', type=float, default=1e-5)
    parser.add_argument('--pvgg_coef', type=float, default=1e-5)
    parser.add_argument('--pdisc_coef', type=float, default=0.)
    parser.add_argument('--noise_reg', type=float, default=1e-5)
    
    # add an initialization term so that we do not go far from the initial value
    parser.add_argument('--init_reg', type=float, default=0)
    
    
    
    # output options
    parser.add_argument('--video', default=False, action='store_true')
    
    # mis options
    parser.add_argument('--device', default='cuda')
                                        
    args = parser.parse_args()

    main_func(args)


