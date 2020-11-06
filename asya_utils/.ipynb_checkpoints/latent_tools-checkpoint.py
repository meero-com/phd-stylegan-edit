from .processing.network_postproc import PostSynthesisProcessing, GeneratedImageHook, normalize_tens
from .view_tools import show_grid, vision_args
from .processing.image_tools import images_to_video

from torchvision import transforms, utils
import torch
import torch.nn.functional as F
from fastai.vision.data import imagenet_stats

from PIL import Image
import os

import lpips
    
def get_projection_info(ckpt_path, encoder, g_ema, view_encoding=True, imsize=224, im_dir='/home/data/images/aligned_images', save=''):
    # views threesome and returns score
    lpips_criterion = lpips.LPIPS(net='alex')
    mse_criterion = torch.nn.MSELoss()
    try:
        
        name = ckpt_path.split('w_projections')[1].split('01')[0] + '01.png'
        name = name[1:]
        im_path = os.path.join(im_dir, name)
        print(ckpt_path)
    except Exception:
        print("Error; try another checkpoint path.")
        raise Exception
        
    transform_encoder = transforms.Compose(
    [
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_stats[0], imagenet_stats[1]),
    ]
    )

    # this is so I can view normalize the input image and just do normal operation
    transform_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # this is so I can normalize the final output!!
    ])
    
    raw_image = Image.open(im_path).convert('RGB')
    ref_image = transform_tensor(raw_image).unsqueeze(0)
    
    input_enc = transform_encoder(raw_image).unsqueeze(0).cuda() # transform the image so that I can insert it into my encoder
    encoded = encoder(input_enc)
    
    print('shape:', ref_image.shape)
    
    if view_encoding:
        output_w_enc, _ = g_ema([encoded], input_is_latent='z' not in ckpt_path)
        output_w_enc = normalize_tens(output_w_enc.to('cpu'))
    else:
        output_w_enc = torch.zeros(0, ref_image.shape[1], ref_image.shape[2], ref_image.shape[3]) # this is like a "dummy"
    
    
    
    # find the original image from this path
    try:
        stp, w, noises = torch.load(ckpt_path)
    except Exception:
        print("no noises")
        stp, w = torch.load(ckpt_path)
        noises = None
        
        
    output_w_opt, _ = g_ema([w], input_is_latent=True, noise=noises)
    output_w_opt = normalize_tens(output_w_opt.to('cpu'))
    
    
    # normalize between -1 and 1
    
    
    zero_noises = g_ema.make_noise()
    for noise in zero_noises:
        noise *= 0
    output_w_no_noise, _ = g_ema([w], input_is_latent=True, noise=zero_noises)
    output_w_no_noise = normalize_tens(output_w_no_noise.to('cpu'))
    
    lpips_score = lpips_criterion(ref_image, output_w_opt)
    mse_score = mse_criterion(ref_image, output_w_opt)
    scores={'lpips': lpips_score.detach().item(), 'mse_score': mse_score.detach().item()}
    
    
    if noises is not None:
        
        zero_noises = g_ema.make_noise()
        for noise in zero_noises:
            noise *= 0
        
        print("\t\tOriginal\t\t\tEncoded\t\t\tOptimized\t\t\tNo Noise")
        show_grid(utils.make_grid(torch.cat((ref_image, output_w_enc, output_w_opt, output_w_no_noise)), **vision_args))
        if save:
            utils.save_image(torch.cat((ref_image, output_w_enc, output_w_opt, output_w_no_noise)), '/home/data/results/expes/3some.png', **vision_args)
    
    
    else:
        print("\t\t    Original\t\t\t\t     Encoded\t\t\t\t     Optimized")
        show_grid(utils.make_grid(torch.cat((ref_image, output_w_enc, output_w_opt)), **vision_args))
        if save != '':
            utils.save_image(torch.cat((ref_image, output_w_enc, output_w_opt)), '/home/data/results/expes/{}.png'.format(save), **vision_args)
    return w, noises, scores




def view_latents(person, w_list, noise_list, g_ema, imsize=224):
    '''
    This is so I can view different latents for a single person
    For example if I have different methods like StyleGAN inversion,
    Image2StyleGAN inversion, my inversion, I can put all the outpts
    here and then visualize them side by side.
    The input image will be first. 
    '''
    
    im_dir='/home/data/images/aligned_images'
    
    if '02' in person:
        try:
            path_to_image = os.path.join(im_dir, person) + '.png'
        except Exception:
            print("image not found in im_dir")
            raise Exception
    else:
        try:
            path_to_image = os.path.join(im_dir, person) + '_01.png'
        except Exception:
            print("Image not found in im_dir.")
            raise Exception

    # this is so I can view normalize the input image and just do normal operation
    transform_tensor = transforms.Compose(
    [
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # this is so I can normalize the final output!!
    ])
    
        
    raw_image = Image.open(path_to_image).convert('RGB') # PIL Image
    ref_image = normalize_tens(transform_tensor(raw_image).unsqueeze(0)).to('cpu') 
    
    outs = [ref_image]
    for w, noises in zip(w_list, noise_list):
        output_w, _ = g_ema([w], input_is_latent=True, noise=noises)
        output_w = normalize_tens(output_w.to('cpu'))
        output_w = F.interpolate(output_w, size=imsize)
        outs.append(output_w)
    
    show_grid(utils.make_grid(torch.cat(outs)), **vision_args)
        
    
def interpolation_imgs(g_ema, latent1, latent2, noises1=None, noises2=None, nb_interpolations=100, resize=None):
    '''
    g_ema: our generator
    latent1: first w
    latent2: second 2
    noises1: first noise; can be None
    noises2: second noise; can be None
    nb_interpolation_steps: how many interpolation steps in our video
    '''
    
    all_imgs = []
    
    for i in range(nb_interpolations):
        w_int = torch.lerp(latent1, latent2, (i) / (nb_interpolations - 1))
    
        if (noises1 is None or noises2 is None):
            noises_int = None
        else:
            noises_int = []
            for noise_index in range(len(noises1)):
                noise_1 = noises1[noise_index]
                noise_2 = noises2[noise_index]
                noises_int.append(torch.lerp(noise_1, noise_2, (i) / (nb_interpolations - 1)))

        output_im, _ = g_ema([w_int], input_is_latent=True, noise=noises_int)
        
        if resize is not None and resize != output_im.shape[-1]:
            output_im = F.interpolate(output_im, size=resize)
            
        all_imgs.append(output_im.detach().to('cpu'))
        
    
    return all_imgs
    
    
def interpolation_imgs_trunc(g_ema, latent1, avg_latent, noises1=None, truncs=None, nb_interpolations=100, resize=None):
    # noises1 can be None
    # user can pass the desired truncations, otherwise we will use nb_interpolation steps
    
    all_imgs = []
    
    if truncs is None:
        truncs = np.linspace(1, -1, nb_interpolations)
    
    
    for truncation in truncs:
        output_im, _ = g_ema([latent1], 
                             input_is_latent=True, 
                             noise=noises1, 
                             truncation=truncation, 
                             truncation_latent=avg_latent)
        
        if resize is not None and resize != output_im.shape[-1]:
            output_im = F.interpolate(output_im, size=resize)
        
        all_imgs.append(output_im.detach().to('cpu'))
    
    return all_imgs
    
    
def interpolation_video(g_ema, 
                        latent1, 
                        latent2, 
                        filename, 
                        noises1=None, 
                        noises2=None, 
                        nb_interpolations=100, 
                        nb_pause=15, 
                        image_size=1024,
                        video_dir='/home/data/results/videos'):
    '''
    g_ema: our generator
    latent1: our first w
    latent2: our second 2
    filename: name of our video WITHOUT .avi extension
    nb_interpolation_steps: how many interpolation steps in our video
    nb_pause: pause for first and last frames
    image_size: must be the same size as our generator outputs
    '''
    
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    frames = []
    
    filename = filename.split('.')[0]
    video_path = os.path.join(video_dir, filename) + '.avi'
    
    postprocess = PostSynthesisProcessing()
    #generated_image_hook = GeneratedImageHook(postprocess, every_n=1)
    
    images = interpolation_imgs(g_ema, 
                                latent1=latent1, 
                                latent2=latent2, 
                                noises1=noises1, 
                                noises2=noises2, 
                                nb_interpolations=nb_interpolations, 
                                resize=image_size)
    
    for im in images:
        frames.append(postprocess(im).numpy()[0])
    
    images_to_video(frames, video_path, image_size=image_size, nb_pause=nb_pause)
    
    
def interpolation_video_trunc(g_ema, 
                              latent1, 
                              avg_latent, 
                              filename, 
                              noises1=None, 
                              truncs=None, 
                              nb_interpolations=100, 
                              nb_pause=15, 
                              image_size=1024, 
                              video_dir='/home/data/results/videos'):
    '''
    g_ema: our generator
    latent1: our first w
    latent2: our second 2
    filename: name of our video WITHOUT .avi extension
    nb_interpolation_steps: how many interpolation steps in our video
    nb_pause: pause for first and last frames
    image_size: must be the same size as our generator outputs
    '''
    frames = []
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    filename = filename.split('.')[0]
    video_path = os.path.join(video_dir, filename) + '.avi'
    
    postprocess = PostSynthesisProcessing()
    
    images = interpolation_imgs_trunc(g_ema, 
                                      latent1, 
                                      noises1=noises1, 
                                      avg_latent=avg_latent, 
                                      truncs=truncs, 
                                      nb_interpolations=nb_interpolations, 
                                      resize=image_size)
    
    for im in images:
        frames.append(postprocess(im).numpy()[0])
    
    images_to_video(frames, video_path, image_size=image_size, nb_pause=nb_pause)
    