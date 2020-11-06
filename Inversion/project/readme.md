# Inversion

## Step 1. Train a resnet to find a first estimation for a latent code

This repository is to find a latent code for a real image using a pretrained generator. In this case, the pretrained generator is StyleGAN2 on FFHQ. 

The first thing that we can do is train a resnet model to estimate the initial inversion. This will be done with our code `train_resnet.py`. We can execute the command `python train_resnet.py --basename latent_image_pairs_500000_1024_1` to train with our image/latent pairs in this folder. The default parameters should be ok. Once this is done, the result model will be saved in datadrive/asya/checkpoints/stylegan2/latent_encoder. We can either train to predict with 18 or with 1, with 1 it works much better, this is because stylegan2 uses 1 latent code to represent an image (so the training data is kind of skewed this way). 

## Step 2. Optimize even more
The second thing to do is to optimize the latent vector w even more after this initial estimation. We tried many tests, the main code is in `train_latent.py`. 

Basically, I wanted to make this function take --opt_w, --opt_z or --opt_noise (in which case --opt_noise_from is obligatory). In the latter case, basically I wanted to do like in image2stylegan2 where they optimized the latent first and then the noise. However, this doesn't really work. So just call this code with --opt_w and --opt_noise together. 

