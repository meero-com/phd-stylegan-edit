# based on this code: https://github.com/jacobhallberg/pytorch_stylegan_encoder/blob/master/encode_image.py

from image_tools import load_images, GeneratedImageHook, images_to_video
from model_tools import LatentOptimizer, LatentLoss, get_stylegan_model, get_resnet_model, VGGProcessing, PostSynthesisProcessing, PostProcess
from tqdm import tqdm 
import wandb
import os
import torch
import lpips
from torch.nn import functional as F
import glob
from torchvision import transforms
from PIL import Image
from torch import optim
import math
from torch.optim.lr_scheduler import StepLR

import sys

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
	except (IndexError, ValueError) as e:
		return 0

def get_next_number(base_checkpoint_path):
	max_number = 0
	current_runs = glob.glob(base_checkpoint_path + '*')
	numbers = sorted([get_number(run) for run in current_runs])
	if len(numbers) > 0:
		max_number = numbers[-1]
	return max_number + 1
	

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def main(path_to_image):

	# these are the parameters to modify
	config = dict(
		# the checkpoint name will be implicitly known from the path to image and the resume_training_run
		path_to_image = path_to_image, # '/home/asya/work_git/recherche/StyleGAN/aligned_images/valentin_01.png',
		resume_training = False,
		resume_training_run = 0, # not obligatory
		vgg_layer = 12,
		use_resnet = True,
		learning_rate = 0.01,
		nb_iterations = 50000,
		epochs = 20,
		pro_model = True,
		noise_regularize = 1e5,
		mse = 0.2,
		noise = 0.05,
		noise_ramp = 0.75,
		video = True, 
		video_path = '/home/asya/work_git/recherche/StyleGAN/w_projections',
		use_noise = True

	) 

	# have an argument to use adam instead of sgd...
	checkpoint_dir = '/home/asya/work_git/recherche/StyleGAN/stylegan2-pytorch/checkpoint/w_projections'
	person_name = os.path.splitext(os.path.basename(config['path_to_image']))[0]
	base_checkpoint_path = os.path.join(checkpoint_dir, person_name)

	print("PERSON:", person_name)

	if config['resume_training']:
		full_checkpoint_path = base_checkpoint_path + '_run_{}'.format(config['resume_training_run'])
		if not os.path.exists(full_checkpoint_path):
			print("error, checkpoint does not exist")
			raise ValueError
	else:
		full_checkpoint_path = base_checkpoint_path + '_run_{}'.format(get_next_number(base_checkpoint_path))

	# do not modify
	config['checkpoint'] = full_checkpoint_path

	config['video_path'] = os.path.join(config['video_path'], person_name + '.avi')

	wandb.init(config=config, project="w_" + person_name, dir='/mnt/datadrive/asya/wandb')
	config = wandb.config



	#ARGS:
	resize = 256 # this is hardcoded because for vgg features it has to be 256x256 images

	# i need this transform for the pretrained rsnet
	transform = transforms.Compose(
	    [
	        transforms.Resize(resize),
	        transforms.CenterCrop(resize),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	    ]
	)



	# get styleganmodel
	styleganmodel = get_stylegan_model(size=256, strict=False)
	stylegan_full = get_stylegan_model()
	#preprocess = VGGProcessing()
	postprocess = PostSynthesisProcessing()


	if config.video:
		generated_image_hook = GeneratedImageHook(postprocess, every_n=10)


	ref_image = Image.open(config.path_to_image).convert('RGB') #load_images([config.path_to_image]) # this is a numpy array
	ref_image_full = Image.open(config.path_to_image).convert('RGB')
	ref_image = transform(ref_image).cuda()#torch.from_numpy(ref_image).cuda() # this is in gpu now
	#ref_image = preprocess(ref_image) # this does the needed normalization
	#ref_features = latent_opt.vgg16(ref_image).detach()
	ref_image = ref_image.unsqueeze(0)
	ref_image = ref_image.detach()

	print("shape of ref_image: ", ref_image.shape)

	if config.use_resnet:
		# get the resnet model for an initial estimate of w
		resnet = get_resnet_model()
		resnet.eval()

		w = resnet(ref_image).detach().cuda()#.requires_grad_(True)
		w.requires_grad = True

	else:
		raise NotImplementedError 


	cur_step = 0
	if config.resume_training:
		if config.checkpoint is not None:
			print("resuming training")
			cur_step, w = torch.load(config.checkpoint)

	# show the initial w 
	generated_image, _ = styleganmodel([w], input_is_latent=True)
	# generated_image_full, _ = stylegan_full([w], input_is_latent=True)
	wandb.log({"Image Prediction": [wandb.Image(ref_image, caption="input image"), wandb.Image(generated_image, caption="predicted image")]}, step=cur_step)
	# wandb.log({"Full Image Prediction": [wandb.Image(ref_image_full, caption="input image"), wandb.Image(generated_image_full, caption="predicted image")]}, step=cur_step)


	# I think I should optimize the noises in any case

	if config.use_noise:
		noises = styleganmodel.make_noise()
		for noise in noises:
			noise.requires_grad = True
	else:
		noises = []


	percept = lpips.PerceptualLoss(
			model='net-lin', net='vgg', use_gpu=True
			)

	if config.pro_model:
		optimizer = optim.Adam([w] + noises, lr=config.learning_rate) # from projector.py
		
	else:
		optimizer = torch.optim.SGD([w] + noises, lr=config.learning_rate) 

	scheduler = StepLR(optimizer, step_size=500, gamma=0.05)


	latent_std = .5

	progress_bar = tqdm(range(cur_step, config.nb_iterations))
	for step in progress_bar:

		if config.pro_model:
			t = step / config.nb_iterations
			lr = get_lr(t, config.learning_rate)
			optimizer.param_groups[0]['lr'] = lr

		if config.use_noise:
			noise_strength = latent_std * config.noise * max(0, 1 - t / config.noise_ramp) ** 2
			latent_n = latent_noise(w, noise_strength) # latent noise
			n_loss = noise_regularize(noises)
		else:
			noises = None
			latent_n = w
			n_loss = 0

		img_gen, _ = styleganmodel([latent_n], input_is_latent=True, noise=noises)
		im_proced = postprocess(img_gen)
		p_loss = percept(img_gen, ref_image).sum()

		
		mse_loss = F.mse_loss(img_gen, ref_image)
		loss = p_loss + config.noise_regularize * n_loss + config.mse * mse_loss

		optimizer.zero_grad()
		loss.backward()
		loss = loss.item()
		optimizer.step()
		scheduler.step()
		progress_bar.set_description("Step: {}, Loss: {}".format(step, loss))
		wandb.log({"loss": loss, 'learning_rate': optimizer.param_groups[0]['lr']}, step=step)
		# show the image every 100 steps
		if step % 100 == 0:
			generated_image, _ = styleganmodel([w], input_is_latent=True)
			#generated_image_full, _ = stylegan_full([w], input_is_latent=True) doesn't really do anything
			wandb.log({"Image Prediction": [wandb.Image(ref_image, caption="input image"), wandb.Image(generated_image, caption="predicted image")]}, step=step)

			# wandb.log({"Full Image Prediction": [wandb.Image(ref_image_full, caption="input image"), wandb.Image(generated_image_full, caption="predicted image")]}, step=step)
			# save the model
			torch.save([step, w], config.checkpoint)

	print("saving model in ", config.checkpoint)
	torch.save([step, w], config.checkpoint)


	if config.video:
		print("saving video")

		images_to_video(generated_image_hook.get_images(), config.video_path)

if __name__ == '__main__':
	main(sys.argv[1])


	# print('hi')
	# aligned = '/home/asya/work_git/recherche/StyleGAN/aligned_images'
	# all_images = glob.glob(os.path.join(aligned, '*.png'))
	# for im_path in all_images:
	# 	main(im_path)