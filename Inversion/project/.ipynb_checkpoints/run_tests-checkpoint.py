from train_latent import main_func
from types import SimpleNamespace

args = SimpleNamespace()



# this is fixed
args.stylegan_ckpt = '/mnt/datadrive/asya/checkpoints/stylegan2/stylegan2-ffhq-config-f.pt'
args.checkpoint_dir = '/mnt/datadrive/asya/checkpoints/stylegan2/w_projections'
args.encoder_dir = '/mnt/datadrive/asya/checkpoints/stylegan2/latent_encoder'
args.video = True 
args.device = 'cuda'


# for im_path in ['/home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png', '/home/asya/work_git/recherche/StyleGAN/aligned_images/jean_01.png', '/home/asya/work_git/recherche/StyleGAN/aligned_images/loic_01.png']:
# 	for nlatent in [18, 1]:
# 		for opt_w_initialization in [(True, 'a'), (True, 'e'), (True, 'r'), (False, 'a'), (False, 'r')]:
# 			opt_w, init = opt_w_initialization
# 			for mse_coef in [0, 1, 5, 10]:
# 				for vgg_coef in [0, 1, 5, 10]:
# 					for pdiscmode in [('norm', 0), ('norm', 1), ('norm', 2), ('dec', 1), ('dec', 2)]:
# 						pattern, pdisc_coef = pdiscmode

# 						if mse_coef == vgg_coef == pdisc_coef == 0:
# 							continue

# 						# this will vary
# 						args.im_path = im_path # vary between me, loic, jean
# 						args.nlatent = nlatent # vary between 1 and 18
# 						args.opt_z = not opt_w # vary between true and false
# 						args.pattern = pattern # vary between 'norm', 'inc', 'dec', 'zigup', 'zigdown'
# 						args.init = init

# 						args.mse_coef = mse_coef # vary between [0, 1, 5, 10]
# 						args.pvgg_coef = vgg_coef # vary between [0, 1, 5, 10]
# 						args.pdisc_coef = pdisc_coef # vary between [0, 1, 2]

# 						print("\n\n")
# 						pp.pprint(vars(args))
# 						print("\n\n")

# 						main_func(args)


'''
# first of all: which converges faster ? just optimizing p-disc, just p-vgg, just mse ? what are the metrics for all 3 of them when they converge?
im_path = '/home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png'
pattern = 'norm'
for nlatent in [18, 1]:
	for opt_w_initialization in [(True, 'a'), (True, 'e'), (True, 'r')]:
		opt_w, init = opt_w_initialization
		if init == 'a':
			continue
		for mse_coef, vgg_coef, pdisc_coef in zip([0, 0, 1.], [0, 1., 0], [1., 0, 0]):
			args.im_path = im_path
			args.nlatent = nlatent 
			args.opt_z = not opt_w 
			args.pattern = pattern
			args.init = init

			args.mse_coef = mse_coef
			args.pvgg_coef = vgg_coef
			args.pdisc_coef = pdisc_coef


			main_func(args)
'''




# i=0
# for mse_coef, vgg_coef, pdisc_coef in zip([0, 0, 1.], [0, 1., 0], [1., 0, 0]):
    
#     print('mse_coef, vgg_coef, pdisc_coef', mse_coef, vgg_coef, pdisc_coef)
#     if i==0:
#         i=1
#         continue
    
    
#     args.mse_coef = mse_coef
#     args.pvgg_coef = vgg_coef
#     args.pdisc_coef = pdisc_coef
    

#     args.train_alternate = True

#     main_func(args)

#     args.train_alternate = False
#     main_func(args)



        
# same parameters as run 65
        

    


# first: let's see the differences in different initializations for w

args.lr = 0.008
args.nlatent = 18
args.nb_it = 1500
args.opt_noise = True
args.opt_z = False
args.opt_noise_from = None
args.zero_noise = False
args.init = 'a'

im_path = '/home/asya/work_git/recherche/StyleGAN/aligned_images/asya_01.png'

args.im_path = im_path
args.opt_w = True


# runs 1-4

# args.mse_coef = 1e-5
# args.pdisc_coef = 0
# args.pvgg_coef = 1e-5
# main_func(args)

# args.mse_coef = 1e-5
# args.pdisc_coef = 1e-5
# args.pvgg_coef = 0
# main_func(args)

# args.pdisc_coef = 0
# args.pvgg_coef = 0
# args.mse_coef = 1e-5
# main_func(args)

# args.mse_coef = 1e-5
# args.pdisc_coef = 1e-5
# args.pvgg_coef = 1e-5
# main_func(args)

# args.mse_coef = 1e-5
# args.pdisc_coef = 0
# args.pvgg_coef = 1e-5
# main_func(args)

# args.mse_coef = 1e-5
# args.pdisc_coef = 1e-5
# args.pvgg_coef = 1e-5
# main_func(args)


# args.init = 'r'
# args.mse_coef = 1e-5
# args.pdisc_coef = 0
# args.pvgg_coef = 1e-5
# main_func(args)

# args.mse_coef = 1e-5
# args.pdisc_coef = 1e-5
# args.pvgg_coef = 1e-5
# main_func(args)




# args.mse_coef = 1e-5
# args.pdisc_coef = 1e-5
# args.pvgg_coef = 0#1e-5
# main_func(args)


# args.opt_noise = False
# args.mse_coef = 1e-5
# args.pdisc_coef = 1e-5
# args.pvgg_coef = 0#1e-5
# main_func(args)


args.opt_noise = True
args.opt_w = False
args.opt_z = False
args.opt_noise_from = 'opt_w_init_a_run_1'

args.mse_coef = 1e-4
args.pdisc_coef = 0
args.pvgg_coef = 0

main_func(args)