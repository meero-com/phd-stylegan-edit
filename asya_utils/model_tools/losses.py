import os
import torch.nn.functional as F
from torchvision.models import vgg16
import torch

import sys
path_to_stylegan = '/home/stylegan2'
sys.path.append(path_to_stylegan)
from model import Generator, Discriminator

# our models for image to latent learning
# https://github.com/jacobhallberg/pytorch_stylegan_encoder
from Inversion.pytorch_stylegan_encoder.models.image_to_latent import ImageToLatent

sys.path.append('/home')
from asya_utils.model_tools.model_loaders import get_stylegan_models




# losses

'''
this loss is used for my encoder for latent2image
it measures the difference between the input latent and the target latent
it pen
'''
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true, pred):
        loss = true - pred
        return torch.mean(torch.log(torch.cosh(loss + 1e-12)))
    
class L1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, true, pred):
        return torch.mean(torch.abs(true - pred))
        
class LatentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = L1Loss()
        self.log_cosh_loss = LogCoshLoss()
        self.l2_loss = torch.nn.MSELoss()
    
    def forward(self, real_features, generated_features, average_dlatents = None, dlatents = None):
        # Take a look at:
            # https://github.com/pbaylies/stylegan-encoder/blob/master/encoder/perceptual_model.py
            # For additional losses and practical scaling factors.
            
        loss = 0
        # Possible TODO: Add more feature based loss functions to create better optimized latents.            
            
        # Modify scaling factors or disable losses to get best result (Image dependent).
        
        # VGG16 Feature Loss
        # Absolute vs MSE Loss
        # loss += 1 * self.l1_loss(real_features, generated_features)      
        loss += 1 * self.l2_loss(real_features, generated_features)

        # Pixel Loss
#         loss += 1.5 * self.log_cosh_loss(real_image, generated_image)

        # Dlatent Loss - Forces latents to stay near the space the model uses for faces.
        if average_dlatents is not None and dlatents is not None:
            loss += 1 * 512 * self.l1_loss(average_dlatents, dlatents)

        return loss

    
    
# DEPCRECATED
class PerceptualDiscLoss(torch.nn.Module):
    # we initialize the perceptual loss with the trained discriminator. Here we will add forward hooks to the discriminator
    # in order to retreive the activations from the intermediate layers
    
    def __init__(self, discriminator, pattern, fact=1., layers=None): # layers where 0 is first layer
        super().__init__()
        raise ValueError("DEPRECATED")
        
        assert pattern in ['norm', 'inc', 'dec', 'zigup', 'zigdown']
        
        if layers is not None:
            self.layers = sorted(list(set(layers)))
            if min(self.layers) < 0 or max(self.layers) > 9:
                raise ValueError("ERROR, layers input is false")
        else:
            self.layers = None
            
        self.m_fact = fact
        self.pattern = pattern
        self.discriminator = discriminator
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output#.detach()
            return hook
        for i in range(len(discriminator.convs)):
            self.discriminator.convs[i].register_forward_hook(get_activation("conv_{}".format(i)))
    
    def forward(self, pred_image, real_image):
        # pass them both through the discriminator
        out_pred = self.discriminator(pred_image)
        activations_pred = self.activation.copy()
        
        out_real = self.discriminator(real_image)
        activations_real = self.activation.copy()
        
        # now we will compare all of them
        perceptual_loss = 0
        coef = 1
        num_act = len(self.activation.keys())
        
        
        
        if self.pattern == 'norm':
            self.fact = 1.
            ordered_keys = self.activation.keys()
        elif self.pattern in ['inc', 'zigup']:
            ordered_keys = sorted(self.activation.keys())
        else:
            ordered_keys = reversed(sorted(self.activation.keys()))
        
        
        
        if not self.layers:
            self.layers = range(len(ordered_keys)) # we get all of them
        
        
        
        index_chosen_layer = 0
            
        for layer_num, conv in enumerate(ordered_keys):
            
            if index_chosen_layer == len(self.layers):
                break
                
            if layer_num < self.layers[index_chosen_layer]: # this is not one of the layers we chose; continue
                continue
            
            
            perceptual_loss += coef * (((activations_pred[conv] - activations_real[conv])**2).mean())
            if self.pattern in ['zigup', 'zigdown'] and layer_num > num_act // 2:
                coef /= self.m_fact
            else: # so if it's not zig it will always go here; if it is zig then it will only go here in the beginning. if it's normal then the coef will never change since m_fact is 1
                coef *= self.m_fact
            
            index_chosen_layer += 1
            
        return perceptual_loss
    
###
###   
### PERCEPETUAL LOSS !!!!
###
###
###


class DiscWrapper(torch.nn.Module):
    # 'conv_0', 'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8']
    def __init__(self): # 
        super().__init__()
        
        self.activation = {}
        _, self.discriminator, _ = get_stylegan_models('cuda')
        
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.to('cpu')
            return hook
        for i in range(len(self.discriminator.convs)):
            self.discriminator.convs[i].register_forward_hook(get_activation("conv_{}".format(i)))
    
    def forward(self, im):
        out = self.discriminator(im.to('cuda')).to('cpu')
        return self.activation.copy()
    
    
class VGG16Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        features = vgg16(pretrained=True).features
        self.to_conv_1_1 = torch.nn.Sequential() 
        self.to_conv_1_2 = torch.nn.Sequential() 
        self.to_conv_2_1 = torch.nn.Sequential() 
        self.to_conv_2_2 = torch.nn.Sequential() 
        self.to_conv_3_1 = torch.nn.Sequential() 
        self.to_conv_3_2 = torch.nn.Sequential() 
        self.to_conv_3_3 = torch.nn.Sequential()
        self.to_conv_4_1 = torch.nn.Sequential()
        self.to_conv_4_2 = torch.nn.Sequential()
        self.to_conv_4_3 = torch.nn.Sequential()
        self.to_conv_5_1 = torch.nn.Sequential()
        self.to_conv_5_2 = torch.nn.Sequential()
        self.to_conv_5_3 = torch.nn.Sequential()
        
        # the layer indexes (inclusive) corresponding to the names; indexes start at 0
        # this is from the pytorch architecture with pretrained vgg16 network
        # we can visualize these layer numbers easily with :
        # import torchvision.models as models
        # vgg16 = models.vgg16(pretrained=True)
        # vgg16.features
        
        conv11 = 1
        conv12 = 3
        conv21 = 6
        conv22 = 8
        conv31 = 11
        conv32 = 13
        conv33 = 15
        conv41 = 18
        conv42 = 20
        conv43 = 22
        conv51 = 25
        conv52 = 27
        conv53 = 29
        

        for x in range(conv11 + 1):
            self.to_conv_1_1.add_module(str(x), features[x])            
        for x in range(conv11 + 1, conv12 + 1):
            self.to_conv_1_2.add_module(str(x), features[x])            
        for x in range(conv12 + 1, conv21 + 1):
            self.to_conv_2_1.add_module(str(x), features[x])        
        for x in range(conv21 + 1, conv22 + 1):
            self.to_conv_2_2.add_module(str(x), features[x])        
        for x in range(conv22 + 1, conv31 + 1):
            self.to_conv_3_1.add_module(str(x), features[x])        
        for x in range(conv31 + 1, conv32 + 1):
            self.to_conv_3_2.add_module(str(x), features[x])        
        for x in range(conv32 + 1, conv33 + 1):
            self.to_conv_3_3.add_module(str(x), features[x])        
        for x in range(conv33 + 1, conv41 + 1):
            self.to_conv_4_1.add_module(str(x), features[x])        
        for x in range(conv41 + 1, conv42 + 1):
            self.to_conv_4_2.add_module(str(x), features[x])            
        for x in range(conv42 + 1, conv43 + 1):
            self.to_conv_4_3.add_module(str(x), features[x])        
        for x in range(conv43 + 1, conv51 + 1):
            self.to_conv_5_1.add_module(str(x), features[x])        
        for x in range(conv51 + 1, conv52 + 1):
            self.to_conv_5_2.add_module(str(x), features[x])        
        for x in range(conv52 + 1, conv53 + 1):
            self.to_conv_5_3.add_module(str(x), features[x])
            
        # don't need the gradients, we just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_conv_1_1(x)
        conv11 = h     
        h = self.to_conv_1_2(h)
        conv12 = h   
        h = self.to_conv_2_1(h)
        conv21 = h 
        h = self.to_conv_2_2(h)
        conv22 = h 
        h = self.to_conv_3_1(h)
        conv31 = h       
        h = self.to_conv_3_2(h)
        conv32 = h       
        h = self.to_conv_3_3(h)
        conv33 = h        
        h = self.to_conv_4_1(h)
        conv41 = h        
        h = self.to_conv_4_2(h)
        conv42 = h        
        h = self.to_conv_4_3(h)
        conv43 = h       
        h = self.to_conv_5_1(h)
        conv51 = h        
        h = self.to_conv_5_2(h)
        conv52 = h       
        h = self.to_conv_5_3(h)
        conv53 = h
        
        out = {'conv11': conv11, 'conv12': conv12, 'conv21': conv21, 'conv22': conv22, 
               'conv31': conv31, 'conv32': conv32, 'conv33': conv33, 'conv41': conv41, 
               'conv42': conv42, 'conv43': conv43, 'conv51': conv51, 'conv52': conv52, 'conv53': conv53}
        #for k in out:
        #    out[k] = out[k].to('cpu')
        return out

    

# this has to take 224 x 224 input
class PerceptualLoss(torch.nn.Module):
    def __init__(self, net):
        super(PerceptualLoss, self).__init__()
        
        if net == 'disc':
            self.model = DiscWrapper()
            self.imsize = 1024
            self.mean = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1)) 
            self.std = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1))
        elif net == 'vgg':
            self.model = VGG16Wrapper()
            self.imsize = 224
            self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) 
            self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        else:
            raise ValueError("no model fitting net type")
        
        self.transform = F.interpolate
        
        
    def forward(self, im1, im2):
        im1 = im1.to('cpu')
        im2 = im2.to('cpu')
        
        if not (im1.shape[2] == im1.shape[3] == im2.shape[2] == im2.shape[3] == self.imsize):
            print("Warning, images are not 224x224. Transforming images to match.")
            im1 = self.transform(im1, mode='bilinear', size=(self.imsize, self.imsize), align_corners=False)
            im2 = self.transform(im2, mode='bilinear', size=(self.imsize, self.imsize), align_corners=False)
        
        im1 = (im1 - self.mean) / self.std
        im2 = (im2 - self.mean) / self.std
        
        res = {}
        
        features1 = self.model(im1)
        features2 = self.model(im2)
        
        for key in features1.keys(): # this is conv11, conv12, conv21...
            res[key] = ((features1[key] - features2[key]) ** 2).mean()
        
        return res
    
    def forward_for_keys(self, im1, im2, keys='all'):
        all_layers = self.forward(im1, im2)
        res = 0
        if keys == 'all':
            keys = all_layers.keys()
        for key in keys:
            if key not in all_layers.keys():
                raise ValueError("keys must be a valid key (conv name). Call forward to see the key names or use default for all keys")
            res += all_layers[key]
        return res
        
    
    
    

####
####
# image preprocessing for VGG
####
####

# class PostProcess(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.min = -1
#         self.max = 1

#     def norm_ip(self, img, min, max):
#         img.clamp_(min=min, max=max)
#         img.add_(-min).div_(max - min + 1e-5)
#         return img

#     def forward(self, img):
#         return self.norm_ip(img, self.min, self.max)


# not entirely convinced by this function ...



# class VGGProcessing(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.image_size = 256
#         self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(-1, 1, 1)
#         self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(-1, 1, 1)

#     def forward(self, image):
#         image = image / torch.tensor(255).float()
#         image = F.adaptive_avg_pool2d(image, self.image_size)

#         image = (image - self.mean) / self.std

#         return image


# class LatentOptimizer(torch.nn.Module):
#     def __init__(self, synthesizer, layer=12):
#         super().__init__()

#         self.synthesizer = get_stylegan_model(size=256, strict=False).eval()
#         self.post_synthesis_processing = PostSynthesisProcessing()
#         self.vgg_processing = VGGProcessing()
#         self.vgg16 = vgg16(pretrained=True).features[:layer].cuda().eval()


#     def forward(self, dlatents):
#         generated_image, _ = self.synthesizer([dlatents], input_is_latent=True)
#         generated_image = self.post_synthesis_processing(generated_image)
#         generated_image = self.vgg_processing(generated_image)
#         features = self.vgg16(generated_image)

#         return features








# def get_resnet_model(checkpoint='/home/asya/work_git/recherche/StyleGAN/stylegan2-pytorch/checkpoint/image_to_latent.pt'):
#     image_to_latent = ImageToLatent(256).cuda()
#     step, min_val_loss, model_state_dict = torch.load(checkpoint)
#     image_to_latent.load_state_dict(model_state_dict)
#     return image_to_latent



    
        
