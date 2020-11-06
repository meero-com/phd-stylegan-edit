import torch
import matplotlib.pyplot as plt
import numpy as np




vision_args = {'normalize': True, 'range': (-1, 1)}


# takes a make_grid image
def show_grid(img):
    """show_grid takes an image done rendered by 'make_grid'
    
    Keyword arguments:
    img: image rendered by 'torchvision.utils.make_grid'
    """
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()




def imshow(tensor):
    """plots the tensor as an image with matplotlib. THIS IS A 3D tensor !!!
    
    
    Keyword arguments:
    tensor: a pytorch tensor, untouched
    """
    try:
        npimg = tensor.detach().cpu().numpy()
    except AttributeError:
        npimg = tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

