import torch
import torch.nn.functional as F
from torchvision import transforms

'''
see celeba_tests.ipynb for use cases of these functions
'''

def normalize_tens(tensor, normrange=(-1, 1)):
    ecart = normrange[1] - normrange[0]
    return ((tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))) * ecart - normrange[1]
    
def denorm(tensor, norm):
    """denormalizes the tensor make to its original distribution
    
    Keyword arguments:
    tensor: the tensor we wish to denormalize (usually the output of a dataloader)
    norm: a 2d array where the first element is mean, the second element is std; which was used to transform the image
    """
    
    # since we have img = (img - mean) / std (with img in [0,1])
    # to denormalize, we multiply by std, then add the mean to go back to [0,1], then multiply by 255
    # to go back to [0, 255]
    
    n_std = torch.tensor(norm[1])
    n_mean = torch.tensor(norm[0])

    n_std = n_std.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    n_mean = n_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    # de-normalize
    return torch.tensor((n_std * tensor + n_mean) * 255, dtype=torch.uint8)


def transform_for_discriminator(size=1024):
    discriminator_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
    return discriminator_transform

class PostSynthesisProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.min_value = -1
        self.max_value = 1

    def forward(self, synthesized_image):

        #self.min_value = float(synthesized_image.min())
        #self.max_value = float(synthesized_image.max())

        synthesized_image = torch.clamp(synthesized_image, self.min_value, self.max_value) # between -1 and 1
        synthesized_image = (synthesized_image - self.min_value) * torch.tensor(255).float() / (self.max_value - self.min_value + 1e-5) # between 0 and 255

        return synthesized_image
    
    
def make_image(tensor):
    """turns tensor into numpy cpu array in RGB format; automatic denormalization
    
    Keyword arguments:
    tensor: a pytorch tensor, untouched
    """
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
    )
 

class GeneratedImageHook:
    # Pytorch forward pass module hook.
    # this can be used for video which saves a bunch of images in a hook while 
    # calling the generator

    '''
    change_factor is to allow to save many images in the beginning, but few at the end
    this is because during beginning of optimization process, there are a lot of changes
    but then it becomes much more stable
    
    set to 0 to ignore this
    
    will change every_n by change_factor every every_n * change_factor times
    for example: if every_n = 1 and change_factor = 10 then:
        for the first 10 x 5 = 50, save every image (50 frames)
        for the next 500, save every 10 images (50 frames)
        for all the rest, save every 50 images ()
    etc etc
    '''

    def __init__(self, module, every_n=10, change_factor=0):
        self.generated_images = []
        self.count = 1
        self.last_image = None
        self.max_every_n = 50

        self.every_n = every_n
        self.change_factor = change_factor



        self.hook = module.register_forward_hook(self.save_generated_image)
        print('initializing hook')

    def save_generated_image(self, module, input, output):
        image = output.detach().cpu().numpy()[0]
        if self.count % self.every_n == 0:
            self.generated_images.append(image)

        # update every_n if necessary:

        # after i save every 100
        if self.every_n < self.max_every_n and self.count == self.every_n * self.change_factor * 5: # if change_factor = 0, this will never be true so we will never change every_n
            self.every_n = min(self.change_factor * self.every_n, self.max_every_n)

        self.last_image = image
        self.count += 1

    def close(self):
        self.hook.remove()

    def get_images(self):
        return self.generated_images