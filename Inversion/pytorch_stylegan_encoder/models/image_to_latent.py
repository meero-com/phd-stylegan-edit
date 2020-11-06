import torch
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from torchvision import transforms

class ImageToLatent(torch.nn.Module):
    # WE ASSUME THAT THE INPUT IS 224 x 224 !!!
    # latent_n is the number of latent variables we want, it is either 18 or 1 (in which case it is the same one repeated 18 times)
    def __init__(self, latent_n=18, input_size=224):
        super().__init__()
        
        self.latent_n = latent_n

        self.activation = torch.nn.ELU()
        
        self.resnet = list(resnet50(pretrained=True).children())[:-2]
        self.resnet = torch.nn.Sequential(*self.resnet)
        self.conv2d = torch.nn.Conv2d(2048, 256, kernel_size=1) # kernel_size=1 so the dimensions don't change, it was still 64x64 here
        self.flatten = torch.nn.Flatten()
        if input_size == 256:
            midsize = 16384
        elif input_size == 224:
            midsize = 12544
        else:
            raise ValueError("Input size must be either 224 or 256")
        self.dense1 = torch.nn.Linear(midsize, 256) # 12544 is only because the input is 224 x 224 !
        self.dense2 = torch.nn.Linear(256, self.latent_n * 512)

    def forward(self, image):
        #print("x shape at beginning:", image.shape)

        x = self.resnet(image)
        #print("after resnet application:", x.shape)

        x = self.conv2d(x)
        #print("after conv2d:", x.shape)

        x = self.flatten(x)
        #print("after flatten:", x.shape)

        x = self.dense1(x)
        #print("after dense1:", x.shape)

        x = self.dense2(x)
        #print("after dense2:", x.shape)

        if self.latent_n != 1:
            x = x.view((-1, self.latent_n, 512))
        else:
            x = x.view((-1, 512))
        #print("after final view:", x.shape)

        return x


class ImageTestSet(torch.utils.data.Dataset):
    def __init__(self, filenames, transforms1=None, transforms2=None):
        self.filenames = filenames
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        #self.dlatents = dlatents

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        #dlatent = self.dlatents[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))
        if self.transforms1:
            image1 = self.transforms1(image)
        if self.transforms2:
            image2 = self.transforms2(image)

        return image1, image2, filename

    def load_image(self, filename):
        image = np.asarray(Image.open(filename).convert('RGB'))

        return image

class ImageLatentDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, dlatents, transforms=None):
        self.filenames = filenames
        self.dlatents = dlatents
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        dlatent = self.dlatents[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))

        if self.transforms:
            image = self.transforms(image)

        return image, dlatent

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))

        return image

class EvalImageLatentDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, dlatents, transforms1=None, transforms2=None):
        self.filenames = filenames
        self.dlatents = dlatents
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        dlatent = self.dlatents[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))

        if self.transforms2:
            image2 = self.transforms2(image)
        else:
            image2 = None
        if self.transforms1:
            image = self.transforms1(image)
        

        return image, image2, dlatent

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))

        return image