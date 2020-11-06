import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import pandas as pd 
from PIL import Image

import numpy as np
import os


class CelebADataset(torch.utils.data.Dataset):
    '''

    root: This is the directory of celeba. As I downloaded it, I assume that the general structure of celeba is as follows:
        celeba
            > Anno
                > identity_CelebA.txt
                > list_attr_celeba.txt
                > list_bbox_celeba.txt
                > list_landmarks_allign_celeba.txt
                > list_landmarks_celeba.txt
            > Eval
                > list_eval_partition.txt
            > Img
                > img_align_celeba 
                    > aligned_images.jpg 
                > img_align_celeba_png  
                    > aligned_images.png
                > img_celeba
                    > unaligned_images.jpg

    split: 3 possibilities:
        - "train"
        - "valid"
        - "test"
        - "all"

    target_type: one value or a list of values. !!!! By default, the filenames are also given
        - "attr"
        - "id"
        - "bbox"
        - "landmarks"
        - "filename"
        - "orig_img"

    transforms: transforms for the image

    target_transforms: currently not implemented, transforms to apply to the target

    use_png: whether to get png images or jpg images
    '''

    def verify_split_string(self, string):
        if string not in ['train', 'valid', 'test', 'train_valid', 'all']:
            raise ValueError("split argument must be either 'train', 'valid', 'test', 'train_valid', or 'all'")
        return string
    
    # if use_png = True then we will get the images from Img/img_align_celeba_png
    # otherwise we will get the images from Img/img_align_celeba
    
    def __init__(self, root, split='train', target_type='attr', relevant_attribute_names=None, transform=None, target_transform=None, use_png=True):
        # possible values for target_type:
        # - "attr"
        # - "id"
        # - "bbox"
        # - "landmarks"

        #super(CelebADataset, self).__init__()
        
        self.split = split
        self.root = root
        self.use_png = use_png
        self.transform = transform

        
        # for attributes - let's see if I keep this
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        
        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        for t in self.target_type:
            if t not in ['orig_img', 'attr', 'id', 'bbox', 'landmarks', 'filename']:
                print("Target type {} is not a valid option. Ignoring.".format(t))
        
        ## each image belongs either to a train, validation or test set inside celeba.
        # we want to know which one we are getting
        # by default, we will use the train set (if )
        split_map = {'train': 0, 'valid': 1, 'test': 2, 'train_valid': 3, 'all': None}
        my_split = split_map[self.verify_split_string(split)]
        
        # get all relative information
        partition_path = os.path.join(root, 'Eval/list_eval_partition.txt')
        identity_path = os.path.join(root, 'Anno/identity_CelebA.txt')
        bbox_path = os.path.join(root, 'Anno/list_bbox_celeba.txt')
        landmarks_path = os.path.join(root, 'Anno/list_landmarks_align_celeba.txt')
        attributes_path = os.path.join(root, 'Anno/list_attr_celeba.txt')
        
        splits = pd.read_csv(partition_path, delim_whitespace=True, header=None, index_col=0)
        identity = pd.read_csv(identity_path, delim_whitespace=True, header=None, index_col=0)
        bbox = pd.read_csv(bbox_path, delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pd.read_csv(landmarks_path, delim_whitespace=True, header=1)
        attributes = pd.read_csv(attributes_path, delim_whitespace=True, header=1)
        
        # get only the ones that interest me
        if my_split is None:
            mask = splits[1] >= 0 # just get all of my elements
        elif my_split == 3:
            mask = splits[1] < 2 # get train and validation
        else:
            mask = splits[1] == my_split # get the pertinent elements
            
        self.filenames = splits[mask].index.values
        if use_png:
            self.filenames = [fname.replace('.jpg', '.png') for fname in self.filenames]
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attributes = torch.as_tensor(attributes[mask].values, dtype=torch.float32)
        self.attributes = (self.attributes + 1) // 2 # map attributes -1,1 --> 0,1
        self.attribute_names = list(attributes[mask].columns)
        self.splits = splits[mask].values.T[0]# torch.as_tensor(splits[mask].values)
        

        if relevant_attribute_names:
            self.relevant_indexes = [self.attribute_names.index(name) for name in relevant_attribute_names]
            self.relevant_attribute_names = relevant_attribute_names
            self.relevant_attributes = self.attributes[:, self.relevant_indexes]
        else:
            self.relevant_indexes = None #range(len(self.attribute_names))
            self.relevant_attribute_names = self.attribute_names
            self.relevant_attributes = self.attributes

    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        if self.use_png:
            filename = os.path.splitext(filename)[0] + '.png' # change filename to .png extension
            image_path = os.path.join('Img/img_align_celeba_png', filename)
        else:
            image_path = os.path.join('Img/img_align_celeba', filename)
        path_to_image = os.path.join(self.root, image_path)
        
        # get the image
        X_orig = Image.open(path_to_image)
        #X = np.asarray(Image.open(path_to_image))
        #X = X.transpose((2, 0, 1))
        #X = torch.from_numpy(X)

        target = []
        
        for t in self.target_type:
            if t == 'orig_img':
                target.append(np.array(X_orig))
            if t == 'attr':
                if self.relevant_indexes:
                    target.append(self.attributes[index][self.relevant_indexes])
                else: # append everything
                    target.append(self.attributes[index])
            if t == 'id':
                target.append(self.identity[index])
            if t == 'bbox':
                target.append(self.bbox[index])
            if t == 'landmarks':
                target.append(self.landmarks_align[index])
            if t == 'filename':
                target.append(filename)
        
        if self.transform: 
            X = self.transform(X_orig)
        
        if target:
            if len(target) > 1:
                target = tuple(target)
            else:
                target = target[0]
        else:
            target = None
            
        # target transform, skip for now. 
        
        return X, target

class CelebAFastaiDataset(CelebADataset):
    def __init__(self, root, split='train'):
        super(CelebADataset, self).__init__(root=root, split='train')
        self.c = len(self.attribute_names)
        self.classes = self.attribute_names

class ImageTestSet(torch.utils.data.Dataset):
    def __init__(self, filenames, image_size=256, transforms1=None, transforms2=None):
        self.filenames = filenames
        self.image_size = image_size
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))
        
        if self.transforms1:
            image1 = self.transforms1(image)
        if self.transforms2:
            image2 = self.transforms2(image)

        return image1, image2

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))
        #print(filename)
        #print(image.shape)
        if image.shape[-1] == 4: # sometimes there is an alpha channel that we don't want
            image = image[:, :, :3]

        return image