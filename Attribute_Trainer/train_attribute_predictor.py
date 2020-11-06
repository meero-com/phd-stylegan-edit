from fastai.vision.learner import cnn_learner
import sys
sys.path.append("/home")

# this will be the savepath, remember that we are in my docker image currently!! (which is why datadrive is in home)
checkpoint_path = '/home/datadrive/asya/checkpoints/celeba_attributes/attribute_predictor_40'

from utils.model_tools.customdatasets import CelebADataset
from pathlib import Path
import pandas as pd
import time

from fastai.vision.data import *
from fastai.vision import *
from fastai.vision.all import *
from fastai.data.transforms import *


# get the paths to celebA
path_celeba = Path('/home/datadrive/asya/datasets/celeba')
img_path = path_celeba/'Img/img_align_celeba_png'
anno = path_celeba/'Anno'


# attributes
all_attributes = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
all_attributes = all_attributes.split(' ')



relevent_attribute_names = all_attributes
celeba_ds = CelebADataset(path_celeba, 
                          relevant_attribute_names=relevent_attribute_names,
                          split='train_valid')
celeba_ds_test = CelebADataset(path_celeba, 
                          relevant_attribute_names=relevent_attribute_names,
                          split='test')





# Data Organization:

df = pd.DataFrame()
df['names'] = celeba_ds.filenames

# get the labels

tiled = np.tile(celeba_ds.relevant_attribute_names, reps=(df.shape[0], 1))
mask = np.array(celeba_ds.relevant_attributes == 0) # because the mask below signifies INvalid entries
attr = np.ma.masked_array(tiled, mask=mask)
list_atts = attr.tolist()
labels = [' '.join([aa for aa in a if aa is not None]) for a in list_atts]
df['labels'] = labels

# validation set
df['is_valid'] = celeba_ds.splits == 1 # if it is a validation entity, since we are only with train/validation data




# dataloader for test set 
df_test = pd.DataFrame()
df_test['names'] = celeba_ds_test.filenames

# get the labels
tiled = np.tile(celeba_ds_test.relevant_attribute_names, reps=(df_test.shape[0], 1))
mask = np.array(celeba_ds_test.relevant_attributes == 0) # because the mask below signifies INvalid entries
attr = np.ma.masked_array(tiled, mask=mask)
list_atts = attr.tolist()
labels = [' '.join([aa for aa in a if aa is not None]) for a in list_atts]
df_test['labels'] = labels



augments = aug_transforms(max_rotate=0., max_zoom=1., max_warp=0., p_affine=0., size=224) + [Normalize.from_stats(*imagenet_stats)]

# different dataloaders. celeba_dummy is just for testing to make sure things work.
celeba = ImageDataLoaders.from_df(df, path=img_path, valid_col='is_valid', label_delim=' ', bs=64, batch_tfms=augments) # big batch size according to fastai
celeba_test = ImageDataLoaders.from_df(df_test, path=img_path, valid_pct=0., label_delim=' ', bs=64, batch_tfms=augments) # valid_pct=0 means that everything here will be in the "train" part
celeba_dummy = ImageDataLoaders.from_df(df_test, path=img_path, valid_pct=0.9, label_delim=' ', bs=64, batch_tfms=augments) # we put almost nothing in the train part so that we can do tests quickly on this set

metrics = [accuracy_multi]

learn = cnn_learner(celeba, resnet50, metrics=metrics)
lr_min, lr_steep = learn.lr_find()

s = time.time()
learn.fine_tune(epochs=5, base_lr=lr_steep, freeze_epochs=2) 
e = time.time()
print("total time to train:", e - s)

# save model !
learn.save(checkpoint_path)