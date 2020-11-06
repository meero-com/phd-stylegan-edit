# Attribute Trainer

## Requirements
All requirements are in docker image

## Usage
We can use this code directly by simply calling `python train_attribute_predictor.py`
This will train a resnet50 model with the fastai library to learn all 40 attributes of celebA. 
We can refer to the notebooks for fastai usage and other metrics.

The only thing that we should potentially change is the checkpoint path if we want to save in a different location or the attribute names (must be a subset of the 40 attributes). 


After we predict, we can use the notebook "save_attribute_scores" to save all the attribute scores for our latent dataset. Attention, no need to do this if we already have these scores saved!