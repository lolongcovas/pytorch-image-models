import torch
import numpy as np
import os
import torch_neuron
from timm import create_model
import logging

## Enable logging so we can see any important warnings
logger = logging.getLogger('Neuron')
logger.setLevel(logging.INFO)

bsize = 16

image = torch.zeros([bsize, 3, 224, 224], dtype=torch.float32)

## Load a pretrained ResNet50 model
# model = models.resnet50(pretrained=True)
model = create_model(model_name="resnet50", pretrained=True, num_classes=1000)

## Tell the model we are using it for evaluation (not training)
model.eval()

## Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model( model, example_inputs=[image] )

## Now compile the model - with logging set to "info" we will see
## what compiles for Neuron, and if there are any fallbacks
## Note: The "-O2" setting is default in recent releases, but may be needed for DLAMI
##       and older installed environments
model_neuron = torch.neuron.trace(model, example_inputs=[image], compiler_args="-O2")

## Export to saved model
model_neuron.save("resnet50_neuron.pt")
