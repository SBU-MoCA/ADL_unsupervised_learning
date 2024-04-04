import torch
from torchvision.models import vit_b_16

# Load a pre-trained Vision Transformer model
model = vit_b_16(pretrained=True)

# Model is ready to be used for inference or further training
