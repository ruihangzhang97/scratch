import einops
from tqdm import tdqm

from torchsummary import summary

import torch
from torch import nn
import torchvision
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

patch_size = 16
latent_size = 768
n_channels = 3
num_heads = 12
num_encoders = 12
dropout = 0.1
num_classes = 10
size = 224

epochs = 10
base_lr = 10e-3
weight_decay = 0.03
batch_size = 4

class InputEmbedding(nn.Module):
    def __init__(self, patch_size=patch_size, n_channels=n_channels, device=device, latent_size=latent_size, batch_size=batch_size):
        super(InputEmbedding, self).__init__()
        self.latent_size = latent_size
        self.patch_size = latent_size
        self.n_channels = n_channels
        self.device = device
        self.batch_size = batch_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels # when flattened

        # linear projection layer
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)
        
        # class token
        self.class_token = nn.Parameter(torch.rand(self.batch_size, 1, self.latent_size)).to(self.device)

        # position embed
        self.pos_embedding = nn.Parameter(torch.rand(self.batch_size, 1, self.latent_size)).to(self.device)

    def forward(self, input_data):
        input_data = input_data.to(self.device)

        # patchify input image
            # b: batch_size
            # c: num_channels
            # h: number of patches along the height of the image
            # h1: height of each patch in pixels
                # (h h1): (h*h1) is height of image in total where we have h regions of h1 pixels
            # w: number of patches along the width of the image
            # w1: width of each patch in pixels
                # (w w1): (ws*w1) is width of image where we have w regions of w1 pixels
        patches = einops.rearrange(
            input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size
        )

        # example
        # input_data.size() = [1, 3, 224, 224]: [batch_size, num_channels, height, width]
        # patches.size() = [1, 196, 768]: [batch_size, num_patches, flattened_dim_of_patches]

        # N = HW/P^2 = 224 * 224 / (16^2) = 196 --> Number of patches of the input image
        # 768 = 16 * 16 * 3  

        linear_projection = self.linearProjection(patches).to(self.device)

        b, n, _ = linear_projection.shape

        # prepending class tokens
        # [1, 196, 768] -> [1, 197, 768]
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)

        # [1, 197, 768] -> same as linear_projection
        pos_embed = einops.repeat(self.pos_embedding, 'b 1 d -> b m d', m=n+1)

        linear_projection += pos_embed
        
        return linear_projection

