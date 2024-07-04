from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import random
from utils import *
import torch.nn.functional as F

import warnings
warnings.simplefilter("ignore", category=UserWarning)
        
class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SingleHeadAttention, self).__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.scale = (embed_dim // 1) ** -0.5  # Single head
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 1, C // 1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_weights = attn.clone()
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_weights

class CroDINO(nn.Module):
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", pretrained=True):
        super(CroDINO, self).__init__()
        self.original_model = torch.hub.load(repo_name, model_name)
        
        self.patch_embed = self.original_model.patch_embed
        self.blocks = self.original_model.blocks
        self.norm = self.original_model.norm
        self.head = self.original_model.head
        self.cls_token = self.original_model.cls_token
        
        # Final single-head attention layer
        embed_dim = self.original_model.patch_embed.proj.out_channels
        self.final_attention = SingleHeadAttention(embed_dim)

        # Positional Encoding
        # self.pos_embed_1 = nn.Parameter(self.original_model.pos_embed)  # use the original positional embedding and adjust dynamically
        # self.pos_embed_2 = nn.Parameter(self.original_model.pos_embed)  # use the original positional embedding and adjust dynamically
        # self.pos_embed_1 = self.original_model.pos_embed.copy()
        # self.pos_embed_2 = self.original_model.pos_embed.copy()
        self.pos_embed_1 = self.original_model.pos_embed
        self.pos_embed_2 = self.original_model.pos_embed

        # Freeze parameters if pretrained is True
        if pretrained:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for param in self.blocks.parameters():
                param.requires_grad = False
            for param in self.norm.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False

    def forward(self, x1, x2, return_attention=False):
        # x1_patches = self.patch_embed(x1)
        # x2_patches = self.patch_embed(x2)
        
        # # Compute positional encodings dynamically
        # num_patches_x1 = x1_patches.size(1)
        # num_patches_x2 = x2_patches.size(1)
        # print("x1 shape: ", x1_patches.shape)
        # print("x2 shape: ", x2_patches.shape)

        # # Interpolate positional embeddings to match the number of patches
        # pos_embed_x1 = self.original_model.prepare_tokens_with_masks(x1)
        # pos_embed_x2 = self.original_model.prepare_tokens_with_masks(x2)
        # print("pos_embed_x1 shape: ", pos_embed_x1.shape)
        # print("pos_embed_x2 shape: ", pos_embed_x2.shape)
        
        # # Concatenate tokens from both images
        # x_patches = torch.cat((x1_patches, x2_patches), dim=1)
        
        # # Add positional encoding
        # pos_embed = torch.cat((pos_embed_x1, pos_embed_x2), dim=1)
        # x = x_patches + pos_embed

        x1 = self.original_model.prepare_tokens_with_masks(x1)
        x2 = self.original_model.prepare_tokens_with_masks(x2)
        x = torch.cat((x1, x2), dim=1)
        
        # Process through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Final single-head attention
        _, final_attn = self.final_attention(x)         # NOTE: I'm computing the attention without affecting the patches           
        
        x = self.norm(x)
        x = self.head(x)
        
        if return_attention:
            return x, final_attn
        else:
            return x
        
    
    # def interpolate_pos_embed(self, pos_embed, num_patches):
    #     """Interpolate positional embeddings to match the number of patches."""
    #     original_num_patches = pos_embed.size(1)
    #     if num_patches != original_num_patches:
    #         # Calculate scaling factors for each dimension
    #         scale_h = float(num_patches) / float(original_num_patches)
    #         scale_w = 1.0  # Keep the width unchanged
    #         # Calculate new positional embedding size
    #         new_pos_embed_size = (int(scale_h * pos_embed.shape[1] + 0.5), int(scale_w * pos_embed.shape[2]))
    #         # Interpolate the positional embedding tensor
    #         pos_embed = F.interpolate(pos_embed.unsqueeze(0), size=new_pos_embed_size, mode='bicubic', align_corners=False)
    #         pos_embed = pos_embed.squeeze(0)
    #     return pos_embed

        

def get_patch_embeddings(model, x):
    x = model.patch_embed(x)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x


class Dinov2Matcher:
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=448, half_precision=False, device="cuda"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device

        if self.half_precision:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])

    def prepare_image(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]

        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size # crop a bit from right and bottom parts
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size, resize_scale

    def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
        cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()
        return resized_mask

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()

    def idx_to_source_position(self, idx, grid_size, resize_scale):
        row = (idx // grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
        col = (idx % grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
        return row, col

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
        return normalized_tokens

    def get_combined_embedding_visualization(self, tokens1, token2, grid_size1, grid_size2, mask1=None, mask2=None, random_state=20):
        pca = PCA(n_components=3, random_state=random_state)

        token1_shape = tokens1.shape[0]
        if mask1 is not None:
            tokens1 = tokens1[mask1]
        if mask2 is not None:
            token2 = token2[mask2]
        combinedtokens= np.concatenate((tokens1, token2), axis=0)
        reduced_tokens = pca.fit_transform(combinedtokens.astype(np.float32))

        if mask1 is not None and mask2 is not None:
            resized_mask = np.concatenate((mask1, mask2), axis=0)
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        elif mask1 is not None and mask2 is None:
            return sys.exit("Either use both masks or none")
        elif mask1 is None and mask2 is not None:
            return sys.exit("Either use both masks or none")

        print("tokens1.shape", tokens1.shape)
        print("token2.shape", token2.shape)
        print("reduced_tokens.shape", reduced_tokens.shape)
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))

        rgbimg1 = normalized_tokens[0:token1_shape,:]
        rgbimg2 = normalized_tokens[token1_shape:,:]

        rgbimg1 = rgbimg1.reshape((*grid_size1, -1))
        rgbimg2 = rgbimg2.reshape((*grid_size2, -1))
        return rgbimg1,rgbimg2
