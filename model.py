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
import math

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

        self.patch_size = self.original_model.patch_size
        self.interpolate_offset = self.original_model.interpolate_offset
        self.interpolate_antialias = self.original_model.interpolate_antialias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.original_model.to(self.device)

        self.patch_embed = self.original_model.patch_embed
        self.blocks = self.original_model.blocks
        self.norm = self.original_model.norm
        self.head = self.original_model.head
        self.cls_token_1 = self.original_model.cls_token.clone()
        self.cls_token_2 = self.original_model.cls_token.clone()
        self.pos_embed_1 = self.original_model.pos_embed.clone()
        self.pos_embed_2 = self.original_model.pos_embed.clone()

        # Final single-head attention layer
        embed_dim = self.original_model.patch_embed.proj.out_channels
        self.final_attention = SingleHeadAttention(embed_dim)

        # Freeze parameters if pretrained
        if pretrained:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for param in self.blocks.parameters():
                param.requires_grad = False
            for param in self.norm.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False

    def forward(self, x1, x2, debug=False):

        # -- Original Model Processing --- #
        self.original_model.eval()

        x1_dino = self.prepare_tokens(x1, img_cls=1)
        x2_dino = self.prepare_tokens(x2, img_cls=2)

        for blk in self.blocks:
            x1_dino = blk(x1_dino)
            x2_dino = blk(x2_dino)

        x1_dino = self.norm(x1_dino)
        x2_dino = self.norm(x2_dino)

        x1_dino = x1_dino[:, 1:, :]
        x2_dino = x2_dino[:, 1:, :]

        
        if debug:
            print("x1_img shape: ", x1.shape)
            print("x2_img shape: ", x2.shape)
            print("x1_dino shape: ", x1_dino.shape)
            print("x2_dino shape: ", x2_dino.shape)

        # CroDINO Processing
        x1 = self.prepare_tokens(x1, img_cls=1, debug=True)
        x2 = self.prepare_tokens(x2, img_cls=2)
        x = torch.cat((x1, x2), dim=1)
        
        # Process through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Final single-head attention
        _, final_attn = self.final_attention(x)         # NOTE: I'm computing the attention without affecting the patches 
        
        x = self.norm(x)
        x = self.head(x)
        
        return x1_dino, x2_dino, final_attn
    

    def interpolate_pos_encoding(self, x, img_cls, w, h):

        if img_cls == 1:
            previous_dtype = x.dtype
            npatch = x.shape[1] - 1
            N = self.pos_embed_1.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed_1
            pos_embed = self.pos_embed_1.float()
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            w0 = w // self.patch_size
            h0 = h // self.patch_size
            M = int(math.sqrt(N))  # Recover the number of patches in each dimension
            assert N == M * M
            kwargs = {}
            if self.interpolate_offset:
                # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
                # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
                sx = float(w0 + self.interpolate_offset) / M
                sy = float(h0 + self.interpolate_offset) / M
                kwargs["scale_factor"] = (sx, sy)
            else:
                # Simply specify an output size instead of a scale factor
                kwargs["size"] = (w0, h0)
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=self.interpolate_antialias,
                **kwargs,
            )
            assert (w0, h0) == patch_pos_embed.shape[-2:]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
        
        elif img_cls == 2:
            previous_dtype = x.dtype
            npatch = x.shape[1] - 1
            N = self.pos_embed_2.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed_2
            pos_embed = self.pos_embed_2.float()
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            w0 = w // self.patch_size
            h0 = h // self.patch_size
            M = int(math.sqrt(N))  # Recover the number of patches in each dimension
            assert N == M * M
            kwargs = {}
            if self.interpolate_offset:
                # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
                # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
                sx = float(w0 + self.interpolate_offset) / M
                sy = float(h0 + self.interpolate_offset) / M
                kwargs["scale_factor"] = (sx, sy)
            else:
                # Simply specify an output size instead of a scale factor
                kwargs["size"] = (w0, h0)
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=self.interpolate_antialias,
                **kwargs,
            )
            assert (w0, h0) == patch_pos_embed.shape[-2:]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
        
        else:
            return sys.exit("Invalid image class")
        
    
    def prepare_tokens(self, x, img_cls, debug=False):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        if img_cls == 1:
            x = torch.cat((self.cls_token_1.to(self.device).expand(x.shape[0], -1, -1), x), dim=1)
        elif img_cls == 2:
            x = torch.cat((self.cls_token_2.to(self.device).expand(x.shape[0], -1, -1), x), dim=1)
        
        x = x + self.interpolate_pos_encoding(x, img_cls, w, h).to(self.device)

        return x

        

def get_patch_embeddings(model, x):
    x = model.patch_embed(x)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x

# reg
class Dinov2Matcher:
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", half_precision=False, device="cuda"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.half_precision = half_precision
        self.device = device

        if self.half_precision:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])

    def prepare_image(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]
        # resize_scale = 1.0

        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size # crop a bit from right and bottom parts
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size, resize_scale

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

    def get_embedding_visualization(self, tokens, grid_size):
        pca = PCA(n_components=3)
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
        return normalized_tokens

    def get_combined_embedding_visualization(self, tokens1, tokens2, grid_size1, grid_size2, random_state=20):
        pca = PCA(n_components=3, random_state=random_state)

        token1_shape = tokens1.shape[0]
        combined_tokens = np.concatenate((tokens1, tokens2), axis=0)
        reduced_tokens = pca.fit_transform(combined_tokens.astype(np.float32))

        print("tokens1.shape", tokens1.shape)
        print("tokens2.shape", tokens2.shape)
        print("reduced_tokens.shape", reduced_tokens.shape)
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))

        rgbimg1 = normalized_tokens[0:token1_shape, :]
        rgbimg2 = normalized_tokens[token1_shape:, :]

        rgbimg1 = rgbimg1.reshape((*grid_size1, -1))
        rgbimg2 = rgbimg2.reshape((*grid_size2, -1))
        return rgbimg1, rgbimg2
    


def get_combined_embedding_visualization_all(tokens1, tokens2, tokens3, tokens4, grid_size1, grid_size2, grid_size3, grid_size4, random_state=20, debug=False):
        pca = PCA(n_components=3, random_state=random_state)

        token1_shape = tokens1.shape[0]
        token2_shape = tokens2.shape[0]
        token3_shape = tokens3.shape[0]
        token4_shape = tokens4.shape[0]

        combined_tokens = np.concatenate((tokens1, tokens2, tokens3, tokens4), axis=0)
        reduced_tokens = pca.fit_transform(combined_tokens.astype(np.float32))

        if debug:
            print("tokens_1.shape", tokens1.shape)
            print("tokens_2.shape", tokens2.shape)
            print("tokens_3.shape", tokens3.shape)
            print("tokens_4.shape", tokens4.shape)
            print("reduced_tokens.shape", reduced_tokens.shape)

        normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (np.max(reduced_tokens) - np.min(reduced_tokens))

        rgbimg1 = normalized_tokens[0:token1_shape, :]
        rgbimg2 = normalized_tokens[token1_shape:token1_shape+token2_shape, :]
        rgbimg3 = normalized_tokens[token1_shape+token2_shape:token1_shape+token2_shape+token3_shape, :]
        rgbimg4 = normalized_tokens[token1_shape+token2_shape+token3_shape:, :]

        if debug:
            print("rgbimg1 shape", rgbimg1.shape)
            print("rgbimg2 shape", rgbimg2.shape)
            print("rgbimg3 shape", rgbimg3.shape)
            print("rgbimg4 shape", rgbimg4.shape)

        rgbimg1 = rgbimg1.reshape((*grid_size1, -1))
        rgbimg2 = rgbimg2.reshape((*grid_size2, -1))
        rgbimg3 = rgbimg3.reshape((*grid_size3, -1))
        rgbimg4 = rgbimg4.reshape((*grid_size4, -1))

        return rgbimg1, rgbimg2, rgbimg3, rgbimg4



class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x1, x2):
        # x1_flat = x1.view(x1.size(0), -1)           # flatten the last two dimensions
        # x2_flat = x2.view(x2.size(0), -1)
        cos_sim = F.cosine_similarity(x1, x2, dim=-1)         # compute cosine similarity        
        loss = 1 - cos_sim.mean()                                       # convert similarity to loss
        # print("cos_sim shape: ", cos_sim.shape)
        # print("loss: ", loss)

        # 768 is the inner dimension -> output is 256 x 256


        # normalize so that each token has norm 1
        # then matrix multiplication for its self to be MxM
        # I want all diagonal elements to eb 1 and non-diag to be 0

        ## Implementation:
        # compute cross-entropy loss for the matrixs
        return loss