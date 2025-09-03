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
import torchvision.models as models

from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import normalize
from torchinfo import summary

from transformers import ViTImageProcessor, AutoModel

import warnings
warnings.simplefilter("ignore", category=UserWarning)
        

class CrossviewModel(nn.Module):
    def __init__(self, backbone='dinov2', frozen=True, device=None):
        super(CrossviewModel, self).__init__()

        self.backbone = backbone
        self.pretrained = frozen
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])

        if backbone == 'dinov2':

            self.original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.patch_size = self.original_model.patch_size
            self.interpolate_offset = self.original_model.interpolate_offset
            self.interpolate_antialias = self.original_model.interpolate_antialias
            self.original_model.to(self.device)
            self.original_model.eval()

            self.patch_embed = self.original_model.patch_embed
            self.blocks = self.original_model.blocks
            self.norm = self.original_model.norm
            self.head = self.original_model.head
            self.cls_token = self.original_model.cls_token.clone()
            self.pos_embed = self.original_model.pos_embed.clone()

            if frozen:
                for param in self.patch_embed.parameters():
                    param.requires_grad = False
                for param in self.blocks.parameters():
                    param.requires_grad = False
                for param in self.norm.parameters():
                    param.requires_grad = False
                for param in self.head.parameters():
                    param.requires_grad = False

        elif backbone == 'clip':

            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
            self.patch_size = 16
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        elif backbone == "resnet50":

            self.patch_size = 32  # ResNet50 does not use patch embeddings, but we can set a dummy value
            self.model = models.resnet50(pretrained=frozen)
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])  # remove fully connected (FC) layers and keep convolutional feature extractor

            if frozen:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

        elif backbone == "dinov3":

            self.processor = ViTImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
            self.model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")

            self.patch_size = self.model.config.patch_size

        elif backbone == "dinov3_crossview":

            self.ground_processor = ViTImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
            self.ground_model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")

            self.aerial_processor = ViTImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")
            self.aerial_model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")

            self.patch_size = self.ground_model.config.patch_size

        elif backbone == 'dinov3_sat':

            self.processor = ViTImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")
            self.model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")

            self.patch_size = self.model.config.patch_size

    def _forward_dinov3_crossview(self, ground_input, aerial_input, debug=False):

        ground_outputs = self.ground_model(**ground_input)
        aerial_outputs = self.aerial_model(**aerial_input)

        ground_batch_size, _, ground_img_height, ground_img_width = ground_input.pixel_values.shape
        aerial_batch_size, _, aerial_img_height, aerial_img_width = aerial_input.pixel_values.shape

        ground_num_patches_height, ground_num_patches_width = ground_img_height // self.patch_size, ground_img_width // self.patch_size
        aerial_num_patches_height, aerial_num_patches_width = aerial_img_height // self.patch_size, aerial_img_width // self.patch_size

        ground_num_patches_flat = ground_num_patches_height * ground_num_patches_width
        aerial_num_patches_flat = aerial_num_patches_height * aerial_num_patches_width

        ground_last_hidden_states = ground_outputs.last_hidden_state
        # print(ground_last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
        assert ground_last_hidden_states.shape == (ground_batch_size, 1 + self.ground_model.config.num_register_tokens + ground_num_patches_flat, self.ground_model.config.hidden_size)

        aerial_last_hidden_states = aerial_outputs.last_hidden_state
        # print(aerial_last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
        assert aerial_last_hidden_states.shape == (aerial_batch_size, 1 + self.aerial_model.config.num_register_tokens + aerial_num_patches_flat, self.aerial_model.config.hidden_size)

        ground_cls_token = ground_last_hidden_states[:, 0, :]
        ground_patch_features_flat = ground_last_hidden_states[:, 1 + self.ground_model.config.num_register_tokens:, :]
        ground_patch_features = ground_patch_features_flat.unflatten(1, (ground_num_patches_height, ground_num_patches_width))

        aerial_cls_token = aerial_last_hidden_states[:, 0, :]
        aerial_patch_features_flat = aerial_last_hidden_states[:, 1 + self.aerial_model.config.num_register_tokens:, :]
        aerial_patch_features = aerial_patch_features_flat.unflatten(1, (aerial_num_patches_height, aerial_num_patches_width))

        return ground_patch_features_flat, aerial_patch_features_flat

    def _forward_clip(self, ground_image, aerial_image, debug):

        ground_inputs = self.processor(images=ground_image, return_tensors="pt", do_rescale=False)
        aerial_inputs = self.processor(images=aerial_image, return_tensors="pt", do_rescale=False)

        # Get the intermediate feature maps
        ground_features = self.model.vision_model(pixel_values=ground_inputs["pixel_values"].to(self.device)).last_hidden_state.detach()
        aerial_features = self.model.vision_model(pixel_values=aerial_inputs["pixel_values"].to(self.device)).last_hidden_state.detach()

        ground_tokens = ground_features[:, 1:, :]
        aerial_tokens = aerial_features[:, 1:, :]

        if debug:
            print("ground_tokens shape: ", ground_tokens.shape)
            print("ground_cls shape: ", ground_features[:, 0, :].shape)

        ground_tokens = F.normalize(ground_tokens, dim=-1)
        aerial_tokens = F.normalize(aerial_tokens, dim=-1)

        if debug:
            print("ground_tokens shape: ", ground_tokens.shape)
            print("aerial_features shape: ", aerial_tokens.shape)

        return ground_tokens, aerial_tokens

    def _forward_dinov2(self, ground_image, aerial_image, debug):

        ground_tokens = self.prepare_tokens(ground_image)
        aerial_tokens = self.prepare_tokens(aerial_image)

        for blk in self.blocks:
            ground_tokens = blk(ground_tokens)
            aerial_tokens = blk(aerial_tokens)

        ground_tokens = self.norm(ground_tokens)
        aerial_tokens = self.norm(aerial_tokens)

        ground_tokens = ground_tokens[:, 1:, :]
        ground_cls = ground_tokens[:, :1, :]
        aerial_tokens = aerial_tokens[:, 1:, :]
        aerial_cls = aerial_tokens[:, :1, :]

        
        if debug:
            print("x1_img shape: ", ground_image.shape)
            print("x2_img shape: ", aerial_image.shape)
            print("x1_dino shape: ", ground_tokens.shape)
            print("x2_dino shape: ", aerial_tokens.shape)
            print("x1_cls shape: ", ground_cls.shape)
            print("x2_cls shape: ", aerial_cls.shape)

        return ground_tokens, aerial_tokens

    def _forward_dinov3(self, ground_input, aerial_input, debug=False):

        ground_outputs = self.model(**ground_input)
        aerial_outputs = self.model(**aerial_input)

        ground_batch_size, _, ground_img_height, ground_img_width = ground_input.pixel_values.shape
        aerial_batch_size, _, aerial_img_height, aerial_img_width = aerial_input.pixel_values.shape

        ground_num_patches_height, ground_num_patches_width = ground_img_height // self.patch_size, ground_img_width // self.patch_size
        aerial_num_patches_height, aerial_num_patches_width = aerial_img_height // self.patch_size, aerial_img_width // self.patch_size

        ground_num_patches_flat = ground_num_patches_height * ground_num_patches_width
        aerial_num_patches_flat = aerial_num_patches_height * aerial_num_patches_width

        ground_last_hidden_states = ground_outputs.last_hidden_state
        # print(ground_last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
        assert ground_last_hidden_states.shape == (ground_batch_size, 1 + self.model.config.num_register_tokens + ground_num_patches_flat, self.model.config.hidden_size)

        aerial_last_hidden_states = aerial_outputs.last_hidden_state
        # print(aerial_last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
        assert aerial_last_hidden_states.shape == (aerial_batch_size, 1 + self.model.config.num_register_tokens + aerial_num_patches_flat, self.model.config.hidden_size)

        ground_cls_token = ground_last_hidden_states[:, 0, :]
        ground_patch_features_flat = ground_last_hidden_states[:, 1 + self.model.config.num_register_tokens:, :]
        ground_patch_features = ground_patch_features_flat.unflatten(1, (ground_num_patches_height, ground_num_patches_width))

        aerial_cls_token = aerial_last_hidden_states[:, 0, :]
        aerial_patch_features_flat = aerial_last_hidden_states[:, 1 + self.model.config.num_register_tokens:, :]
        aerial_patch_features = aerial_patch_features_flat.unflatten(1, (aerial_num_patches_height, aerial_num_patches_width))

        return ground_patch_features_flat, aerial_patch_features_flat

    def _forward_resnet50(self, ground_image, aerial_image, debug):

        ground_features = self.feature_extractor(ground_image)
        aerial_features = self.feature_extractor(aerial_image)

        ground_tokens = ground_features.view(ground_features.size(0), -1, 2048)
        aerial_tokens = aerial_features.view(aerial_features.size(0), -1, 2048)

        return ground_tokens, aerial_tokens

    def forward(self, ground_input, aerial_input, debug=False):
        
        if self.backbone == 'clip':
            return self._forward_clip(ground_input, aerial_input, debug)
        elif self.backbone == 'dinov2':
            return self._forward_dinov2(ground_input, aerial_input, debug)
        elif self.backbone == 'resnet50':
            return self._forward_resnet50(ground_input, aerial_input, debug)
        elif self.backbone == 'dinov3':
            return self._forward_dinov3(ground_input, aerial_input, debug)
        elif self.backbone == 'dinov3_crossview':
            return self._forward_dinov3_crossview(ground_input, aerial_input, debug)
        elif self.backbone == 'dinov3_sat':
            return self._forward_dinov3(ground_input, aerial_input, debug)
        else:
            raise ValueError("Unsupported backbone: {}".format(self.backbone))

    def prepare_tokens(self, x, debug=False):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.to(self.device).expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h).to(self.device)
        return x
    
    def get_patch_embeddings(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def interpolate_pos_encoding(self, x, w, h):
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        M = int(math.sqrt(patch_pos_embed.shape[1]))
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            size=(w0, h0),
            mode="bicubic",
            align_corners=False
        ).permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def show(self):
        print("\n\n=====================================================================================\n")
        print(self)
        if self.backbone == "cnn":
            summary(self, input_size=(16, 512, 512, 3))
        else:
            summary(self)

    def get_embedding_visualization(self, tokens, grid_size):
        pca = PCA(n_components=3)
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
        return normalized_tokens
    
    def show_tokens(self, imgs_tokens, grid_shape=None, mode="show", results_path=None, dpi=300, return_tokens=False):

        B, n, C = imgs_tokens.shape
        n_patches = int(math.sqrt(n))

        if return_tokens:
            out = []

        if grid_shape is None:
            side = int(np.ceil(np.sqrt(B)))
            grid_shape = (side, side)

        fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(grid_shape[1]*4, grid_shape[0]*4))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = np.array([axes])

        for i in range(B):
            ax = axes[i]
            img_tokens = imgs_tokens[i, :, :].detach().cpu().numpy()
            
            ax.axis('off')

            vis_tokens = self.get_embedding_visualization(img_tokens, (n_patches, n_patches))
            out.append(vis_tokens) if return_tokens else None
            ax.imshow(vis_tokens)

        # Hide any remaining axes
        for j in range(B, len(axes)):
            axes[j].axis('off')

        fig.tight_layout()

        if mode == "save":
            if results_path is None:
                save_path = "tokens.png"
            else:
                save_path = results_path
            fig.savefig(save_path, dpi=dpi)
            plt.show()
            plt.close(fig)
        elif mode == "show":
            plt.show()

        if return_tokens:
            vis_tokens = np.stack(out, axis=0)
            return vis_tokens
        
    def prepare_image(self, rgb_image_numpy, patch_size=None):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]
        # resize_scale = 1.0

        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size # crop a bit from right and bottom parts
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        if patch_size is None:
            grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)
        else:
            grid_size = (cropped_height // patch_size, cropped_width // patch_size)
        return image_tensor, grid_size, resize_scale

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half()
            else:
                image_batch = image_tensor.unsqueeze(0)

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


# reg
class Dinov2Matcher:
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", half_precision=False, device="cuda"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.half_precision = half_precision
        self.device = device

        if self.half_precision:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half()
        else:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])

    def prepare_image(self, rgb_image_numpy, patch_size=None):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]
        # resize_scale = 1.0

        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size # crop a bit from right and bottom parts
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        if patch_size is None:
            grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        else:
            grid_size = (cropped_height // patch_size, cropped_width // patch_size)
        return image_tensor, grid_size, resize_scale

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half()
            else:
                image_batch = image_tensor.unsqueeze(0)

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