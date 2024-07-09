import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import ImageFile
from tqdm import tqdm
import numpy as np
from dataset import PairedImagesDataset, sample_paired_images
from model import CroDINO, CosineSimilarityLoss

def train(model, train_loader, val_loader, device, criterion, optimizer, epochs=1, save_path='untitled', debug=False):
    model.to(device)
    
    model_path = os.path.join('models', save_path)
    os.makedirs(model_path, exist_ok=True)

    best_val_loss = np.inf
    patience_counter = 0
    best_model_path = None
    sparse_attention_weight=0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for ground_images, aerial_images in train_loader:
                ground_images, aerial_images = ground_images.to(device), aerial_images.to(device)

                # Forward pass
                ground_tokens, aerial_tokens, attention = model(ground_images, aerial_images)            

                # Visualize the final single-head attention layer
                attention = attention.mean(dim=1)  # average across heads only

                # Calculate the number of patches for ground and aerial images
                num_patches_ground = (ground_images.shape[-1] // model.patch_size) * (ground_images.shape[-2] // model.patch_size)
                num_patches_aerial = (aerial_images.shape[-1] // model.patch_size) * (aerial_images.shape[-2] // model.patch_size)

                # Remove the first row and column
                attention = attention[:, 1:, 1:]

                # Remove the row and column corresponding to 1+num_patches_ground
                attention = torch.cat((attention[:, :num_patches_ground-1, :], attention[:, num_patches_ground:, :]), dim=1)
                attention = torch.cat((attention[:, :, :num_patches_ground-1], attention[:, :, num_patches_ground:]), dim=2)

                # Get the Cross Attentions
                cross_attention_A2G = attention[:, :num_patches_ground, num_patches_ground:]
                cross_attention_G2A = attention[:, num_patches_ground:, :num_patches_ground]

                # Reconstruct the images from the tokens
                reconstructed_aerial = torch.matmul(cross_attention_G2A, ground_tokens)
                reconstructed_ground = torch.matmul(cross_attention_A2G, aerial_tokens)

                if debug:
                    print("attention shape: ", attention.shape)
                    print("attention dtype: ", attention.dtype)
                    print("num_patches_ground: ", num_patches_ground)
                    print("num_patches_aerial: ", num_patches_aerial)
                    print("cross_attention_G2A shape: ", cross_attention_G2A.shape)
                    print("cross_attention_A2G shape: ", cross_attention_A2G.shape)
                    print("reconstructed_ground shape: ", reconstructed_ground.shape)
                    print("reconstructed_aerial shape: ", reconstructed_aerial.shape)

                # Compute Loss
                loss_ground = criterion(reconstructed_ground, ground_tokens)
                loss_aerial = criterion(reconstructed_aerial, aerial_tokens)
                # loss = loss_ground + loss_aerial
                loss = loss_aerial

                # # Calculate sparse attention loss
                # sparse_attention_loss = sparse_attention_weight * torch.norm(attention, p=1)    # L1 norm
                # loss += sparse_attention_loss
                
                running_loss += loss.item()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                pbar.set_postfix({'Loss': running_loss / (pbar.n + 1)})
                pbar.update()

        train_loss = running_loss / len(train_loader)
        
        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = os.path.join(model_path, f'best_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

    # Save the last model
    torch.save(model.state_dict(), os.path.join(model_path, f'last_model_epoch_{epoch+1}.pth'))

    print('Training Complete!\nBest Validation Loss:', best_val_loss)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for ground_images, aerial_images in val_loader:
                ground_images, aerial_images = ground_images.to(device), aerial_images.to(device)

                # Forward pass
                ground_tokens, aerial_tokens, attention = model(ground_images, aerial_images)            

                # Visualize the final single-head attention layer
                attention = attention.mean(dim=1)  # average across heads only

                # Calculate the number of patches for ground and aerial images
                num_patches_ground = (ground_images.shape[-1] // model.patch_size) * (ground_images.shape[-2] // model.patch_size)
                num_patches_aerial = (aerial_images.shape[-1] // model.patch_size) * (aerial_images.shape[-2] // model.patch_size)

                # Remove the first row and column
                attention = attention[:, 1:, 1:]

                # Remove the row and column corresponding to 1+num_patches_ground
                attention = torch.cat((attention[:, :num_patches_ground-1, :], attention[:, num_patches_ground:, :]), dim=1)
                attention = torch.cat((attention[:, :, :num_patches_ground-1], attention[:, :, num_patches_ground:]), dim=2)

                # Get the Cross Attentions
                cross_attention_A2G = attention[:, :num_patches_ground, num_patches_ground:]    # OCCHIO QUA
                cross_attention_G2A = attention[:, num_patches_ground:, :num_patches_ground]

                # Reconstruct the images from the tokens
                reconstructed_aerial = torch.matmul(cross_attention_G2A, ground_tokens)
                reconstructed_ground = torch.matmul(cross_attention_A2G, aerial_tokens)

                # Compute Loss
                loss_ground = criterion(reconstructed_ground, ground_tokens)
                loss_aerial = criterion(reconstructed_aerial, aerial_tokens)
                # loss = loss_ground + loss_aerial
                loss = loss_aerial

                val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--save_path', '-p', type=str, default='untitled', help='Path to save the model and results')
    parser.add_argument('--epochs', '-e', type=int, default=1, help='Number of epochs to train')
    args = parser.parse_args()

    # Constants
    image_size = 224
    aerial_scaling = 2
    batch_size = 16
    shuffle = True

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    repo_name="facebookresearch/dinov2"
    model_name="dinov2_vitb14"
    model = CroDINO(repo_name, model_name, pretrained=True).to(device)
    print(model)


    # Optimizer
    learning_rate = 1e-3
    weight_decay = 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Loss function
    # criterion = nn.MSELoss()
    criterion = CosineSimilarityLoss()

    # Transformations
    transform_ground = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    transform_aerial = transforms.Compose([
        # transforms.CenterCrop((image_size // aerial_scaling, image_size // aerial_scaling)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    # Enable loading truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Sample paired images
    dataset_path = '/home/lrusso/cvusa'
    train_filenames, val_filenames = sample_paired_images(dataset_path, sample_percentage=0.01, split_ratio=0.8, groundtype='cutouts')

    # Instantiate the dataset and dataloader
    train_dataset = PairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground)
    val_dataset = PairedImagesDataset(val_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    # Train the model
    train(model, train_loader, val_loader, device, criterion, optimizer, epochs=args.epochs, save_path=args.save_path)
