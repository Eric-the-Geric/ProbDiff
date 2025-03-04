import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Download and load the MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',  # Directory to save the dataset
    train=True,     # Use the training set
    download=True,  # Download if not already present
    transform=transform
)

# Create a DataLoader
batch_size = 128
dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Shuffle the data for training
    num_workers=2  # Use multiple workers for faster loading
)

# Test the DataLoader
for batch in dataloader:
    x_start, _ = batch  # x_start: (batch_size, 1, 28, 28)
    print("Batch shape:", x_start.shape)
    break

# 1. Define Noise Schedule (Forward Process)
def linear_beta_schedule(T=100, beta_start=1e-4, beta_end=0.02):
    """Linear schedule for beta_t (variance of noise)"""
    return torch.linspace(beta_start, beta_end, T)

T = 1000
betas = linear_beta_schedule(T)

# Pre-calculate terms for efficiency
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 2. forward Diffusion (Add noise)
def q_sample(x_start, t, noise=None):
    """Add noise to data at timestep t"""
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alpha = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    
    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise


# 3. Reverse Process (nerual net)

import torch.nn as nn

class UNet(nn.Module):
    """Simplified U-Net to predict noise"""
    def __init__(self):
        super().__init__()
        # Example architecture for MNIST (28x28 images)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64*28*28, 28*28)  # Predicts noise for each pixel
        
    def forward(self, x, t):
        # x: input noisy image (batch_size, 1, 28, 28)
        # t: timestep (batch_size)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x).view(-1, 1, 28, 28)  # Reshape to image


# 4. loss fn
def p_losses(model, x_start, t):
    """Compute loss between predicted and actual noise"""
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, noise)  # Add noise
    predicted_noise = model(x_noisy, t)     # Predict noise
    return nn.functional.mse_loss(noise, predicted_noise)

# 5. Training loop
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for epoch in range(10):
    cum_loss = 0
    print(f"{epoch}/10")
    for batch in dataloader:  # Assume MNIST DataLoader
        x_start, _ = batch
        t = torch.randint(0, T, (x_start.size(0),))  # Random timesteps
        loss = p_losses(model, x_start, t)
        cum_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(cum_loss)
    losses.append(cum_loss/128)
plt.plot(losses)
plt.show()



@torch.no_grad()
def p_sample(model, x, t):
    """Denoise for one timestep"""
    predicted_noise = model(x, t)
    # Simplified: Directly subtract predicted noise
    return x - predicted_noise * betas[t]

@torch.no_grad()
def sample(model, image_size=28):
    """Generate image from noise"""
    x = torch.randn((1, 1, image_size, image_size))  # Start with noise
    for t in reversed(range(T)):
        x = p_sample(model, x, t)
    return x.clamp(-1, 1)  # Clamp to valid pixel range


generated_image = sample(model, image_size=28)

# Visualize the generated image
import matplotlib.pyplot as plt
plt.imshow(generated_image.squeeze().cpu().numpy(), cmap='gray')
plt.show()
