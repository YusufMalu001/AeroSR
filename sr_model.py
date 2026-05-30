import torch
import torch.nn as nn
import torch.nn.functional as F

class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network (ESPCN)
    Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.
    """
    def __init__(self, upscale_factor=4, channels=3):
        super(ESPCN, self).__init__()
        
        # Feature extraction layers in low-resolution space
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Sub-pixel convolution layer
        # Output channels = channels * (upscale_factor ** 2)
        self.conv3 = nn.Conv2d(32, channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        
        # Pixel shuffle reshapes (channels * r^2) x H x W into channels x (r*H) x (r*W)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

if __name__ == '__main__':
    model = ESPCN(upscale_factor=4)
    # Test with a dummy Low Resolution image batch (B, C, H, W)
    dummy_input = torch.randn(1, 3, 200, 200)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape} -> ESPCN -> Output: {output.shape}")
