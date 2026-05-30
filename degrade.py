import cv2
import numpy as np

def synthesize_haze(image, t=0.85, A=240):
    """
    Simulate atmospheric scattering (haze).
    I(x) = J(x)t(x) + A(1 - t(x))
    t: transmission map (scalar for uniform haze)
    A: atmospheric light
    """
    # Convert image to float32
    img_float = image.astype(np.float32)
    # Apply haze model
    hazed = img_float * t + A * (1 - t)
    # Clip and convert back to uint8
    return np.clip(hazed, 0, 255).astype(np.uint8)

def degrade_image(image, scale_factor=4, noise_sigma=15, blur_kernel=5, haze_t=0.85, haze_A=240, apply_haze=True):
    """
    Synthetically degrades a high-resolution image to simulate a low-resolution, noisy, blurry, hazy aerial image.
    
    Returns:
        lr_img: The actual low-resolution degraded image (W//4, H//4)
        bicubic_hr: The LR image upscaled back to original HR dimensions using bicubic interpolation (baseline)
    """
    h, w = image.shape[:2]
    
    # 1. Downscale
    lr = cv2.resize(image, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_CUBIC)
    
    # 2. Add Haze
    if apply_haze:
        lr = synthesize_haze(lr, t=haze_t, A=haze_A)
        
    # 3. Add Blur
    if blur_kernel > 0:
        lr = cv2.GaussianBlur(lr, (blur_kernel, blur_kernel), 0)
        
    # 4. Add Noise
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, lr.shape).astype(np.float32)
        lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    # Create the bicubic baseline
    bicubic_hr = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return lr, bicubic_hr

if __name__ == '__main__':
    # Simple test
    test_img = np.zeros((800, 800, 3), dtype=np.uint8)
    lr, bicubic = degrade_image(test_img)
    print(f"Original: {test_img.shape}, LR: {lr.shape}, Bicubic: {bicubic.shape}")
