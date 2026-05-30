import os
import cv2
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from sr_model import ESPCN
from degrade import degrade_image

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')
ORIGINAL_DIR = os.path.join(DATA_DIR, 'original')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')
EVAL_LIST = os.path.join(DATA_DIR, 'eval_images.txt')
WEIGHTS_DIR = os.path.join(WORKSPACE_DIR, 'weights')

EVAL_DIR = os.path.join(DATA_DIR, 'eval')

# Class mapping from DIOR
NAMES = {
    0: 'Expressway-Service-area', 1: 'Expressway-toll-station', 2: 'airplane', 3: 'airport',
    4: 'baseballfield', 5: 'basketballcourt', 6: 'bridge', 7: 'chimney', 8: 'dam', 9: 'golffield',
    10: 'groundtrackfield', 11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium', 15: 'storagetank',
    16: 'tenniscourt', 17: 'trainstation', 18: 'vehicle', 19: 'windmill'
}

def setup_eval_dirs():
    dirs = {
        'original': os.path.join(EVAL_DIR, 'original'),
        'degraded': os.path.join(EVAL_DIR, 'degraded'),
        'sr_enhanced': os.path.join(EVAL_DIR, 'sr_enhanced')
    }
    
    for name, path in dirs.items():
        img_dir = os.path.join(path, 'images')
        lbl_dir = os.path.join(path, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
    return dirs

def generate_eval_data(dirs):
    if not os.path.exists(EVAL_LIST):
        print(f"Eval list {EVAL_LIST} not found.")
        return False
        
    with open(EVAL_LIST, 'r') as f:
        eval_images = [line.strip() for line in f if line.strip()]
        
    if not eval_images:
        print("No eval images found.")
        return False
        
    device = torch.device('cpu')
    model_path = os.path.join(WEIGHTS_DIR, 'espcn_dior.pt')
    
    escpn = ESPCN(upscale_factor=4).to(device)
    if os.path.exists(model_path):
        escpn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        escpn.eval()
    else:
        print(f"Warning: ESPCN weights not found at {model_path}. Using untrained model.")
        
    print(f"Generating evaluation sets for {len(eval_images)} images...")
    
    for img_name in tqdm(eval_images):
        img_path = os.path.join(ORIGINAL_DIR, img_name)
        lbl_path = os.path.join(LABELS_DIR, img_name.replace('.jpg', '.txt'))
        
        hr_img = cv2.imread(img_path)
        if hr_img is None: continue
        
        # Original (just copy)
        cv2.imwrite(os.path.join(dirs['original'], 'images', img_name), hr_img)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(dirs['original'], 'labels', img_name.replace('.jpg', '.txt')))
            
        # Degraded
        lr_img, bicubic_img = degrade_image(hr_img, scale_factor=4, noise_sigma=15, blur_kernel=5, apply_haze=True)
        cv2.imwrite(os.path.join(dirs['degraded'], 'images', img_name), bicubic_img)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(dirs['degraded'], 'labels', img_name.replace('.jpg', '.txt')))
            
        # SR Enhanced
        # Convert LR to tensor
        lr_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            sr_tensor = escpn(lr_tensor.to(device))
            
        sr_rgb = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(dirs['sr_enhanced'], 'images', img_name), sr_bgr)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(dirs['sr_enhanced'], 'labels', img_name.replace('.jpg', '.txt')))
            
    return True

def create_yaml(name, path_dir):
    yaml_path = os.path.join(EVAL_DIR, f"{name}.yaml")
    
    yaml_content = f"""
path: {os.path.abspath(path_dir)}
train: images
val: images

names:
"""
    for k, v in NAMES.items():
        yaml_content += f"  {k}: {v}\n"
        
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    return yaml_path

def run_evaluation(dirs):
    yolo_model_path = os.path.join(WEIGHTS_DIR, 'yolov8_dior.pt')
    if not os.path.exists(yolo_model_path):
        print(f"YOLO weights not found at {yolo_model_path}.")
        return
        
    # Set YOLO to be completely quiet and suppress progress bars so it doesn't clutter output
    os.environ['YOLO_VERBOSE'] = 'False'
    model = YOLO(yolo_model_path)
    
    benchmarks = {}
    
    for name, dir_path in dirs.items():
        yaml_path = create_yaml(name, dir_path)
        print(f"\nEvaluating YOLOv8 on {name} dataset...")
        
        # Provide verbose=False to suppress the excessive output
        results = model.val(data=yaml_path, imgsz=800, split='val', device='cpu', verbose=False, plots=False)
        
        metrics = results.results_dict
        # Extract mAP50 and mAP50-95
        map50 = metrics.get('metrics/mAP50(B)', 0.0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)
        precision = metrics.get('metrics/precision(B)', 0.0)
        recall = metrics.get('metrics/recall(B)', 0.0)
        
        benchmarks[name] = {
            'mAP50': float(map50),
            'mAP50_95': float(map50_95),
            'precision': float(precision),
            'recall': float(recall)
        }
        
        print(f"{name} Results -> mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}")
        
    # Save benchmarks
    with open(os.path.join(DATA_DIR, 'benchmark_results.json'), 'w') as f:
        json.dump(benchmarks, f, indent=4)
        
    print("\nBenchmark saved to data/benchmark_results.json")

if __name__ == '__main__':
    dirs = setup_eval_dirs()
    if generate_eval_data(dirs):
        run_evaluation(dirs)
