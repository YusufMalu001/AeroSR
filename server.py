import os
import cv2
import json
import base64
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

from sr_model import ESPCN
from degrade import degrade_image

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')
EVAL_ORIGINAL_DIR = os.path.join(DATA_DIR, 'eval', 'original', 'images')
WEIGHTS_DIR = os.path.join(WORKSPACE_DIR, 'weights')

app = FastAPI()

# Mount frontend directory
frontend_dir = os.path.join(WORKSPACE_DIR, 'frontend')
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Load Models
device = torch.device('cpu')
espcn_model = None
yolo_model = None

@app.on_event("startup")
def load_models():
    global espcn_model, yolo_model
    
    # Load ESPCN
    espcn_path = os.path.join(WEIGHTS_DIR, 'espcn_dior.pt')
    espcn_model = ESPCN(upscale_factor=4).to(device)
    if os.path.exists(espcn_path):
        espcn_model.load_state_dict(torch.load(espcn_path, map_location=device, weights_only=True))
        espcn_model.eval()
        print("ESPCN model loaded.")
    else:
        print("Warning: ESPCN weights not found!")
        
    # Load YOLO
    yolo_path = os.path.join(WEIGHTS_DIR, 'yolov8_dior.pt')
    if os.path.exists(yolo_path):
        os.environ['YOLO_VERBOSE'] = 'False'
        yolo_model = YOLO(yolo_path)
        print("YOLOv8 model loaded.")
    else:
        print("Warning: YOLOv8 weights not found!")

class ProcessRequest(BaseModel):
    image_name: str
    noise_sigma: float = 15.0
    blur_kernel: int = 5
    haze_t: float = 0.85

def run_yolo(image_bgr):
    if not yolo_model:
        return []
        
    # We set conf=0.25 to see more detections (often degraded images drop confidence)
    results = yolo_model.predict(source=image_bgr, imgsz=800, conf=0.25, verbose=False)
    
    boxes = []
    if results and len(results) > 0:
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = result.names[cls_id]
            
            boxes.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': conf, 'class': name
            })
    return boxes

def encode_image(image_bgr):
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/api/images")
def list_images():
    if not os.path.exists(EVAL_ORIGINAL_DIR):
        return []
    images = [f for f in os.listdir(EVAL_ORIGINAL_DIR) if f.endswith('.jpg')]
    return sorted(images)[:20] # Return top 20 for the UI dropdown

@app.post("/api/process")
def process_image(req: ProcessRequest):
    img_path = os.path.join(EVAL_ORIGINAL_DIR, req.image_name)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    hr_img = cv2.imread(img_path)
    if hr_img is None:
        raise HTTPException(status_code=500, detail="Failed to read image")
        
    # 1. Original
    orig_boxes = run_yolo(hr_img)
    
    # 2. Degrade
    lr_img, bicubic_img = degrade_image(
        hr_img, scale_factor=4,
        noise_sigma=req.noise_sigma,
        blur_kernel=req.blur_kernel,
        apply_haze=True,
        haze_t=req.haze_t
    )
    bicubic_boxes = run_yolo(bicubic_img)
    
    # 3. SR Enhance
    lr_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
    lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    with torch.no_grad():
        sr_tensor = espcn_model(lr_tensor.to(device))
        
    sr_rgb = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
    
    sr_boxes = run_yolo(sr_bgr)
    
    # Also generate the LR image directly encoded (but resized up so it matches display size)
    h, w = hr_img.shape[:2]
    degraded_visual = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_NEAREST)
    degraded_boxes = run_yolo(degraded_visual)
    
    return {
        'original': {'image': encode_image(hr_img), 'boxes': orig_boxes},
        'degraded': {'image': encode_image(degraded_visual), 'boxes': degraded_boxes},
        'bicubic': {'image': encode_image(bicubic_img), 'boxes': bicubic_boxes},
        'sr': {'image': encode_image(sr_bgr), 'boxes': sr_boxes}
    }

@app.get("/api/benchmarks")
def get_benchmarks():
    bm_file = os.path.join(DATA_DIR, 'benchmark_results.json')
    loss_file = os.path.join(DATA_DIR, 'loss_history.json')
    
    benchmarks = {}
    if os.path.exists(bm_file):
        with open(bm_file, 'r') as f:
            benchmarks = json.load(f)
            
    loss = []
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            loss = json.load(f)
            
    return {
        'benchmarks': benchmarks,
        'loss_history': loss
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
