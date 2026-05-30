import os
import zipfile
import shutil
import random
import requests
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- Configuration ---
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
ORIGINAL_DIR = os.path.join(DATA_DIR, 'original')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')
WEIGHTS_DIR = os.path.join(WORKSPACE_DIR, 'weights')

DOWNLOADS_DIR = os.path.expanduser('~\\Downloads')
EXISTING_YOLO_ZIP = os.path.join(DOWNLOADS_DIR, 'Object_Detection_Satellite_Imagery_Yolov8_DIOR-main.zip')

ANNOTATIONS_ID = '1KoQzqR20qvIXDf1qsXCHGxD003IPmXMw'
TEST_IMAGES_ID = '11SXPqcESez9qTn4Z5Q3v35K9hRwO_epr'

ANNOTATIONS_ZIP = os.path.join(RAW_DIR, 'Annotations.zip')
TEST_IMAGES_ZIP = os.path.join(RAW_DIR, 'JPEGImages-test.zip')

TOTAL_IMAGES = 300
TRAIN_IMAGES = 200

# DIOR Classes
CLASS_NAMES = [
    'Expressway-Service-area', 'Expressway-toll-station', 'airplane', 'airport', 'baseballfield',
    'basketballcourt', 'bridge', 'chimney', 'dam', 'golffield', 'groundtrackfield', 'harbor',
    'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]
CLASS_DICT = {name: i for i, name in enumerate(CLASS_NAMES)}

def setup_dirs():
    for d in [DATA_DIR, RAW_DIR, ORIGINAL_DIR, LABELS_DIR, WEIGHTS_DIR]:
        os.makedirs(d, exist_ok=True)

def extract_yolo_weights():
    target_weight = os.path.join(WEIGHTS_DIR, 'yolov8_dior.pt')
    if os.path.exists(target_weight):
        print("YOLOv8 weights already exist.")
        return

    print(f"Extracting best.pt from {EXISTING_YOLO_ZIP}...")
    if not os.path.exists(EXISTING_YOLO_ZIP):
        print(f"Error: {EXISTING_YOLO_ZIP} not found!")
        return

    with zipfile.ZipFile(EXISTING_YOLO_ZIP, 'r') as z:
        source_path = 'Object_Detection_Satellite_Imagery_Yolov8_DIOR-main/runs/detect/yolov8n_epochs50_batch16/weights/best.pt'
        with z.open(source_path) as source, open(target_weight, 'wb') as target:
            shutil.copyfileobj(source, target)
    print("YOLOv8 weights extracted.")

def download_gdrive_file(file_id, dest_path):
    if os.path.exists(dest_path):
        print(f"File {os.path.basename(dest_path)} already exists. Skipping download.")
        return True
        
    print(f"Downloading {os.path.basename(dest_path)}...")
    session = requests.Session()
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    r = session.get(url)
    
    if r.status_code != 200:
        print(f"Failed to fetch initial URL: {r.status_code}")
        return False
        
    html = r.text
    action_match = re.search(r'action="([^"]*)"', html)
    if action_match:
        action = action_match.group(1)
        params = {}
        for name, value in re.findall(r'<input type="hidden" name="([^"]*)" value="([^"]*)"', html):
            params[name] = value
        if 'confirm' not in params:
            params['confirm'] = 't'
        r = session.get(action, params=params, stream=True)
    else:
        r = session.get(url, stream=True)
        
    if r.status_code != 200:
        print(f"Download request failed with status: {r.status_code}")
        return False
        
    with open(dest_path, "wb") as f:
        # Use simple progress bar for download
        # Total size may not be available in headers for chunked transfer
        total_size = int(r.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
    print(f"Downloaded successfully.")
    return True

def extract_subset():
    if len(os.listdir(ORIGINAL_DIR)) >= TOTAL_IMAGES:
        print("Subset already extracted.")
        return

    print("Extracting a random subset of test images and annotations...")
    
    # 1. Get all image paths from the test zip
    with zipfile.ZipFile(TEST_IMAGES_ZIP, 'r') as z_img:
        all_img_files = [f for f in z_img.namelist() if f.endswith('.jpg')]
        random.seed(42) # For reproducibility
        selected_imgs = random.sample(all_img_files, min(TOTAL_IMAGES, len(all_img_files)))
        
        # Extract images
        for img_path in tqdm(selected_imgs, desc="Extracting Images"):
            img_name = os.path.basename(img_path)
            target_path = os.path.join(ORIGINAL_DIR, img_name)
            with z_img.open(img_path) as source, open(target_path, 'wb') as target:
                shutil.copyfileobj(source, target)

    # 2. Extract corresponding annotations
    selected_basenames = [os.path.basename(f).replace('.jpg', '') for f in selected_imgs]
    
    with zipfile.ZipFile(ANNOTATIONS_ZIP, 'r') as z_ann:
        all_ann_files = [f for f in z_ann.namelist() if f.endswith('.xml')]
        
        for ann_path in tqdm(all_ann_files, desc="Processing Annotations"):
            ann_name = os.path.basename(ann_path)
            basename = ann_name.replace('.xml', '')
            
            if basename in selected_basenames:
                # Parse XML directly from zip
                with z_ann.open(ann_path) as source:
                    xml_content = source.read()
                    
                root = ET.fromstring(xml_content)
                size_elem = root.find('size')
                if size_elem is None:
                    continue
                    
                width = float(size_elem.find('width').text)
                height = float(size_elem.find('height').text)
                
                yolo_lines = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in CLASS_DICT:
                        continue
                    class_id = CLASS_DICT[class_name]
                    
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    # YOLO Format: class x_center y_center width height (normalized)
                    x_center = ((xmin + xmax) / 2.0) / width
                    y_center = ((ymin + ymax) / 2.0) / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height
                    
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
                # Write YOLO txt file
                txt_path = os.path.join(LABELS_DIR, f"{basename}.txt")
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))

def split_dataset():
    images = sorted([f for f in os.listdir(ORIGINAL_DIR) if f.endswith('.jpg')])
    if not images:
        return
        
    random.seed(42)
    random.shuffle(images)
    
    train_imgs = images[:TRAIN_IMAGES]
    eval_imgs = images[TRAIN_IMAGES:]
    
    with open(os.path.join(DATA_DIR, 'train_images.txt'), 'w') as f:
        f.write('\n'.join(train_imgs))
        
    with open(os.path.join(DATA_DIR, 'eval_images.txt'), 'w') as f:
        f.write('\n'.join(eval_imgs))
        
    print(f"Split dataset: {len(train_imgs)} train, {len(eval_imgs)} eval.")

if __name__ == '__main__':
    setup_dirs()
    extract_yolo_weights()
    download_gdrive_file(ANNOTATIONS_ID, ANNOTATIONS_ZIP)
    download_gdrive_file(TEST_IMAGES_ID, TEST_IMAGES_ZIP)
    extract_subset()
    split_dataset()
    print("Dataset preparation complete!")
